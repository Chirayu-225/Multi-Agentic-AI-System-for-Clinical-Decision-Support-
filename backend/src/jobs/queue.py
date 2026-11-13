r"""
src/jobs/queue.py

A small durable job queue on top of the application database.

Why the database and not Celery/Redis: the app already needs PostgreSQL for its transactions,
and a job here is created in the same place and the same moment as the rest of the upload's
state. One system to run, back up and reason about; the queue survives a restart for free;
and the throughput this workload needs (a report takes tens of seconds of OCR + model calls,
so a handful of jobs per minute) is orders of magnitude below what a table with the right
index can claim. Where this stops being the right choice: sustained hundreds of jobs/second,
or fan-out across many machines that should not all hit the primary - then move to a broker.
The functions below are the only place that knows how jobs are stored, so that swap is local.

Delivery guarantee: AT LEAST ONCE. A job is handed out, a lease (a deadline) is attached, and
the worker keeps extending it while it works. If the worker dies, the lease runs out and the
job is handed to someone else. Because the same job can therefore run twice, the code that does
the work must be idempotent (the upload pipeline's final write is: the report id is a primary
key, and a re-run that finds the report already saved just finishes the job).

States:   queued --claim--> running --done--> complete
                              |  \--error--> queued (retry after backoff)  | failed (out of attempts)
                              \--lease expired--> claimable again (or failed if out of attempts)

Claiming: SELECT ... FOR UPDATE SKIP LOCKED on PostgreSQL (concurrent workers never block each
other and never pick the same row), followed on every database by a compare-and-swap UPDATE whose
WHERE clause re-checks eligibility; the row count says whether this worker won. SQLite ignores
FOR UPDATE, and the compare-and-swap alone keeps it correct there.

Every transition made by a worker is guarded by `locked_by = <that worker>`, so a worker that
lost its lease (paused, partitioned) cannot overwrite the state of the worker that took over.
"""
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.models import UploadJob

logger = logging.getLogger("medsight.jobs")

QUEUED, RUNNING, COMPLETE, FAILED = "queued", "running", "complete", "failed"
TERMINAL = (COMPLETE, FAILED)


class DuplicateJob(Exception):
    """A job with this id already exists."""


class PermanentJobError(Exception):
    """Raise from the job body when retrying cannot help (a corrupt file, say). The job is
    failed immediately instead of burning the remaining attempts."""


def utcnow() -> datetime:
    # Naive UTC, matching the rest of the schema (DateTime columns without a timezone).
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_worker_id() -> str:
    return f"w-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class JobSnapshot:
    """An immutable copy of a job row, safe to hand to the pipeline after the session closes."""
    id: str
    owner: str
    ip: str | None
    patient_id: str
    filename: str | None
    ext: str | None
    report_date: str | None
    language: str
    spool_path: str
    attempts: int
    max_attempts: int
    worker_id: str | None = None

    @property
    def is_last_attempt(self) -> bool:
        return self.attempts >= self.max_attempts


def _snapshot(row: UploadJob) -> JobSnapshot:
    return JobSnapshot(
        id=row.id, owner=row.owner, ip=row.ip, patient_id=row.patient_id, filename=row.filename,
        ext=row.ext, report_date=row.report_date, language=row.language or "en",
        spool_path=row.spool_path, attempts=row.attempts, max_attempts=row.max_attempts,
        worker_id=row.locked_by,
    )


def job_view(row: UploadJob) -> dict:
    """What the API shows about a job. No path, no owner, no traceback."""
    return {"job_id": row.id, "report_id": row.id, "status": row.status, "stage": row.stage,
            "percent": row.percent, "error": row.error, "attempt": row.attempts,
            "max_attempts": row.max_attempts}


# ── Producer side ──────────────────────────────────────────────────────────────

def enqueue(db: Session, *, job_id: str, owner: str, ip: str | None, patient_id: str, filename: str,
            ext: str, report_date: str, language: str, spool_path: str, max_attempts: int = 3,
            now: datetime | None = None) -> UploadJob:
    now = now or utcnow()
    row = UploadJob(id=job_id, owner=owner, ip=ip, patient_id=patient_id, filename=filename, ext=ext,
                    report_date=report_date, language=language, spool_path=spool_path,
                    status=QUEUED, stage=QUEUED, percent=0, attempts=0, max_attempts=max_attempts,
                    next_attempt_at=now, created_at=now)
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:       # primary key: same id twice, including a concurrent double-submit
        db.rollback()
        raise DuplicateJob(job_id) from exc
    return row


def get(db: Session, job_id: str) -> UploadJob | None:
    return db.get(UploadJob, job_id)


def depth_by_status(db: Session) -> dict[str, int]:
    counts = {QUEUED: 0, RUNNING: 0, COMPLETE: 0, FAILED: 0}
    for status, n in db.query(UploadJob.status, func.count()).group_by(UploadJob.status):
        counts[status] = n
    return counts


def active_count(db: Session) -> int:
    return db.query(func.count()).select_from(UploadJob).filter(UploadJob.status.in_((QUEUED, RUNNING))).scalar() or 0


# ── Consumer side ──────────────────────────────────────────────────────────────

def _eligible(now: datetime):
    """Claimable: waiting and past its backoff, or 'running' under a lease that has expired."""
    return and_(
        UploadJob.attempts < UploadJob.max_attempts,
        or_(
            and_(UploadJob.status == QUEUED, UploadJob.next_attempt_at <= now),
            and_(UploadJob.status == RUNNING, UploadJob.lease_expires_at < now),
        ),
    )


def claim_next(db: Session, worker_id: str, lease_seconds: float, now: datetime | None = None) -> JobSnapshot | None:
    """Atomically take the oldest eligible job, or return None. Safe to call from many workers."""
    now = now or utcnow()
    for _ in range(5):      # lost races are normal under contention; try the next candidate
        candidate = (
            db.query(UploadJob.id)
            .filter(_eligible(now))
            .order_by(UploadJob.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
            .first()
        )
        if candidate is None:
            db.rollback()
            return None
        won = db.execute(
            update(UploadJob)
            .where(UploadJob.id == candidate[0], _eligible(now))
            .values(status=RUNNING, locked_by=worker_id, lease_expires_at=now + timedelta(seconds=lease_seconds),
                    attempts=UploadJob.attempts + 1, started_at=func.coalesce(UploadJob.started_at, now),
                    stage="starting", percent=0, error=None)
        ).rowcount
        db.commit()
        if won == 1:
            row = db.get(UploadJob, candidate[0])
            db.refresh(row)
            return _snapshot(row)
    return None


def reap_exhausted(db: Session, now: datetime | None = None) -> list[JobSnapshot]:
    """Jobs whose worker died during their LAST attempt: nobody will ever run them again, so
    mark them failed (a job must not sit in 'running' forever). Returns them so the caller can
    clean up their spool files and tell the client."""
    now = now or utcnow()
    rows = (db.query(UploadJob)
            .filter(UploadJob.status == RUNNING, UploadJob.lease_expires_at < now,
                    UploadJob.attempts >= UploadJob.max_attempts)
            .all())
    out = []
    for row in rows:
        won = db.execute(
            update(UploadJob)
            .where(UploadJob.id == row.id, UploadJob.status == RUNNING, UploadJob.lease_expires_at < now)
            .values(status=FAILED, finished_at=now, locked_by=None, lease_expires_at=None,
                    stage="failed", error="Processing was interrupted and could not be completed. Please upload again.")
        ).rowcount
        if won == 1:
            out.append(_snapshot(row))
    db.commit()
    return out


def heartbeat(db: Session, job_id: str, worker_id: str, lease_seconds: float, now: datetime | None = None) -> bool:
    """Extend the lease. False means this worker no longer owns the job (lease lost)."""
    now = now or utcnow()
    n = db.execute(
        update(UploadJob)
        .where(UploadJob.id == job_id, UploadJob.locked_by == worker_id, UploadJob.status == RUNNING)
        .values(lease_expires_at=now + timedelta(seconds=lease_seconds))
    ).rowcount
    db.commit()
    return n == 1


def set_progress(db: Session, job_id: str, worker_id: str, stage: str, percent: int) -> None:
    """Record progress. Never moves backwards, so late or reordered writes are harmless."""
    db.execute(
        update(UploadJob)
        .where(UploadJob.id == job_id, UploadJob.locked_by == worker_id, UploadJob.percent <= percent)
        .values(stage=stage, percent=percent)
    )
    db.commit()


def complete(db: Session, job_id: str, worker_id: str, now: datetime | None = None) -> bool:
    now = now or utcnow()
    n = db.execute(
        update(UploadJob)
        .where(UploadJob.id == job_id, UploadJob.locked_by == worker_id)
        .values(status=COMPLETE, stage="pipeline_complete", percent=100, finished_at=now, error=None,
                locked_by=None, lease_expires_at=None)
    ).rowcount
    db.commit()
    return n == 1


def backoff_seconds(attempts: int, base: float, cap: float = 300.0, jitter: bool = True) -> float:
    """Exponential backoff after the Nth failed attempt, with up to 25% jitter so that jobs which
    failed together (the model provider was down) do not all retry at the same instant."""
    delay = min(cap, base * (2 ** max(attempts - 1, 0)))
    return delay + (random.uniform(0, delay * 0.25) if jitter and delay else 0)


def fail_or_retry(db: Session, job_id: str, worker_id: str, error: str, *, retry_base_seconds: float,
                  permanent: bool = False, now: datetime | None = None) -> str | None:
    """After an attempt failed: requeue with backoff if attempts remain, else fail the job.
    Returns 'retry', 'failed', or None if this worker no longer owned the job."""
    now = now or utcnow()
    row = db.query(UploadJob).filter(UploadJob.id == job_id, UploadJob.locked_by == worker_id).first()
    if row is None:
        db.rollback()
        return None
    if permanent or row.attempts >= row.max_attempts:
        db.execute(update(UploadJob).where(UploadJob.id == job_id, UploadJob.locked_by == worker_id)
                   .values(status=FAILED, stage="failed", error=error, finished_at=now, locked_by=None,
                           lease_expires_at=None))
        db.commit()
        return FAILED
    delay = backoff_seconds(row.attempts, retry_base_seconds)
    db.execute(update(UploadJob).where(UploadJob.id == job_id, UploadJob.locked_by == worker_id)
               .values(status=QUEUED, stage="retrying", percent=0, locked_by=None, lease_expires_at=None,
                       next_attempt_at=now + timedelta(seconds=delay)))
    db.commit()
    return "retry"


def release(db: Session, job_id: str, worker_id: str, now: datetime | None = None) -> bool:
    """Give a job back on clean shutdown. The interrupted attempt does not count against the job."""
    now = now or utcnow()
    n = db.execute(
        update(UploadJob)
        .where(UploadJob.id == job_id, UploadJob.locked_by == worker_id, UploadJob.status == RUNNING)
        .values(status=QUEUED, stage="queued", percent=0, locked_by=None, lease_expires_at=None,
                next_attempt_at=now, attempts=UploadJob.attempts - 1)
    ).rowcount
    db.commit()
    return n == 1


def prune(db: Session, older_than: timedelta, now: datetime | None = None) -> int:
    """Delete finished jobs after the retention period. Returns how many."""
    now = now or utcnow()
    n = db.query(UploadJob).filter(UploadJob.status.in_(TERMINAL), UploadJob.finished_at < now - older_than).delete(
        synchronize_session=False)
    db.commit()
    return n
