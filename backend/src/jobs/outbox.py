"""
src/jobs/outbox.py

Transactional outbox for the one write that cannot share the database transaction: the ChromaDB
history store.

  add()           put the event in the SAME transaction as the report row.
  drain()         claim due events (compare-and-swap, so several processes can run it), apply
                  them with a handler, and either finish them, back them off, or give up (dead).
  OutboxRelay     the background loop that calls drain() on an interval.

Delivery is at-least-once: a crash between "applied" and "marked done" repeats the event, so the
handler must be idempotent. A claimed event is hidden by pushing `next_attempt_at` out by a lease;
if the process dies, the event simply becomes due again.
"""
import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import update
from sqlalchemy.orm import Session

from src.monitoring import observability as obs
from src.db.models import OutboxEvent

logger = logging.getLogger("medsight.outbox")

PENDING, DONE, DEAD, CANCELLED = "pending", "done", "dead", "cancelled"
HISTORY_INDEX = "history.index"

obs.describe("outbox_applied_total", "Outbox events applied, by outcome (done, retry, dead, skipped).")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def add(db: Session, *, kind: str, report_id: str, patient_id: str, payload: dict,
        now: datetime | None = None) -> OutboxEvent:
    """Stage an event. Does NOT commit: the caller's commit makes the event and the data it
    describes appear together, or not at all."""
    now = now or utcnow()
    ev = OutboxEvent(kind=kind, report_id=report_id, patient_id=patient_id, payload=payload,
                     status=PENDING, attempts=0, next_attempt_at=now, created_at=now)
    db.add(ev)
    return ev


def backoff_seconds(attempts: int, base: float, cap: float = 900.0) -> float:
    delay = min(cap, base * (2 ** max(attempts - 1, 0)))
    return delay + random.uniform(0, delay * 0.25)


def claim_due(db: Session, *, lease_seconds: float, limit: int = 20, report_id: str | None = None,
              now: datetime | None = None) -> list[dict]:
    """Claim up to `limit` due events. Each claim is a conditional UPDATE, so of any number of
    concurrent claimers exactly one wins a given event."""
    now = now or utcnow()
    q = db.query(OutboxEvent.id).filter(OutboxEvent.status == PENDING, OutboxEvent.next_attempt_at <= now)
    if report_id:
        q = q.filter(OutboxEvent.report_id == report_id)
    claimed: list[dict] = []
    for (eid,) in q.order_by(OutboxEvent.id).limit(limit).all():
        won = db.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == eid, OutboxEvent.status == PENDING, OutboxEvent.next_attempt_at <= now)
            .values(next_attempt_at=now + timedelta(seconds=lease_seconds), attempts=OutboxEvent.attempts + 1)
        ).rowcount
        db.commit()
        if won != 1:
            continue
        row = db.get(OutboxEvent, eid)
        db.refresh(row)
        claimed.append({"id": row.id, "kind": row.kind, "report_id": row.report_id,
                        "patient_id": row.patient_id, "payload": row.payload or {}, "attempts": row.attempts})
    return claimed


def mark_done(db: Session, event_id: int, now: datetime | None = None) -> None:
    db.execute(update(OutboxEvent).where(OutboxEvent.id == event_id, OutboxEvent.status == PENDING)
               .values(status=DONE, done_at=now or utcnow(), last_error=None, payload=None))
    db.commit()


def mark_failed(db: Session, event_id: int, error: str, *, attempts: int, max_attempts: int,
                base_seconds: float, now: datetime | None = None) -> str:
    """Back off, or give up (dead) once attempts are used up. Returns 'retry' or 'dead'."""
    now = now or utcnow()
    error = error[:500]
    if attempts >= max_attempts:
        db.execute(update(OutboxEvent).where(OutboxEvent.id == event_id, OutboxEvent.status == PENDING)
                   .values(status=DEAD, last_error=error, done_at=now))
        db.commit()
        return DEAD
    db.execute(update(OutboxEvent).where(OutboxEvent.id == event_id, OutboxEvent.status == PENDING)
               .values(last_error=error, next_attempt_at=now + timedelta(seconds=backoff_seconds(attempts, base_seconds))))
    db.commit()
    return "retry"


def cancel_for_report(db: Session, report_id: str) -> int:
    """Stop pending events for a report (it was rejected). Does not commit."""
    return db.execute(update(OutboxEvent).where(OutboxEvent.report_id == report_id, OutboxEvent.status == PENDING)
                      .values(status=CANCELLED, payload=None)).rowcount


def delete_for_patient(db: Session, patient_id: str) -> int:
    """Erase every event for a patient, whatever its state. Does not commit."""
    return db.query(OutboxEvent).filter(OutboxEvent.patient_id == patient_id).delete(synchronize_session=False)


def depth_by_status(db: Session) -> dict[str, int]:
    from sqlalchemy import func
    rows = db.query(OutboxEvent.status, func.count()).group_by(OutboxEvent.status).all()
    out = {PENDING: 0, DONE: 0, DEAD: 0, CANCELLED: 0}
    out.update({s: n for s, n in rows})
    return out


def prune(db: Session, older_than: timedelta, now: datetime | None = None) -> int:
    now = now or utcnow()
    n = db.query(OutboxEvent).filter(OutboxEvent.status.in_((DONE, CANCELLED)),
                                     OutboxEvent.done_at < now - older_than).delete(synchronize_session=False)
    n2 = db.query(OutboxEvent).filter(OutboxEvent.status == CANCELLED, OutboxEvent.done_at.is_(None),
                                      OutboxEvent.created_at < now - older_than).delete(synchronize_session=False)
    db.commit()
    return n + n2


def drain(session_factory, handler: Callable[[dict], str | None], *, lease_seconds: float = 120.0,
          max_attempts: int = 8, base_seconds: float = 30.0, limit: int = 20,
          report_id: str | None = None, on_result: Callable[[dict, str, str | None], None] | None = None) -> dict:
    """Apply due events once. `handler(event)` raises to fail an attempt and may return "skipped"
    (nothing to do any more, e.g. the report was rejected). Blocking: run it in a thread.
    Returns counts by outcome."""
    counts = {"done": 0, "skipped": 0, "retry": 0, "dead": 0}
    db = session_factory()
    try:
        events = claim_due(db, lease_seconds=lease_seconds, limit=limit, report_id=report_id)
        for ev in events:
            try:
                outcome = handler(ev) or "done"
                mark_done(db, ev["id"])
                counts["skipped" if outcome == "skipped" else "done"] += 1
                obs.inc("outbox_applied_total", outcome=outcome)
                if on_result:
                    on_result(ev, outcome, None)
            except Exception as exc:  # noqa: BLE001 - one bad event must not block the rest
                db.rollback()
                status = mark_failed(db, ev["id"], f"{type(exc).__name__}: {exc}", attempts=ev["attempts"],
                                     max_attempts=max_attempts, base_seconds=base_seconds)
                counts[status] += 1
                obs.inc("outbox_applied_total", outcome=status)
                logger.warning("outbox event %s (%s) failed attempt %d/%d: %s", ev["id"], ev["kind"],
                               ev["attempts"], max_attempts, type(exc).__name__)
                if on_result:
                    on_result(ev, status, f"{type(exc).__name__}")
        return counts
    finally:
        db.close()


class OutboxRelay:
    """Background loop: drain the outbox every `poll_seconds`, plus housekeeping.
    `config()` is read on every pass (lease_seconds, max_attempts, base_seconds, poll_seconds,
    retention_hours), so settings can change without a restart."""

    def __init__(self, *, session_factory, handler, config: Callable[[], dict], on_result=None):
        self._sf, self._handler, self._config, self._on_result = session_factory, handler, config, on_result
        self._task: asyncio.Task | None = None
        self._last_prune = 0.0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="outbox-relay")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            cfg = self._config()
            try:
                await asyncio.to_thread(drain, self._sf, self._handler, lease_seconds=cfg["lease_seconds"],
                                        max_attempts=cfg["max_attempts"], base_seconds=cfg["base_seconds"],
                                        on_result=self._on_result)
                if loop.time() - self._last_prune > 600:
                    self._last_prune = loop.time()
                    await asyncio.to_thread(self._prune, timedelta(hours=cfg["retention_hours"]))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("outbox relay pass failed")
            await asyncio.sleep(cfg["poll_seconds"])

    def _prune(self, retention: timedelta) -> None:
        db = self._sf()
        try:
            prune(db, retention)
        finally:
            db.close()
