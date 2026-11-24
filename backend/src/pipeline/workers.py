"""Starts and stops the background job worker and the outbox relay."""
import logging
import os
import uuid
from datetime import timedelta

from src.api.auth import ROLE_CLINICIAN
from src.core.config import env_num
from src.db.session import SessionLocal
from src.jobs import outbox, queue as jobq
from src.jobs.worker import JobWorker
from src.monitoring import observability as obs
from src.pipeline.history_sync import apply_history_event, outbox_config, outbox_result
from src.pipeline.progress import make_ws_callback
from src.pipeline.upload import process_upload
from src.security import audit

logger = logging.getLogger("medsight")


# ── Upload jobs ────────────────────────────────────────────────────────────────
# An upload is accepted fast (validated, written to the spool, queued as a row in `upload_jobs`)
# and analysed later by a JobWorker. The job id IS the report id (the client already opens
# /ws/{report_id} before uploading). The queue lives in the database, so a restart does not lose
# work: see src/jobs/queue.py for the delivery guarantee (at-least-once) and why the pipeline's
# final write is idempotent. Nothing in a job row holds PHI: ids, stage, percent, counters.

job_worker: JobWorker | None = None


def job_failure(job: jobq.JobSnapshot, exc: BaseException, final: bool) -> str:
    """Called by the worker for every failed attempt. The full detail (with traceback) goes to the
    server log under an incident id; the client only ever gets the generic message + that id, so
    internal paths and data are never reflected back."""
    incident_id = uuid.uuid4().hex[:8]
    logger.error("Upload failed [incident=%s] attempt=%d/%d final=%s report_id=%s",
                 incident_id, job.attempts, job.max_attempts, final, job.id, exc_info=exc)
    audit.record(actor=job.owner, role=ROLE_CLINICIAN, action="report.upload" if final else "report.upload_retry",
                 resource=f"report:{job.id}", outcome="failure", ip=job.ip,
                 detail={"incident": incident_id, "attempt": job.attempts, "final": final})
    return f"Processing failed. Reference: {incident_id}. Check server logs for detail."


def job_finished(job: jobq.JobSnapshot, status: str, error: str | None) -> None:
    """Terminal state reached: tell the browser and delete the uploaded file (PHI) from the spool."""
    ws = make_ws_callback(job.id)
    if status == jobq.COMPLETE:
        ws({"type": "pipeline_complete", "report_id": job.id})
        ws({"type": "progress", "report_id": job.id, "stage": "pipeline_complete", "percent": 100})
    else:
        ws({"type": "pipeline_error", "report_id": job.id, "detail": error})
    try:
        os.unlink(job.spool_path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("could not delete spool file for job %s", job.id)


# ── History-store outbox ───────────────────────────────────────────────────────
# The ChromaDB write cannot share the report's database transaction, so it is recorded as an
# outbox event in that transaction and applied afterwards with retries (src/jobs/outbox.py).

outbox_relay: "outbox.OutboxRelay | None" = None


async def start_outbox_relay() -> None:
    global outbox_relay
    outbox_relay = outbox.OutboxRelay(session_factory=SessionLocal, handler=apply_history_event,
                                config=outbox_config, on_result=outbox_result)
    await outbox_relay.start()

    def _depth() -> dict:
        db = SessionLocal()
        try:
            return outbox.depth_by_status(db)
        finally:
            db.close()

    obs.gauge("outbox_events", "History-store outbox events by status (database-wide).", _depth)


async def start_job_worker() -> None:
    global job_worker
    workers = env_num("JOB_WORKERS", 2, int)
    if workers <= 0:
        logger.info("JOB_WORKERS=0: this process only accepts uploads; another process must run the jobs")
        job_worker = None
    else:
        job_worker = JobWorker(
            session_factory=SessionLocal, process=process_upload,
            on_failure=job_failure, on_finished=job_finished,
            concurrency=workers,
            lease_seconds=env_num("JOB_LEASE_SECONDS", 60.0),
            poll_seconds=env_num("JOB_POLL_SECONDS", 1.0),
            retry_base_seconds=env_num("JOB_RETRY_BASE_SECONDS", 15.0),
            retention=timedelta(hours=env_num("JOB_RETENTION_HOURS", 24.0)),
        )
        await job_worker.start()

    def _depth() -> dict:
        db = SessionLocal()
        try:
            return jobq.depth_by_status(db)
        finally:
            db.close()

    obs.gauge("upload_jobs", "Upload jobs by status (database-wide).", _depth)
