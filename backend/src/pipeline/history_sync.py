"""Keeps the vector store (ChromaDB) in step with the database through the outbox."""
import logging

from src.core.config import env_num
from src.db.models import Patient, Report
from src.db.session import SessionLocal
from src.jobs import outbox
from src.tools.history_store import delete_patient_history, delete_report_history, store_report as chroma_store

logger = logging.getLogger("medsight")


HISTORY_PENDING = "History update pending: "


def outbox_config() -> dict:
    return {
        "lease_seconds": env_num("OUTBOX_LEASE_SECONDS", 120.0),
        "max_attempts": env_num("OUTBOX_MAX_ATTEMPTS", 8, int),
        "base_seconds": env_num("OUTBOX_RETRY_BASE_SECONDS", 30.0),
        "poll_seconds": env_num("OUTBOX_POLL_SECONDS", 5.0),
        "retention_hours": env_num("OUTBOX_RETENTION_HOURS", 24.0),
    }


def history_text(redacted_text: str, words: int = 512) -> str:
    return " ".join(redacted_text.split()[:words])      # what the vector store embeds, and no more


def apply_history_event(ev: dict) -> str | None:
    """Idempotent: chroma_store upserts by report id. Skips reports that were rejected or erased
    while the event waited, and undoes its own write if that happened while it was writing."""
    def _state():
        db = SessionLocal()
        try:
            r = db.query(Report.review_status).filter(Report.id == ev["report_id"]).first()
            patient_exists = db.query(Patient.id).filter(Patient.id == ev["patient_id"]).first() is not None
            return r, patient_exists
        finally:
            db.close()

    def _undo():
        r, patient_exists = _state()
        if r is not None and r.review_status != "rejected":
            return
        try:
            if patient_exists:
                delete_report_history(ev["patient_id"], ev["report_id"])
            else:
                delete_patient_history(ev["patient_id"])
        except Exception:  # noqa: BLE001
            logger.exception("could not undo a history write for a rejected/erased report")

    r, _ = _state()
    if r is None or r.review_status == "rejected":
        return "skipped"
    p = ev["payload"]
    chroma_store(patient_id=ev["patient_id"], report_text=p.get("text", ""),
                 lab_values=p.get("lab_values", []), report_date=p.get("report_date", ""),
                 report_id=ev["report_id"])
    _undo()      # a rejection or erasure that landed during the write must still win
    return None


def outbox_result(ev: dict, outcome: str, error: str | None) -> None:
    """Keep the report's warning list honest about whether it is in the patient's history."""
    if ev["kind"] != outbox.HISTORY_INDEX:
        return
    db = SessionLocal()
    try:
        r = db.query(Report).filter(Report.id == ev["report_id"]).first()
        if r is None:
            return
        keep = [w for w in (r.ingestion_warnings or [])
                if not w.startswith(HISTORY_PENDING) and "could not be added to the patient's history" not in w]
        if outcome == "retry":
            keep.append(f"{HISTORY_PENDING}this report is not in the patient's history yet ({error}); it will be "
                        f"retried automatically. Until then trend and baseline checks do not include it.")
        elif outcome == "dead":
            keep.append(f"This report could not be added to the patient's history after repeated attempts ({error}), "
                        f"so it will not appear in future trend or baseline checks.")
        if keep != list(r.ingestion_warnings or []):
            r.ingestion_warnings = keep
            db.commit()
    except Exception:  # noqa: BLE001 - bookkeeping only
        db.rollback()
        logger.exception("could not update history-status warning")
    finally:
        db.close()
