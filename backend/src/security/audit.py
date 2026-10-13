"""
src/security/audit.py

Tamper-evident audit trail (OWASP LLM / general security logging).

Every security-relevant event - login success/failure, upload, report view,
clinician approval, export, delete, authorization denial, injection detection,
rate-limit hit - is appended to `audit_events` with a hash that covers the
event's content AND the previous event's hash:

    hash_n = SHA-256( hash_(n-1) || canonical_json(event_n) )

Edit any row and its own hash no longer matches; delete a row in the middle and
the next row's `prev_hash` no longer matches. verify_chain() walks the table and
reports the first broken link.

What this does NOT give you (honest limits):
  * Truncating the END of the log (deleting the newest rows) is undetectable from
    inside the database alone. verify_chain() returns the current head hash;
    periodically copy that value somewhere the DB admin cannot write (a signed
    commit, an email, a separate log sink) to close that gap.
  * Someone with direct DB write access can recompute the whole chain. This
    protects against casual tampering and application bugs, not a determined
    DBA - that needs an external append-only sink.

PHI rule: `detail` carries identifiers, counts and reason codes only - never
report text, lab values, names or summaries.
"""
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone

from src.db.models import AuditEvent

logger = logging.getLogger("medsight.audit")

GENESIS = "GENESIS"
_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("AUDIT_ENABLED", "true").lower() != "false"


def _canonical(event: dict) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(prev_hash: str, event: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(event)).encode("utf-8")).hexdigest()


def _event_payload(row) -> dict:
    return {
        "ts": row.ts, "actor": row.actor, "role": row.role, "action": row.action,
        "resource": row.resource, "outcome": row.outcome, "ip": row.ip, "detail": row.detail,
    }


def record(*, actor: str, action: str, role: str | None = None, resource: str = "",
           outcome: str = "success", detail: dict | None = None, ip: str | None = None,
           critical: bool = False) -> None:
    """
    Append one event. Uses its OWN database session so the event is written even
    if the request's own transaction fails or is rolled back (a failed attempt is
    exactly what an audit log needs to capture).

    critical=True: if the event cannot be written, raise instead of continuing -
    use for actions that must never happen unaudited (approvals, deletes).
    """
    if not _enabled():
        return
    from src.db.session import SessionLocal  # local import: avoids a cycle at import time
    try:
        with _lock:
            db = SessionLocal()
            try:
                last = db.query(AuditEvent).order_by(AuditEvent.id.desc()).first()
                prev = last.hash if last else GENESIS
                ev = {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    "actor": actor, "role": role, "action": action, "resource": resource,
                    "outcome": outcome, "ip": ip, "detail": detail or {},
                }
                db.add(AuditEvent(**ev, prev_hash=prev, hash=compute_hash(prev, ev)))
                db.commit()
            finally:
                db.close()
    except Exception:
        logger.exception("AUDIT WRITE FAILED action=%s outcome=%s", action, outcome)
        if critical:
            raise


def verify_chain(db) -> dict:
    """Recompute every hash in order. Returns
    {ok, checked, first_bad_id, reason, head_hash}."""
    prev = GENESIS
    checked = 0
    for row in db.query(AuditEvent).order_by(AuditEvent.id.asc()).yield_per(500):
        if row.prev_hash != prev:
            return {"ok": False, "checked": checked, "first_bad_id": row.id,
                    "reason": "prev_hash does not match previous event (row deleted or reordered)",
                    "head_hash": None}
        if row.hash != compute_hash(prev, _event_payload(row)):
            return {"ok": False, "checked": checked, "first_bad_id": row.id,
                    "reason": "event content does not match its hash (row edited)",
                    "head_hash": None}
        prev = row.hash
        checked += 1
    return {"ok": True, "checked": checked, "first_bad_id": None, "reason": None,
            "head_hash": prev if checked else None}
