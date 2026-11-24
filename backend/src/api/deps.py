"""Request-level helpers shared by the route modules: client IP, patient-id validation, per-patient access checks, report lookup."""
import os
import re

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from src.api.auth import Principal, ROLE_PATIENT
from src.api.views import effective_status
from src.db.models import Report
from src.security import audit


PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9_.\- ]{1,64}$")


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# ── Helpers ────────────────────────────────────────────────────────────────────

def valid_patient_id(patient_id: str) -> str:
    if not PATIENT_ID_RE.match(patient_id or ""):
        raise HTTPException(400, "patient_id must be 1-64 characters: letters, digits, space, '.', '_' or '-'. "
                                 "Use a pseudonymous ID (e.g. an MRN), not the patient's name.")
    return patient_id


def assert_patient_scope(principal: Principal, patient_id: str) -> None:
    """A patient account may only touch its own patient_id. 404 (not 403) so the
    existence of other patients' records is not revealed."""
    if principal.role == ROLE_PATIENT and principal.patient_id != patient_id:
        audit.record(actor=principal.subject, role=principal.role, action="authz.cross_patient_denied",
                     resource=f"patient:{patient_id}", outcome="denied")
        raise HTTPException(404, "Not found.")


def safe_filename(name: str) -> str:
    base = os.path.basename(name or "report")
    return re.sub(r"[^\w.\- ]", "_", base)[:120] or "report"


def load_report_for(principal: Principal, report_id: str, db: Session) -> Report:
    r = db.query(Report).filter(Report.id == report_id).first()
    if not r:
        raise HTTPException(404, "Report not found.")
    if principal.role == ROLE_PATIENT:
        # Own patient AND clinician-approved, otherwise indistinguishable from "missing".
        if r.patient_id != principal.patient_id or effective_status(r) != "approved":
            audit.record(actor=principal.subject, role=principal.role, action="authz.report_denied",
                         resource=f"report:{report_id}", outcome="denied")
            raise HTTPException(404, "Report not found.")
    return r


# ── Patient endpoints ──────────────────────────────────────────────────────────

def visible_reports(principal: Principal, patient_id: str, db: Session) -> list[Report]:
    q = db.query(Report).filter(Report.patient_id == patient_id)
    if principal.role == ROLE_PATIENT:
        q = q.filter(Report.review_status == "approved")
    return q.order_by(Report.report_date).all()
