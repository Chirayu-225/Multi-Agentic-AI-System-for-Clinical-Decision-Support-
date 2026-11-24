"""Patient history, trends, a patient's own reports, patient list and erasure."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.api.auth import Principal, ROLE_CLINICIAN, ROLE_PATIENT, require_roles
from src.api.deps import assert_patient_scope, client_ip, visible_reports
from src.api.views import effective_status, risk_summary_of
from src.db.models import AgentRun, LlmCall, Patient
from src.db.session import get_db
from src.jobs import outbox
from src.security import audit
from src.tools.history_store import delete_patient_history
from src.tools.trend_series import build_trend_series

logger = logging.getLogger("medsight")

router = APIRouter()


@router.get("/patient/{patient_id}/history")
def patient_history(patient_id: str, request: Request, db: Session = Depends(get_db),
                    principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    assert_patient_scope(principal, patient_id)
    reports = visible_reports(principal, patient_id, db)
    audit.record(actor=principal.subject, role=principal.role, action="patient.history_view",
                 resource=f"patient:{patient_id}", ip=client_ip(request))
    return {
        "patient_id": patient_id,
        "report_count": len(reports),
        "reports": [
            {
                "report_id": r.id,
                "filename": r.filename,
                "report_date": r.report_date,
                "risk_summary": risk_summary_of(r),
                "lab_values": r.lab_values,
                "review_status": effective_status(r),
                "diagnoses": r.entities.get("clinical", {}).get("diagnoses", []) if r.entities else [],
            }
            for r in reports
        ],
    }


@router.get("/patient/{patient_id}/trends")
def patient_trends(patient_id: str, db: Session = Depends(get_db),
                   principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    """Chart-ready lab series (with the ICMR reference band) across a patient's reports."""
    assert_patient_scope(principal, patient_id)
    reports = visible_reports(principal, patient_id, db)
    series = build_trend_series([{
        "report_id": r.id, "report_date": r.report_date, "lab_values": r.lab_values or [],
        "gender": (r.entities or {}).get("patient", {}).get("gender"),
    } for r in reports])
    return {"patient_id": patient_id, "report_count": len(reports), "series": series}


@router.get("/me/reports")
def my_reports(db: Session = Depends(get_db),
               principal: Principal = Depends(require_roles(ROLE_PATIENT))):
    reports = visible_reports(principal, principal.patient_id, db)
    return {"patient_id": principal.patient_id, "reports": [{
        "report_id": r.id, "report_date": r.report_date, "filename": r.filename,
        "overall_severity": risk_summary_of(r).get("overall_severity"),
    } for r in reports]}


@router.get("/patients")
def list_patients(db: Session = Depends(get_db),
                  principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    patients = db.query(Patient).all()
    return {"patients": [p.id for p in patients]}


@router.delete("/patient/{patient_id}")
def delete_patient(patient_id: str, request: Request, db: Session = Depends(get_db),
                   principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found.")
    n_reports = len(patient.reports)
    # Erase the vector-store copy FIRST. If that fails, nothing is deleted and the caller
    # gets an error — never a "deleted" response while report text survives elsewhere.
    try:
        vectors_cleared = delete_patient_history(patient_id)
    except Exception:
        logger.exception("Erasure failed in history store for patient_id=%s", patient_id)
        raise HTTPException(500, "Erasure could not complete (history store). Nothing was deleted; retry.")
    audit.record(actor=principal.subject, role=principal.role, action="patient.delete",
                 resource=f"patient:{patient_id}", ip=client_ip(request), critical=True,
                 detail={"reports_deleted": n_reports, "history_store_cleared": vectors_cleared})
    # agent_runs reference reports by foreign key: they must go first or PostgreSQL refuses the
    # delete (SQLite does not enforce the key, which is how this went unnoticed).
    report_ids = [r.id for r in patient.reports]
    if report_ids:
        db.query(AgentRun).filter(AgentRun.report_id.in_(report_ids)).delete(synchronize_session=False)
        db.query(LlmCall).filter(LlmCall.report_id.in_(report_ids)).delete(synchronize_session=False)
    outbox.delete_for_patient(db, patient_id)
    db.delete(patient)
    db.commit()
    logger.warning("Patient record deleted: patient_id=%s by=%s", patient_id, principal.subject)
    return {"message": f"Patient {patient_id} and all reports deleted."}
