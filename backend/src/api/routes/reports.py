"""Read a report, translate its patient summary, export it as FHIR."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from src.agents.translator import LANGUAGES, is_supported, translate_patient_summary
from src.api.auth import Principal, ROLE_CLINICIAN, ROLE_PATIENT, require_roles
from src.api.deps import client_ip, load_report_for
from src.api.views import effective_status, report_full, report_patient_view
from src.db.models import Report
from src.db.session import get_db
from src.interop.fhir import build_bundle
from src.security import audit
from src.security.deidentification import build_redaction_map, redact, rehydrate
from src.security.rate_limit import allow as rate_limit_allow

router = APIRouter()


# ── Report access ──────────────────────────────────────────────────────────────

@router.get("/report/{report_id}")
def get_report(report_id: str, request: Request, db: Session = Depends(get_db),
               principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    r = load_report_for(principal, report_id, db)
    audit.record(actor=principal.subject, role=principal.role, action="report.view",
                 resource=f"report:{report_id}", ip=client_ip(request), detail={"patient_id": r.patient_id})
    if principal.role == ROLE_PATIENT:
        return report_patient_view(r)
    return report_full(r, db)


@router.post("/report/{report_id}/translate")
async def translate_report(report_id: str, request: Request,
                           language: str = Query(..., min_length=2, max_length=5),
                           db: Session = Depends(get_db),
                           principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    """Translate the plain-language patient summary on demand. The text is
    de-identified again before it reaches Gemini, exactly like the main pipeline."""
    language = language.lower()
    if not is_supported(language):
        raise HTTPException(400, f"Unsupported language. Use one of: {', '.join(LANGUAGES)}.")
    r = load_report_for(principal, report_id, db)
    ok, retry = rate_limit_allow(f"translate:{principal.subject}", 10, 600)
    if not ok:
        raise HTTPException(429, f"Too many translation requests. Retry in {retry}s.",
                            headers={"Retry-After": str(retry)})

    redaction_map = build_redaction_map(r.entities or {}, "")
    redacted = redact(r.patient_summary or "", redaction_map)
    result = await translate_patient_summary(redacted, language)
    text = rehydrate(result["text"], redaction_map) if result.get("text") else None
    if text:
        r.patient_summary_translated = text
        r.translation_language = language
        r.translation_check = {"ok": True, "issues": []}
        db.commit()
    audit.record(actor=principal.subject, role=principal.role, action="report.translate",
                 resource=f"report:{report_id}", ip=client_ip(request),
                 outcome="success" if text else "failure", detail={"language": language})
    if not text:
        raise HTTPException(502, "Translation could not be produced or failed automated safety checks: "
                                 + "; ".join(result.get("issues", [])))
    return {"report_id": report_id, "language": language, "text": text}


@router.get("/report/{report_id}/fhir")
def export_fhir(report_id: str, request: Request, include_name: bool = False,
                db: Session = Depends(get_db),
                principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    r = db.query(Report).filter(Report.id == report_id).first()
    if not r:
        raise HTTPException(404, "Report not found.")
    audit.record(actor=principal.subject, role=principal.role, action="report.export_fhir",
                 resource=f"report:{report_id}", ip=client_ip(request),
                 detail={"include_name": include_name, "review_status": effective_status(r)})
    bundle = build_bundle({
        "report_id": r.id, "patient_id": r.patient_id, "report_date": r.report_date,
        "review_status": effective_status(r), "entities": r.entities or {},
        "rule_flags": r.rule_flags or [], "clinical_summary": r.clinical_summary,
        "filename": r.filename,
    }, include_name=include_name)
    return bundle
