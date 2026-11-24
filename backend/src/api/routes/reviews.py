"""The clinician review queue: approve, reject, edit, and the pending-queue PDF."""
import logging
import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.api.auth import Principal, ROLE_CLINICIAN, require_roles
from src.api.deps import client_ip
from src.api.schemas import ReviewIn
from src.api.views import effective_status, iso, review_warnings, risk_summary_of
from src.db.models import Report
from src.db.session import get_db
from src.jobs import outbox
from src.monitoring import ai_quality
from src.security import audit
from src.security.deidentification import build_redaction_map, redact
from src.tools.history_store import delete_report_history

logger = logging.getLogger("medsight")

router = APIRouter()


# Review state machine. pending -> approved | rejected; approved -> rejected (retracting a
# release); rejected is final, because rejecting removes the report from the patient's
# history store and an approval could not put it back. Re-upload the report instead.
_ALLOWED_REVIEW_TRANSITIONS = {
    "pending_review": {"approved", "rejected"},
    "approved": {"rejected"},
    "rejected": set(),
}


_REVIEW_TRANSITION_HELP = {
    "approved": "It has already been released; you can only retract it by rejecting it.",
    "rejected": "Rejection is final; upload the report again to have it reviewed.",
}


@router.post("/report/{report_id}/review")
def review_report(report_id: str, body: ReviewIn, request: Request, db: Session = Depends(get_db),
                  principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Approve or reject an AI-generated report. Clinician accounts only: the shared
    service key is deliberately NOT accepted, so every release decision is attributable
    to a named person."""
    # FOR UPDATE locks the row on PostgreSQL so two clinicians deciding at once are serialised
    # (the second one then sees the new status and gets the 409 below). SQLite ignores it.
    r = db.query(Report).filter(Report.id == report_id).with_for_update().first()
    if not r:
        raise HTTPException(404, "Report not found.")
    if body.decision == "reject" and len(body.note.strip()) < 3:
        raise HTTPException(422, "A reason is required when rejecting a report.")

    previous = effective_status(r)
    new_status = "approved" if body.decision == "approve" else "rejected"
    if new_status == "approved" and (r.analysis_status or "complete") != "complete":
        raise HTTPException(409, detail={
            "message": "This report is preliminary: the AI summaries are still being generated. "
                       "Lab flags are shown for early review, but it cannot be approved until the analysis completes.",
            "analysis_status": r.analysis_status,
        })
    if new_status not in _ALLOWED_REVIEW_TRANSITIONS.get(previous, set()):
        raise HTTPException(409, detail={
            "message": f"A report that is '{previous}' cannot be {new_status}. "
                       + _REVIEW_TRANSITION_HELP.get(previous, ""),
            "current_status": previous,
        })
    warnings = review_warnings(r)
    if body.decision == "approve" and warnings and not body.acknowledge_warnings:
        raise HTTPException(409, detail={
            "message": "This report has warnings that must be acknowledged before it can be released.",
            "warnings": warnings,
        })

    edits = {}
    if body.decision == "approve":
        for field, new in (("clinical_summary", body.edited_clinical_summary),
                           ("patient_summary", body.edited_patient_summary)):
            if new is not None and new.strip() and new.strip() != (getattr(r, field) or "").strip():
                edits[field] = new.strip()
    # Audit FIRST and fail closed: a release decision that cannot be recorded must not happen.
    audit.record(actor=principal.subject, role=principal.role, action=f"report.{body.decision}",
                 resource=f"report:{report_id}", ip=client_ip(request), critical=True,
                 detail={"patient_id": r.patient_id, "previous_status": previous,
                         "had_security_warnings": bool(r.security_warnings),
                         "had_grounding_warnings": bool(r.grounding_warnings),
                         "warnings": len(warnings), "edited": sorted(edits)})
    if new_status == "rejected":
        # A rejected report must not keep shaping this patient's trends and baselines.
        outbox.cancel_for_report(db, report_id)      # a history write still waiting must not happen
        try:
            delete_report_history(r.patient_id, report_id)
        except Exception:
            logger.exception("Could not remove rejected report %s from history store", report_id)
    if edits:
        r.ai_clinical_summary = r.ai_clinical_summary or r.clinical_summary
        r.ai_patient_summary = r.ai_patient_summary or r.patient_summary
        for field, new in edits.items():
            setattr(r, field, new)
        if "patient_summary" in edits and r.patient_summary_translated:
            # The translation was of the model's text; it no longer matches what was approved.
            r.patient_summary_translated = None
            r.translation_language = None
            r.translation_check = None
    r.review_edited = bool(edits) if body.decision == "approve" else None
    if edits:
        # The feedback signal: how much did the clinician change? (r.ai_* hold the model's originals.)
        r.edit_ratio, r.edit_words_changed = ai_quality.edit_stats([
            (r.ai_clinical_summary if f == "clinical_summary" else r.ai_patient_summary, getattr(r, f))
            for f in edits])
    r.warnings_acknowledged = bool(warnings) and body.decision == "approve"
    r.review_status = new_status
    r.reviewed_by = principal.subject
    r.reviewed_at = datetime.utcnow()
    r.review_note = body.note.strip() or None
    db.commit()
    return {"report_id": report_id, "status": new_status, "reviewed_by": principal.subject,
            "reviewed_at": iso(r.reviewed_at), "edited": sorted(edits)}


@router.get("/reviews/pending")
def pending_reviews(db: Session = Depends(get_db),
                    principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    rows = (db.query(Report)
            .filter((Report.review_status == "pending_review") | (Report.review_status.is_(None)))
            .order_by(Report.created_at.desc()).limit(200).all())
    rows.sort(key=lambda r: not r.has_critical)   # critical results first; stable, so newest-first within
    return {"count": len(rows), "reports": [{
        "analysis_status": r.analysis_status or "complete", "has_critical": bool(r.has_critical),
        "report_id": r.id, "patient_id": r.patient_id, "filename": r.filename,
        "report_date": r.report_date, "created_at": iso(r.created_at),
        "overall_severity": risk_summary_of(r).get("overall_severity"),
        "security_warnings": len(r.security_warnings or []),
        "grounding_warnings": len(r.grounding_warnings or []),
        "interaction_alerts": len((r.drug_interactions or {}).get("interactions", [])),
    } for r in rows]}


@router.get("/reviews/pending/export.pdf")
def export_pending_reviews_pdf(request: Request, include_names: bool = False,
                               db: Session = Depends(get_db),
                               principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """The review queue as a PDF. De-identified unless include_names=true; every export is
    audited, because a downloaded file leaves the system's access control behind."""
    from fastapi.responses import Response
    from src.export.queue_pdf import build_queue_pdf

    rows = (db.query(Report)
            .filter((Report.review_status == "pending_review") | (Report.review_status.is_(None)))
            .order_by(Report.created_at.desc()).limit(200).all())
    items = []
    for r in rows:
        rmap = {} if include_names else build_redaction_map(r.entities or {}, "")
        if include_names:
            clean = lambda t: t or ""  # noqa: E731
        else:
            # Placeholders like "[REDACTED_NAME_1]" are for the model; a reader gets plain words.
            clean = lambda t: re.sub(r"\[REDACTED_([A-Z]+)_\d+\]",  # noqa: E731
                                     lambda m: f"[{m.group(1).lower()} removed]", redact(t or "", rmap))
        # Abnormal by ICMR, or abnormal by the lab's own printed range: both need a look.
        flags = [f for f in (r.rule_flags or [])
                 if str(f.get("severity", "")).replace("Severity.", "") != "NORMAL"
                 or f.get("range_discordant")]
        items.append({
            "patient_id": r.patient_id, "report_date": r.report_date,
            "created_at": r.created_at.strftime("%d %b %Y %H:%M") if r.created_at else None,
            "filename": r.filename if include_names else None,
            "overall_severity": risk_summary_of(r).get("overall_severity"),
            "flags": flags,
            "interactions": (r.drug_interactions or {}).get("interactions", []),
            "warnings": [clean(w) for w in review_warnings(r)],
            "clinical_summary": clean(r.clinical_summary),
            "patient_summary": clean(r.patient_summary),
        })
    pdf = build_queue_pdf(items, generated_by=principal.subject, include_names=include_names)
    audit.record(actor=principal.subject, role=principal.role, action="reviews.export_pdf",
                 resource="reviews:pending", ip=client_ip(request), critical=True,
                 detail={"reports": len(items), "include_names": include_names})
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    return Response(content=pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="review-queue-{stamp}.pdf"',
        "Cache-Control": "no-store",
    })
