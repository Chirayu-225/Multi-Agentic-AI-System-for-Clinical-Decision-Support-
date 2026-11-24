"""Shapes database rows into API responses (clinician view vs patient view) and works out the review warnings."""
from sqlalchemy.orm import Session

from src.agents.summarizer import is_placeholder
from src.core.config import AI_DISCLAIMER
from src.db.models import AgentRun, Report
from src.flagging.risk_flagging import compute_risk_summary


def effective_status(r: Report) -> str:
    return r.review_status or "pending_review"   # NULL (legacy rows) fails closed


def risk_summary_of(r: Report) -> dict:
    """The risk summary to SHOW for a report: always recomputed from the report's stored flags.
    The stored copy is derived data, and reports saved before the severity fix hold a summary
    that ignored every lab flag (a critical report stored as NORMAL). Recomputing on read puts
    those right without a migration and keeps the banner and the flag list from ever disagreeing.
    A row with no stored flags at all (very old) falls back to what was stored."""
    stored = r.risk_summary or {}
    if r.rule_flags is None and r.trend_flags is None and r.baseline_flags is None:
        return stored
    fresh = compute_risk_summary(r.rule_flags or [], r.trend_flags or [], r.baseline_flags or [])
    return {**stored, **fresh, "overall_severity": str(fresh["overall_severity"])}


def iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def report_full(r: Report, db: Session) -> dict:
    """Clinician/service view: the same shape the upload endpoint returns."""
    run = db.query(AgentRun).filter(AgentRun.report_id == r.id).order_by(AgentRun.id.desc()).first()
    pubmed = r.pubmed_references or []
    drug_safety = r.drug_safety or []
    ver = r.verification or {}
    return {
        "report_id": r.id, "patient_id": r.patient_id, "filename": r.filename,
        "report_date": r.report_date, "source_type": r.source_type,
        "analysis_status": r.analysis_status or "complete", "has_critical": bool(r.has_critical),
        "has_hindi": str(r.has_hindi).lower() == "true", "pages": r.pages,
        "entities": r.entities or {}, "rule_flags": r.rule_flags or [],
        "trend_flags": r.trend_flags or [], "baseline_flags": r.baseline_flags or [],
        "forecast_flags": r.forecast_flags or [], "similar_history": r.similar_history or [],
        "risk_summary": risk_summary_of(r), "pubmed_references": pubmed,
        "drug_safety": drug_safety, "drug_interactions": r.drug_interactions or {},
        "ingestion_warnings": r.ingestion_warnings or [],
        "security_warnings": r.security_warnings or [],
        "grounding_warnings": r.grounding_warnings or [],
        "evidence": r.evidence or [],
        "disclaimer": AI_DISCLAIMER,
        "review": {"status": effective_status(r), "reviewed_by": r.reviewed_by,
                   "reviewed_at": iso(r.reviewed_at), "note": r.review_note,
                   "edited": bool(r.review_edited),
                   "warnings_acknowledged": bool(r.warnings_acknowledged),
                   "requires_acknowledgement": review_warnings(r)},
        "llm_output": {
            "clinical_summary": r.clinical_summary, "patient_summary": r.patient_summary,
            "ai_original": ({"clinical_summary": r.ai_clinical_summary,
                             "patient_summary": r.ai_patient_summary}
                            if r.review_edited else None),
            "explained_flags": r.explained_flags or [], "followup_actions": r.followup_actions or [],
            "longitudinal_narrative": r.longitudinal_narrative or "",
            "translation": {"language": r.translation_language, "text": r.patient_summary_translated,
                            "check": r.translation_check} if r.translation_language else None,
            "verification": {
                "clinical_summary_corrected": ver.get("clinical_corrected", False),
                "patient_summary_corrected": ver.get("patient_corrected", False),
                "clinical_unverified": ver.get("clinical_unverified", False),
                "patient_unverified": ver.get("patient_unverified", False),
                "directive_actions": ver.get("directive_actions", []),
            },
        },
        "agent_metadata": {
            "tool_calls": (run.tool_calls if run else []) or [],
            "pubmed_found": len(pubmed), "drug_safety_checked": len(drug_safety),
            "duration_ms": run.duration_ms if run else None, "llm_calls": run.llm_calls if run else None,
        },
    }


def report_patient_view(r: Report) -> dict:
    """What a patient account sees: plain-language summary and their own numbers —
    no clinician-facing text, no model-audit detail, no security internals."""
    flags = [
        {"test": f.get("test"), "value": f.get("value"), "unit": f.get("unit"),
         "severity": str(f.get("severity", "")).replace("Severity.", ""),
         "normal_range": f.get("normal_range"), "direction": f.get("direction")}
        for f in (r.rule_flags or [])
    ]
    return {
        "report_id": r.id, "patient_id": r.patient_id, "report_date": r.report_date,
        "filename": r.filename,
        "review": {"status": effective_status(r), "reviewed_by": r.reviewed_by,
                   "reviewed_at": iso(r.reviewed_at)},
        "risk_summary": {"overall_severity": risk_summary_of(r).get("overall_severity"),
                         "color": risk_summary_of(r).get("color")},
        "patient_summary": r.patient_summary,
        "patient_summary_translated": ({"language": r.translation_language,
                                        "text": r.patient_summary_translated}
                                       if r.translation_language and r.patient_summary_translated else None),
        "lab_values": flags, "disclaimer": AI_DISCLAIMER,
    }


def review_warnings(r: Report) -> list[str]:
    """Everything a clinician must explicitly accept before a report can be released.
    Without this gate the warnings were logged but approval was one click either way —
    the shape that turns review into rubber-stamping."""
    ver = r.verification or {}
    out = []
    if ver.get("clinical_unverified") or ver.get("patient_unverified"):
        if (ver.get("clinical_missing_critical") or ver.get("content_issues")
                or is_placeholder(r.clinical_summary) or is_placeholder(r.patient_summary)):
            # The verifier ran (or had nothing to run on); the reason is in the AI-summary warning below.
            out.append("The AI summaries are NOT verified: one is missing, incomplete, or leaves out "
                       "a critical result.")
        else:
            out.append("The AI Verifier could not run: the summaries have NOT been audited.")
    out += [f"Grounding: {w}" for w in (r.grounding_warnings or [])]
    out += [f"Security: {w}" for w in (r.security_warnings or [])]
    out += [f"Extraction: {w}" for w in (r.ingestion_warnings or [])]
    return out
