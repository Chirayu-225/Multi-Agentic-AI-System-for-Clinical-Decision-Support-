"""AI quality, LLM usage, operational metrics and the audit log."""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.agents import prompts
from src.api.auth import Principal, ROLE_CLINICIAN, require_roles
from src.api.views import effective_status
from src.db.models import AgentRun, AuditEvent, LlmCall, Report
from src.db.session import get_db
from src.monitoring import ai_quality
from src.security import audit

router = APIRouter()


# ── Operations: metrics + audit ────────────────────────────────────────────────

def _percentile(values: list[float], pct: float) -> int | None:
    if not values:
        return None
    s = sorted(values)
    return int(s[min(len(s) - 1, int(round(pct * (len(s) - 1))))])


@router.get("/admin/ai-quality")
def ai_quality_report(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db),
                      principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """What clinicians did with the AI's summaries (approved as is / edited / rejected), per prompt
    version and model. Aggregates only: no summary text, no patient identifiers."""
    since = datetime.utcnow() - timedelta(days=days)
    reports = (db.query(Report).filter(Report.review_status.in_(("approved", "rejected")),
                                       Report.reviewed_at >= since)
               .order_by(Report.reviewed_at.desc()).limit(5000).all())
    runs: dict[str, AgentRun] = {}
    if reports:
        for run in (db.query(AgentRun).filter(AgentRun.report_id.in_([r.id for r in reports]))
                    .order_by(AgentRun.id).all()):
            runs[run.report_id] = run                       # the latest run per report wins
    records = []
    for r in reports:
        run, ver = runs.get(r.id), r.verification or {}
        pv = (run.prompt_versions if run else None) or {}
        records.append({
            "status": r.review_status, "edited": bool(r.review_edited), "edit_ratio": r.edit_ratio,
            "verifier_corrected": bool(ver.get("clinical_corrected") or ver.get("patient_corrected")),
            "verifier_unverified": bool(ver.get("clinical_unverified") or ver.get("patient_unverified")),
            "summarizer_version": pv.get("summarizer"), "verifier_version": pv.get("verifier"),
            "model": run.model if run else None,
        })
    return {"window_days": days, **ai_quality.quality_summary(records)}


@router.get("/admin/llm-usage")
def llm_usage(days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db),
              principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Model usage over the last N days: calls, failures, tokens, latency and NOTIONAL cost, per
    purpose and per prompt version. Aggregates only - no report content, no patient identifiers."""
    since = datetime.utcnow() - timedelta(days=days)
    rows = (db.query(LlmCall).filter(LlmCall.created_at >= since)
            .order_by(LlmCall.id.desc()).limit(50000).all())
    groups: dict[tuple, list[LlmCall]] = {}
    for c in rows:
        groups.setdefault((c.purpose, c.prompt_version or "unversioned"), []).append(c)
    by_prompt = []
    for (purpose, version), items in sorted(groups.items()):
        ok = [c for c in items if c.status == "ok"]
        lat = [c.latency_ms for c in ok if c.latency_ms is not None]
        by_prompt.append({
            "purpose": purpose, "prompt_version": version, "calls": len(items), "ok": len(ok),
            "failed": len(items) - len(ok),
            "input_tokens": sum(c.input_tokens or 0 for c in items),
            "output_tokens": sum(c.output_tokens or 0 for c in items),
            "notional_cost_usd": round(sum(c.cost_usd or 0 for c in items), 6),
            "latency_ms_p50": _percentile(lat, 0.5), "latency_ms_p95": _percentile(lat, 0.95),
        })
    reports = {c.report_id for c in rows if c.report_id}
    total_cost = sum(c.cost_usd or 0 for c in rows)
    statuses: dict[str, int] = {}
    for c in rows:
        statuses[c.status] = statuses.get(c.status, 0) + 1
    return {
        "window_days": days, "calls": len(rows), "by_status": statuses,
        "reports": len(reports), "calls_per_report": round(len(rows) / len(reports), 2) if reports else None,
        "notional_cost_usd": round(total_cost, 6),
        "notional_cost_per_report_usd": round(total_cost / len(reports), 6) if reports else None,
        "cost_is_notional": True, "by_prompt": by_prompt,
        "prompt_versions_now": prompts.current_versions(),
    }


@router.get("/admin/metrics")
def admin_metrics(db: Session = Depends(get_db),
            principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Operational + safety metrics. Aggregates only - no patient content."""
    now = datetime.utcnow()
    day_ago = now - timedelta(hours=24)
    reports = db.query(Report).order_by(Report.created_at.desc()).limit(2000).all()
    runs = db.query(AgentRun).order_by(AgentRun.id.desc()).limit(2000).all()

    def _rate(n, d):
        return round(n / d, 4) if d else 0.0

    n = len(reports)
    status_counts = {"pending_review": 0, "approved": 0, "rejected": 0}
    corrected = unverified = with_security = with_grounding = with_interactions = 0
    for r in reports:
        status_counts[effective_status(r)] = status_counts.get(effective_status(r), 0) + 1
        v = r.verification or {}
        corrected += bool(v.get("clinical_corrected") or v.get("patient_corrected"))
        unverified += bool(v.get("clinical_unverified") or v.get("patient_unverified"))
        with_security += bool(r.security_warnings)
        with_grounding += bool(r.grounding_warnings)
        with_interactions += bool((r.drug_interactions or {}).get("interactions"))

    # ── Product: is review real, and is the tool saving clinicians time? ──────────
    # approve_without_edit_rate high AND minutes_to_review very low is the rubber-stamp
    # signature (automation bias); a high edit rate means the summaries aren't good enough yet.
    reviewed = [r for r in reports if effective_status(r) in ("approved", "rejected") and r.reviewed_at]
    approved = [r for r in reviewed if effective_status(r) == "approved"]
    edited = sum(1 for r in approved if r.review_edited)
    review_minutes = sorted(
        (r.reviewed_at - r.created_at).total_seconds() / 60
        for r in reviewed if r.created_at and r.reviewed_at >= r.created_at)

    def _q(xs, q):
        return round(xs[int(q * (len(xs) - 1))], 1) if xs else None

    durations = [x.duration_ms for x in runs if x.duration_ms]
    llm = [x.llm_calls for x in runs if x.llm_calls]
    models_used: dict[str, int] = {}
    for x in runs:
        if x.model:
            models_used[x.model] = models_used.get(x.model, 0) + 1
    tools = [len(x.tool_calls or []) for x in runs]

    events = db.query(AuditEvent).filter(AuditEvent.ts >= day_ago.isoformat()).all()
    by_action: dict[str, int] = {}
    denied = 0
    for e in events:
        by_action[e.action] = by_action.get(e.action, 0) + 1
        denied += e.outcome == "denied"

    per_day: dict[str, int] = {}
    for r in reports:
        if r.created_at and r.created_at >= now - timedelta(days=7):
            k = r.created_at.date().isoformat()
            per_day[k] = per_day.get(k, 0) + 1
    uploads_7d = [{"date": (now - timedelta(days=i)).date().isoformat(),
                   "count": per_day.get((now - timedelta(days=i)).date().isoformat(), 0)}
                  for i in range(6, -1, -1)]

    return {
        "generated_at": now.isoformat(),
        "reports": {"total": n, "last_24h": sum(1 for r in reports if r.created_at and r.created_at >= day_ago),
                    "by_review_status": status_counts, "uploads_7d": uploads_7d},
        "pipeline": {
            "avg_duration_ms": int(sum(durations) / len(durations)) if durations else None,
            "p95_duration_ms": sorted(durations)[int(0.95 * (len(durations) - 1))] if durations else None,
            "avg_llm_calls": round(sum(llm) / len(llm), 2) if llm else None,
            "avg_tool_calls": round(sum(tools) / len(tools), 2) if tools else None,
            "models": models_used,
        },
        "review": {
            "reviewed": len(reviewed),
            "approve_without_edit_rate": _rate(len(approved) - edited, len(approved)),
            "edit_rate": _rate(edited, len(approved)),
            "rejection_rate": _rate(len(reviewed) - len(approved), len(reviewed)),
            "approved_with_acknowledged_warnings": sum(1 for r in approved if r.warnings_acknowledged),
            "median_minutes_to_review": _q(review_minutes, 0.5),
            "p90_minutes_to_review": _q(review_minutes, 0.9),
            "note": ("Time to review is upload-to-decision, so it includes queue wait. A very high "
                     "approve-without-edit rate with very short review times suggests rubber-stamping."),
        },
        "safety": {
            "verifier_correction_rate": _rate(corrected, n),
            "verifier_unverified_rate": _rate(unverified, n),
            "reports_with_security_warnings": with_security,
            "reports_with_grounding_warnings": with_grounding,
            "reports_with_interaction_alerts": with_interactions,
        },
        "security_24h": {
            "events": len(events), "denied": denied,
            "login_failures": by_action.get("auth.login_failed", 0),
            "authz_denied": by_action.get("authz.denied", 0) + by_action.get("authz.report_denied", 0)
                            + by_action.get("authz.cross_patient_denied", 0),
            "rate_limited": by_action.get("rate_limit.exceeded", 0) + by_action.get("auth.login_throttled", 0),
            "injection_suspected": by_action.get("security.injection_suspected", 0),
            "by_action": dict(sorted(by_action.items(), key=lambda kv: -kv[1])[:12]),
        },
    }


@router.get("/admin/audit")
def audit_log(limit: int = Query(100, ge=1, le=500), action: str | None = None,
              db: Session = Depends(get_db),
              principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    q = db.query(AuditEvent)
    if action:
        q = q.filter(AuditEvent.action == action)
    rows = q.order_by(AuditEvent.id.desc()).limit(limit).all()
    return {"events": [{
        "id": e.id, "ts": e.ts, "actor": e.actor, "role": e.role, "action": e.action,
        "resource": e.resource, "outcome": e.outcome, "ip": e.ip, "detail": e.detail,
        "hash": e.hash[:16],
    } for e in rows]}


@router.get("/admin/audit/verify")
def audit_verify(db: Session = Depends(get_db),
                 principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    result = audit.verify_chain(db)
    audit.record(actor=principal.subject, role=principal.role, action="audit.verify",
                 outcome="success" if result["ok"] else "failure",
                 detail={"checked": result["checked"], "ok": result["ok"]})
    return result
