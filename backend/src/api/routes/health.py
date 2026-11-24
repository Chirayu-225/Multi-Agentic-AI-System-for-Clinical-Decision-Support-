"""Liveness, readiness and Prometheus metrics."""
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core import llm as _llm_settings
from src.core.config import APP_VERSION, env_num
from src.db.session import get_db
from src.jobs import outbox, queue as jobq
from src.monitoring import observability as obs
from src.pipeline import workers

router = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    # LIVENESS: the process is up and answering. Deliberately minimal and unauthenticated, and it
    # touches nothing else, so a slow database cannot make an orchestrator kill a healthy process.
    # provider and model are named so the UI can say which model is really answering, instead of a
    # name typed into the page. Neither is a secret; no key, URL or setting beyond these two is here.
    return {"status": "ok", "version": APP_VERSION,
            "provider": _llm_settings.PROVIDER, "model": _llm_settings.MODEL}


@router.get("/ready")
def ready(db: Session = Depends(get_db)):
    """READINESS: can this instance do useful work right now? Database reachable, job worker
    running (unless JOB_WORKERS=0 turned it off on purpose). Returns 503 otherwise so a load
    balancer stops sending uploads here. Reports counts only, never anything about a report.
    Not proxied by nginx: for the orchestrator, not for the internet."""
    checks: dict[str, str] = {}
    queue: dict[str, int] | None = None
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
        queue = jobq.depth_by_status(db)
        history_outbox = outbox.depth_by_status(db)
    except Exception:  # noqa: BLE001
        checks["database"] = "down"
        history_outbox = None
    checks["outbox_relay"] = "ok" if (workers.outbox_relay is not None and workers.outbox_relay.alive) else "down"
    if env_num("JOB_WORKERS", 2, int) <= 0:
        checks["worker"] = "disabled"
    else:
        checks["worker"] = "ok" if (workers.job_worker is not None and workers.job_worker.alive) else "down"
    ok = all(v in ("ok", "disabled") for v in checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", **checks, "queue": queue, "outbox": history_outbox},
                        status_code=200 if ok else 503)


@router.get("/metrics")
def prometheus_metrics(request: Request):
    """Prometheus metrics. Off (404) unless METRICS_TOKEN is set, then Bearer-protected. Contains
    counts, latencies and queue depth only. Not proxied by nginx."""
    token = os.getenv("METRICS_TOKEN", "")
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(404, "Not found")
    return PlainTextResponse(obs.render(), media_type="text/plain; version=0.0.4")
