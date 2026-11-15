"""
main.py
Agentic Medical Assistant — FastAPI backend.

Endpoints
  Auth        POST /auth/login · GET /auth/me · POST /auth/change-password
  Accounts    GET/POST /admin/users · POST /admin/users/{username}/reset-password
              POST /admin/users/{username}/deactivate · .../reactivate
  Pipeline    POST /report/upload · WS /ws/{report_id}
  Review      POST /report/{id}/review · GET /reviews/pending
  Reports     GET /report/{id} · POST /report/{id}/translate · GET /report/{id}/fhir
  Patients    GET /patients · GET /patient/{id}/history · GET /patient/{id}/trends
              DELETE /patient/{id} · GET /me/reports (patient accounts)
  Operations  GET /admin/metrics · GET /admin/audit · GET /admin/audit/verify · GET /health
"""

import asyncio
import json
import logging
import os
import re
import secrets
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# Load .env by an explicit path next to this file, before any of the
# src.* imports below — src/llm.py reads GEMINI_MODEL at import time, so
# .env must be loaded before those imports run.
# Tests set MEDSIGHT_SKIP_DOTENV=1 (tests/conftest.py): with override=True, a developer's real
# backend/.env would otherwise replace the test values (JWT_SECRET...) and fail the
# auth tests on any machine that has one.
from dotenv import load_dotenv
if os.getenv("MEDSIGHT_SKIP_DOTENV") != "1":
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Query, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.session import get_db, create_tables, SessionLocal
from src.db.models import Patient, Report, AgentRun, LlmCall, User, AuditEvent
from src.ingestion.document_loader import (
    load_document, ALLOWED_EXTS, PDF_EXTS, sniff_file_type, extension_matches_content,
)
from src.agents.orchestrator import run_orchestrator
from src import ai_quality, outbox, prompts
from src.llm import begin_call_tracking, begin_model_tracking, llm_unavailable_for, required_key_name
from src.agents.summarizer import generate_summaries
from src.agents.verifier import verify_and_correct
from src.agents.translator import (LANGUAGES, is_supported, translate_patient_summary)
from src.tools.history_store import (store_report as chroma_store, delete_patient_history,
                                     delete_report_history)
from src.tools.trend_series import build_trend_series
from src.api import auth as authmod
from src.api.auth import (Principal, ROLE_CLINICIAN, ROLE_PATIENT,
                          get_principal, require_roles, verify_ws_auth, fingerprint)
from src import observability as obs
from src.jobs import queue as jobq
from src.jobs.worker import JobWorker
from src.security import audit
from src.security.passwords import (DUMMY_HASH, WeakPassword, hash_password,
                                    validate_password_strength, verify_password)
from src.security.rate_limit import allow as rate_limit_allow
from src.security.llm_guardrails import scan_output_for_leakage, detect_hidden_pdf_text
from src.security.deidentification import rehydrate, redact, build_redaction_map
from src.evidence.grounding import build_evidence, collect_allowed_numbers, find_ungrounded_numbers
from src.interop.fhir import build_bundle

obs.configure_logging()   # one log format everywhere, carrying the request id and job id
logger = logging.getLogger("medsight")

# Every AI-generated summary carries this disclaimer end-to-end (API + UI) —
# this tool assists a clinician's review, it does not replace it.
AI_DISCLAIMER = (
    "AI-generated analysis for clinician review only. Not a diagnosis and not "
    "a substitute for professional medical judgment. Verify all findings "
    "against the source report before acting on them."
)

APP_VERSION = "3.0.0"

app = FastAPI(
    title="Agentic Medical Assistant API",
    description="Multi-Agent Clinical Intelligence Platform",
    version=APP_VERSION,
)

# Allowed browser origins. Entries are trimmed (a stray space after a comma in .env used to
# silently break matching). localhost / 127.0.0.1 on ANY port are also accepted so local
# development works whichever way the dev server is opened; set
# ALLOW_LOCALHOST_ORIGINS=false in production so only CORS_ORIGINS is trusted.
_cors_origins = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",") if o.strip()]
_localhost_regex = (
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    if os.getenv("ALLOW_LOCALHOST_ORIGINS", "true").lower() != "false" else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_localhost_regex,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Defence-in-depth response headers. Responses carry patient data, so they must
    not be cached by browsers or shared proxies, sniffed, or framed."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Give every request a correlation id (logged on every line, returned as X-Request-ID) and
    count it. The metric's route label is the route TEMPLATE (/report/{report_id}), never the
    concrete path, so it cannot grow with the number of reports or leak an id."""
    rid = obs.accept_request_id(request.headers.get("x-request-id"), lambda: uuid.uuid4().hex[:16])
    token = obs.request_id_var.set(rid)
    started = time.monotonic()
    try:
        response = await call_next(request)
    finally:
        obs.request_id_var.reset(token)
    route = getattr(request.scope.get("route"), "path", "unmatched")
    obs.inc("http_requests_total", method=request.method, route=route, status=str(response.status_code))
    obs.observe("http_request_seconds", time.monotonic() - started, route=route)
    response.headers["X-Request-ID"] = rid
    return response


# Active WebSocket connections per report_id
_ws_connections: dict[str, WebSocket] = {}

PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9_.\- ]{1,64}$")
LOGIN_MAX_FAILURES = 5
LOGIN_LOCK_MINUTES = 15


def _require_review() -> bool:
    return os.getenv("REQUIRE_CLINICIAN_REVIEW", "true").lower() != "false"


def _ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@app.on_event("startup")
async def startup():
    # Fail fast and loud on missing secrets rather than 500ing on first
    # request — much easier to diagnose in a demo / interview setting.
    required = [required_key_name()]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"See backend/.env.example."
        )
    authmod.validate_auth_config()
    create_tables()
    _bootstrap_users()
    await _start_job_worker()
    await _start_outbox_relay()


@app.on_event("shutdown")
async def shutdown():
    # Running jobs are handed back to the queue (not lost, not counted as an attempt).
    if _worker is not None:
        await _worker.stop()
    if _relay is not None:
        await _relay.stop()


def _bootstrap_users():
    """Create a first clinician (and optionally a demo patient) from env vars so a fresh
    install is usable. Passwords come from the environment, never from source code."""
    specs = [
        ("BOOTSTRAP_CLINICIAN", ROLE_CLINICIAN),
        ("BOOTSTRAP_PATIENT", ROLE_PATIENT),
    ]
    db = SessionLocal()
    try:
        for prefix, role in specs:
            username = os.getenv(f"{prefix}_USERNAME", "").strip()
            password = os.getenv(f"{prefix}_PASSWORD", "")
            if not username or not password:
                continue
            if db.query(User).filter(User.username == username).first():
                continue
            patient_id = os.getenv(f"{prefix}_PATIENT_ID", "").strip() or None
            if role == ROLE_PATIENT and not patient_id:
                logger.error("%s_PATIENT_ID is required for a patient account - skipped.", prefix)
                continue
            try:
                validate_password_strength(password)
            except WeakPassword as e:
                logger.error("Bootstrap user '%s' skipped: %s", username, e)
                continue
            db.add(User(username=username, display_name=os.getenv(f"{prefix}_DISPLAY_NAME", username),
                        password_hash=hash_password(password), role=role, patient_id=patient_id))
            db.commit()
            logger.warning("Bootstrap %s account created: %s", role, username)
    finally:
        db.close()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    # LIVENESS: the process is up and answering. Deliberately minimal and unauthenticated, and it
    # touches nothing else, so a slow database cannot make an orchestrator kill a healthy process.
    return {"status": "ok", "version": APP_VERSION}


@app.get("/ready")
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
    checks["outbox_relay"] = "ok" if (_relay is not None and _relay.alive) else "down"
    if _env_num("JOB_WORKERS", 2, int) <= 0:
        checks["worker"] = "disabled"
    else:
        checks["worker"] = "ok" if (_worker is not None and _worker.alive) else "down"
    ok = all(v in ("ok", "disabled") for v in checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", **checks, "queue": queue, "outbox": history_outbox},
                        status_code=200 if ok else 503)


@app.get("/metrics")
def metrics(request: Request):
    """Prometheus metrics. Off (404) unless METRICS_TOKEN is set, then Bearer-protected. Contains
    counts, latencies and queue depth only. Not proxied by nginx."""
    token = os.getenv("METRICS_TOKEN", "")
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(404, "Not found")
    return PlainTextResponse(obs.render(), media_type="text/plain; version=0.0.4")


# ── Auth ───────────────────────────────────────────────────────────────────────

class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@app.post("/auth/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    ip = _ip(request)
    username = body.username.strip().lower()

    # Throttle by IP and by username so neither a spray nor a single-account brute force
    # gets unlimited guesses (in-process limiter - see rate_limit.py caveat).
    for key, limit in ((f"login-ip:{ip}", 20), (f"login-user:{username}", 8)):
        ok, retry = rate_limit_allow(key, limit, 300)
        if not ok:
            audit.record(actor=username, action="auth.login_throttled", outcome="denied", ip=ip)
            raise HTTPException(429, f"Too many login attempts. Retry in {retry}s.",
                                headers={"Retry-After": str(retry)})

    user = db.query(User).filter(User.username == username).first()
    now = datetime.utcnow()

    if user and user.locked_until and user.locked_until > now:
        audit.record(actor=username, role=user.role, action="auth.login_locked", outcome="denied", ip=ip)
        raise HTTPException(423, "Account temporarily locked after repeated failed logins.")

    # Always run one scrypt verification (against a dummy hash for unknown users) so
    # response time does not reveal whether the username exists.
    valid = verify_password(body.password, user.password_hash if user else DUMMY_HASH)
    if not user or not user.is_active or not valid:
        if user:
            user.failed_logins = (user.failed_logins or 0) + 1
            if user.failed_logins >= LOGIN_MAX_FAILURES:
                user.locked_until = now + timedelta(minutes=LOGIN_LOCK_MINUTES)
                user.failed_logins = 0
            db.commit()
        audit.record(actor=username, action="auth.login_failed", outcome="denied", ip=ip)
        raise HTTPException(401, "Invalid username or password.")

    user.failed_logins = 0
    user.locked_until = None
    user.last_login_at = now
    db.commit()
    audit.record(actor=user.username, role=user.role, action="auth.login", ip=ip)
    return _session_payload(user)


def _session_payload(user: User) -> dict:
    """Token + profile returned by login and by a password change. A user whose password is
    temporary gets a token that works for nothing except /auth/change-password."""
    must_change = bool(user.must_change_password)
    token, ttl = authmod.create_access_token(username=user.username, role=user.role,
                                             patient_id=user.patient_id,
                                             must_change_password=must_change,
                                             token_version=user.token_version or 0)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl,
            "role": user.role, "username": user.username,
            "display_name": user.display_name or user.username, "patient_id": user.patient_id,
            "must_change_password": must_change}


@app.get("/auth/me")
def whoami(principal: Principal = Depends(get_principal)):
    return {"username": principal.subject, "role": principal.role,
            "patient_id": principal.patient_id, "via": principal.via,
            "must_change_password": principal.must_change_password}


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


@app.post("/auth/change-password")
def change_password(body: ChangePasswordIn, request: Request, db: Session = Depends(get_db),
                    principal: Principal = Depends(get_principal)):
    """Change your own password. Also the only call a temporary-password token may make;
    on success the temporary flag is cleared and a fresh, unrestricted token is returned.
    Uses get_principal (not require_roles) on purpose, see api/auth.py."""
    ip = _ip(request)
    # A stolen token must not become an unlimited oracle for guessing the current password.
    ok, retry = rate_limit_allow(f"chpw:{principal.subject}", 5, 900)
    if not ok:
        audit.record(actor=principal.subject, role=principal.role, action="auth.password_change_throttled",
                     outcome="denied", ip=ip)
        raise HTTPException(429, f"Too many attempts. Retry in {retry}s.", headers={"Retry-After": str(retry)})

    user = db.query(User).filter(User.username == principal.subject).first()
    if not user or not user.is_active:
        raise HTTPException(401, "Invalid or expired token.", headers={"WWW-Authenticate": "Bearer"})
    # 400 (not 401/403): the frontend treats any 401 as "session over" and signs the user out.
    if not verify_password(body.current_password, user.password_hash):
        audit.record(actor=user.username, role=user.role, action="auth.password_change_failed",
                     outcome="denied", ip=ip)
        raise HTTPException(400, "Current password is incorrect.")
    if body.new_password == body.current_password:
        raise HTTPException(400, "Choose a password different from the current one.")
    try:
        validate_password_strength(body.new_password)
    except WeakPassword as e:
        raise HTTPException(400, str(e))

    user.password_hash = hash_password(body.new_password)
    user.must_change_password = False
    user.failed_logins = 0
    user.locked_until = None
    # Every token issued so far (including the one making this call) stops working; the fresh
    # token returned below carries the new version, so this device stays signed in.
    user.token_version = (user.token_version or 0) + 1
    db.commit()
    audit.record(actor=user.username, role=user.role, action="auth.password_changed", ip=ip)
    return _session_payload(user)


# ── Patient accounts (created and reset by clinicians) ─────────────────────────

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
# No 0/O, 1/l/I: a temporary password is read aloud or copied by hand.
_TEMP_PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"


def _temporary_password() -> str:
    return "".join(secrets.choice(_TEMP_PASSWORD_ALPHABET) for _ in range(14))


class CreatePatientAccountIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    patient_id: str = Field(min_length=1, max_length=64)
    display_name: str | None = Field(default=None, max_length=100)


def _account_view(u: User) -> dict:
    return {"username": u.username, "display_name": u.display_name or u.username,
            "patient_id": u.patient_id, "is_active": bool(u.is_active),
            "must_change_password": bool(u.must_change_password),
            "last_login_at": _iso(u.last_login_at)}


@app.get("/admin/users")
def list_patient_accounts(db: Session = Depends(get_db),
                          principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    rows = db.query(User).filter(User.role == ROLE_PATIENT).order_by(User.username).all()
    return {"users": [_account_view(u) for u in rows]}


@app.post("/admin/users", status_code=201)
def create_patient_account(body: CreatePatientAccountIn, request: Request, db: Session = Depends(get_db),
                           principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """A clinician creates a login for a patient, bound to ONE existing patient_id. The
    temporary password is returned once, here, and is never stored in clear. The patient
    must change it at first sign-in. Clinician accounts are not created through the API."""
    username = body.username.strip().lower()
    if not USERNAME_RE.match(username):
        raise HTTPException(422, "Username must be 3-64 characters: lowercase letters, digits, '.', '_' or '-'.")
    patient_id = _valid_patient_id(body.patient_id.strip())
    # Must already exist: a typo here would give someone access to a different person's reports.
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(404, "No patient with this ID. Upload one of their reports first.")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(409, "That username is already taken.")

    temp = _temporary_password()
    audit.record(actor=principal.subject, role=principal.role, action="user.create",
                 resource=f"user:{username}", ip=_ip(request), critical=True,
                 detail={"role": ROLE_PATIENT, "patient_id": patient_id})
    user = User(username=username, display_name=(body.display_name or "").strip() or username,
                password_hash=hash_password(temp), role=ROLE_PATIENT, patient_id=patient_id,
                must_change_password=True)
    db.add(user)
    db.commit()
    return {**_account_view(user), "temporary_password": temp,
            "note": "Shown once. Give it to the patient securely; they must change it at first sign-in."}


def _patient_account_or_404(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username.strip().lower()).first()
    if not user or user.role != ROLE_PATIENT:
        raise HTTPException(404, "No patient account with that username.")
    return user


@app.post("/admin/users/{username}/deactivate")
def deactivate_patient_account(username: str, request: Request, db: Session = Depends(get_db),
                               principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Disable a patient login and end any session it has open. Their reports are untouched."""
    user = _patient_account_or_404(db, username)
    audit.record(actor=principal.subject, role=principal.role, action="user.deactivate",
                 resource=f"user:{user.username}", ip=_ip(request), critical=True,
                 detail={"patient_id": user.patient_id})
    user.is_active = False
    user.token_version = (user.token_version or 0) + 1
    db.commit()
    return _account_view(user)


@app.post("/admin/users/{username}/reactivate")
def reactivate_patient_account(username: str, request: Request, db: Session = Depends(get_db),
                               principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Re-enable a disabled patient login. The password is unchanged; sessions from before the
    deactivation stay dead (the token version was already bumped)."""
    user = _patient_account_or_404(db, username)
    audit.record(actor=principal.subject, role=principal.role, action="user.reactivate",
                 resource=f"user:{user.username}", ip=_ip(request), critical=True,
                 detail={"patient_id": user.patient_id})
    user.is_active = True
    user.failed_logins = 0
    user.locked_until = None
    db.commit()
    return _account_view(user)


@app.post("/admin/users/{username}/reset-password")
def reset_patient_password(username: str, request: Request, db: Session = Depends(get_db),
                           principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Forgotten password: issue a new temporary one, unlock the account, force a change.
    Patient accounts only (there is no admin role yet, so one clinician cannot reset another)."""
    user = _patient_account_or_404(db, username)
    temp = _temporary_password()
    audit.record(actor=principal.subject, role=principal.role, action="user.password_reset",
                 resource=f"user:{user.username}", ip=_ip(request), critical=True,
                 detail={"patient_id": user.patient_id})
    user.password_hash = hash_password(temp)
    user.must_change_password = True
    user.failed_logins = 0
    user.locked_until = None
    user.token_version = (user.token_version or 0) + 1      # signs out any session already open
    db.commit()
    return {**_account_view(user), "temporary_password": temp,
            "note": "Shown once. Give it to the patient securely; they must change it at first sign-in."}


# ── WebSocket — real-time agent streaming ──────────────────────────────────────

@app.websocket("/ws/{report_id}")
async def websocket_endpoint(websocket: WebSocket, report_id: str):
    """
    WebSocket endpoint for real-time agent step streaming (staff only — patient
    accounts have no business watching the analysis pipeline).

    Auth: browsers' native WebSocket API can't set custom headers, so the credential
    travels as a query param instead: `?token=<JWT>`. Use wss://
    in any non-local deployment so the query string is encrypted in transit.
    """
    principal = await verify_ws_auth(websocket)
    if principal is None or not principal.is_staff:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    _ws_connections[report_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        _ws_connections.pop(report_id, None)


def _make_ws_callback(report_id: str):
    """Return a callback that sends agent steps to the connected WebSocket."""
    async def _send(step: dict):
        ws = _ws_connections.get(report_id)
        if ws:
            try:
                await ws.send_text(json.dumps(step))
            except Exception:
                logger.debug("websocket send failed for report_id=%s", report_id)

    def callback(step: dict):
        asyncio.create_task(_send(step))
    return callback


# ── Upload jobs ────────────────────────────────────────────────────────────────
# An upload is accepted fast (validated, written to the spool, queued as a row in `upload_jobs`)
# and analysed later by a JobWorker. The job id IS the report id (the client already opens
# /ws/{report_id} before uploading). The queue lives in the database, so a restart does not lose
# work: see src/jobs/queue.py for the delivery guarantee (at-least-once) and why the pipeline's
# final write is idempotent. Nothing in a job row holds PHI: ids, stage, percent, counters.

_worker: JobWorker | None = None


def _env_num(name: str, default, cast=float):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _spool_dir() -> str:
    """Where uploaded files wait until their job finishes. Must survive a restart (so a queued
    job can still run afterwards) and be readable by every worker. Mode 0700: it holds PHI."""
    path = os.getenv("UPLOAD_SPOOL_DIR") or os.path.join(tempfile.gettempdir(), "medsight-spool")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


# Stage -> percent. Tool calls inside the orchestrator advance 10 -> 50 in small steps.
_STAGE_PERCENT = {
    "queued": 0, "pipeline_start": 10, "orchestrator_complete": 50, "generating_summaries": 60,
    "verifier_start": 75, "verifier_complete": 85, "translating": 90, "saving": 95, "pipeline_complete": 100,
}


def _make_job_callback(job_id: str, report_progress):
    """Forward agent steps to the websocket, add `progress` events, and persist progress on the job
    row (via `report_progress`) so the polling endpoint works even if the websocket never connects."""
    ws = _make_ws_callback(job_id)
    state = {"percent": 0}

    def callback(step: dict):
        ws(step)
        stage = step.get("type")
        pct = _STAGE_PERCENT.get(stage)
        if stage in ("tool_call", "tool_result"):
            pct = min(state["percent"] + 2, 49)
        if pct is None or pct < state["percent"]:
            return
        state["percent"] = pct
        report_progress(stage, pct)
        ws({"type": "progress", "report_id": job_id, "stage": stage, "percent": pct})
    return callback


def _job_failure(job: jobq.JobSnapshot, exc: BaseException, final: bool) -> str:
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


def _job_finished(job: jobq.JobSnapshot, status: str, error: str | None) -> None:
    """Terminal state reached: tell the browser and delete the uploaded file (PHI) from the spool."""
    ws = _make_ws_callback(job.id)
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
# outbox event in that transaction and applied afterwards with retries (src/outbox.py).

_relay: "outbox.OutboxRelay | None" = None
HISTORY_PENDING = "History update pending: "


def _outbox_config() -> dict:
    return {
        "lease_seconds": _env_num("OUTBOX_LEASE_SECONDS", 120.0),
        "max_attempts": _env_num("OUTBOX_MAX_ATTEMPTS", 8, int),
        "base_seconds": _env_num("OUTBOX_RETRY_BASE_SECONDS", 30.0),
        "poll_seconds": _env_num("OUTBOX_POLL_SECONDS", 5.0),
        "retention_hours": _env_num("OUTBOX_RETENTION_HOURS", 24.0),
    }


def _history_text(redacted_text: str, words: int = 512) -> str:
    return " ".join(redacted_text.split()[:words])      # what the vector store embeds, and no more


def _apply_history_event(ev: dict) -> str | None:
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


def _outbox_result(ev: dict, outcome: str, error: str | None) -> None:
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


async def _start_outbox_relay() -> None:
    global _relay
    _relay = outbox.OutboxRelay(session_factory=SessionLocal, handler=_apply_history_event,
                                config=_outbox_config, on_result=_outbox_result)
    await _relay.start()

    def _depth() -> dict:
        db = SessionLocal()
        try:
            return outbox.depth_by_status(db)
        finally:
            db.close()

    obs.gauge("outbox_events", "History-store outbox events by status (database-wide).", _depth)


async def _start_job_worker() -> None:
    global _worker
    workers = _env_num("JOB_WORKERS", 2, int)
    if workers <= 0:
        logger.info("JOB_WORKERS=0: this process only accepts uploads; another process must run the jobs")
        _worker = None
    else:
        _worker = JobWorker(
            session_factory=SessionLocal, process=_process_upload,
            on_failure=_job_failure, on_finished=_job_finished,
            concurrency=workers,
            lease_seconds=_env_num("JOB_LEASE_SECONDS", 60.0),
            poll_seconds=_env_num("JOB_POLL_SECONDS", 1.0),
            retry_base_seconds=_env_num("JOB_RETRY_BASE_SECONDS", 15.0),
            retention=timedelta(hours=_env_num("JOB_RETENTION_HOURS", 24.0)),
        )
        await _worker.start()

    def _depth() -> dict:
        db = SessionLocal()
        try:
            return jobq.depth_by_status(db)
        finally:
            db.close()

    obs.gauge("upload_jobs", "Upload jobs by status (database-wide).", _depth)


@app.get("/report/{report_id}/job")
def get_upload_job(report_id: str, db: Session = Depends(get_db),
                   principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Job status; also the poll fallback for the websocket (late connect, dropped socket,
    or a job being processed by another process). Uploader only."""
    row = jobq.get(db, report_id)
    if not row or row.owner != principal.subject:
        raise HTTPException(404, "No such upload job.")
    return jobq.job_view(row)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _valid_patient_id(patient_id: str) -> str:
    if not PATIENT_ID_RE.match(patient_id or ""):
        raise HTTPException(400, "patient_id must be 1-64 characters: letters, digits, space, '.', '_' or '-'. "
                                 "Use a pseudonymous ID (e.g. an MRN), not the patient's name.")
    return patient_id


def _assert_patient_scope(principal: Principal, patient_id: str) -> None:
    """A patient account may only touch its own patient_id. 404 (not 403) so the
    existence of other patients' records is not revealed."""
    if principal.role == ROLE_PATIENT and principal.patient_id != patient_id:
        audit.record(actor=principal.subject, role=principal.role, action="authz.cross_patient_denied",
                     resource=f"patient:{patient_id}", outcome="denied")
        raise HTTPException(404, "Not found.")


def _effective_status(r: Report) -> str:
    return r.review_status or "pending_review"   # NULL (legacy rows) fails closed


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "report")
    return re.sub(r"[^\w.\- ]", "_", base)[:120] or "report"


def _report_full(r: Report, db: Session) -> dict:
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
        "risk_summary": r.risk_summary or {}, "pubmed_references": pubmed,
        "drug_safety": drug_safety, "drug_interactions": r.drug_interactions or {},
        "ingestion_warnings": r.ingestion_warnings or [],
        "security_warnings": r.security_warnings or [],
        "grounding_warnings": r.grounding_warnings or [],
        "evidence": r.evidence or [],
        "disclaimer": AI_DISCLAIMER,
        "review": {"status": _effective_status(r), "reviewed_by": r.reviewed_by,
                   "reviewed_at": _iso(r.reviewed_at), "note": r.review_note,
                   "edited": bool(r.review_edited),
                   "warnings_acknowledged": bool(r.warnings_acknowledged),
                   "requires_acknowledgement": _review_warnings(r)},
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
            },
        },
        "agent_metadata": {
            "tool_calls": (run.tool_calls if run else []) or [],
            "pubmed_found": len(pubmed), "drug_safety_checked": len(drug_safety),
            "duration_ms": run.duration_ms if run else None, "llm_calls": run.llm_calls if run else None,
        },
    }


def _report_patient_view(r: Report) -> dict:
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
        "review": {"status": _effective_status(r), "reviewed_by": r.reviewed_by,
                   "reviewed_at": _iso(r.reviewed_at)},
        "risk_summary": {"overall_severity": (r.risk_summary or {}).get("overall_severity"),
                         "color": (r.risk_summary or {}).get("color")},
        "patient_summary": r.patient_summary,
        "patient_summary_translated": ({"language": r.translation_language,
                                        "text": r.patient_summary_translated}
                                       if r.translation_language and r.patient_summary_translated else None),
        "lab_values": flags, "disclaimer": AI_DISCLAIMER,
    }


def _load_report_for(principal: Principal, report_id: str, db: Session) -> Report:
    r = db.query(Report).filter(Report.id == report_id).first()
    if not r:
        raise HTTPException(404, "Report not found.")
    if principal.role == ROLE_PATIENT:
        # Own patient AND clinician-approved, otherwise indistinguishable from "missing".
        if r.patient_id != principal.patient_id or _effective_status(r) != "approved":
            audit.record(actor=principal.subject, role=principal.role, action="authz.report_denied",
                         resource=f"report:{report_id}", outcome="denied")
            raise HTTPException(404, "Report not found.")
    return r


# ── Report upload & analysis ───────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
LOW_OCR_CONFIDENCE = 0.6

# The pipeline behind this endpoint is expensive (OCR/NER + 2-3 Gemini calls +
# external API calls). Limits are per caller (user / service key) — see
# src/security/rate_limit.py for why this is process-local and what to replace it
# with at multi-instance scale.
UPLOAD_RATE_LIMIT_MAX = int(os.getenv("UPLOAD_RATE_LIMIT_MAX", "10"))
UPLOAD_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("UPLOAD_RATE_LIMIT_WINDOW_SECONDS", "600"))


@app.post("/report/upload")
async def upload_report(
    request: Request,
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    report_date: str = Form(default=""),
    report_id: str = Form(default=""),
    language: str = Form(default="en"),
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_roles(ROLE_CLINICIAN)),
):
    ip = _ip(request)
    ok, retry_after = rate_limit_allow(
        principal.subject, UPLOAD_RATE_LIMIT_MAX, UPLOAD_RATE_LIMIT_WINDOW_SECONDS
    )
    if not ok:
        audit.record(actor=principal.subject, role=principal.role, action="rate_limit.exceeded",
                     resource="/report/upload", outcome="denied", ip=ip)
        raise HTTPException(
            429,
            detail=f"Rate limit exceeded ({UPLOAD_RATE_LIMIT_MAX} uploads per "
                   f"{UPLOAD_RATE_LIMIT_WINDOW_SECONDS}s). Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    patient_id = _valid_patient_id(patient_id)
    language = (language or "en").lower()
    if language != "en" and not is_supported(language):
        raise HTTPException(400, f"Unsupported language '{language}'. Use 'en' or one of: {', '.join(LANGUAGES)}.")

    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, "Only PDF, JPG, PNG or WebP files are supported.")

    # Use the client-supplied report_id (so the WebSocket connection the
    # frontend opened BEFORE upload matches the ID used for step callbacks).
    # Fall back to generating one if the client didn't send it.
    report_id = report_id or str(uuid.uuid4())
    if not re.match(r"^[A-Za-z0-9\-]{1,64}$", report_id):
        raise HTTPException(400, "Invalid report_id.")
    if jobq.get(db, report_id) or db.query(Report).filter(Report.id == report_id).first():
        raise HTTPException(409, "A report with this id already exists.")

    # Save to temp file, enforcing a size cap (previously unbounded — a large
    # upload could tie up OCR/NER/LLM processing indefinitely). The original
    # extension is kept so the loader can tell a PDF from a photo.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=_spool_dir()) as tmp:
        tmp_path = tmp.name
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.")
            tmp.write(chunk)

    try:
        # The extension is attacker-controlled; the bytes are what the parsers will see.
        if not extension_matches_content(filename, sniff_file_type(tmp_path)):
            audit.record(actor=principal.subject, role=principal.role, action="upload.rejected_content_mismatch",
                         resource=f"patient:{patient_id}", outcome="denied", ip=ip, detail={"ext": ext})
            raise HTTPException(400, "File content does not match its extension.")
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    # Everything slow (OCR, NER, Gemini, DB write) happens in a background job. The file stays in
    # the spool until that job reaches a terminal state, then the worker deletes it.
    if jobq.active_count(db) >= _env_num("JOB_QUEUE_MAX", 200, int):
        os.unlink(tmp_path)
        raise HTTPException(503, "Too many uploads in progress. Retry shortly.",
                            headers={"Retry-After": "30"})
    try:
        row = jobq.enqueue(
            db, job_id=report_id, owner=principal.subject, ip=ip, patient_id=patient_id, filename=filename,
            ext=ext, report_date=report_date, language=language, spool_path=tmp_path,
            max_attempts=_env_num("JOB_MAX_ATTEMPTS", 3, int))
    except jobq.DuplicateJob:
        os.unlink(tmp_path)
        raise HTTPException(409, "A report with this id already exists.")
    except BaseException:
        os.unlink(tmp_path)
        raise
    audit.record(actor=principal.subject, role=principal.role, action="report.upload_queued",
                 resource=f"report:{report_id}", ip=ip, detail={"patient_id": patient_id})
    obs.inc("uploads_accepted_total")
    if _worker is not None:
        _worker.wake()
    return JSONResponse(
        status_code=202,
        content={**jobq.job_view(row), "status_url": f"/report/{report_id}/job"},
        headers={"Location": f"/report/{report_id}/job"},
    )


def _call_cost_usd(tokens_in: int, tokens_out: int) -> float:
    """NOTIONAL cost: what these tokens would cost at paid-tier prices. On the free tier the real
    spend is zero; the number exists so a prompt change can be judged by what it would cost at
    scale. Prices are per million tokens and configurable (LLM_PRICE_INPUT_PER_M / _OUTPUT_PER_M)."""
    return round((tokens_in * _env_num("LLM_PRICE_INPUT_PER_M", 0.30)
                  + tokens_out * _env_num("LLM_PRICE_OUTPUT_PER_M", 2.50)) / 1_000_000, 8)


def _persist_llm_calls(report_id: str, calls: list[dict]) -> None:
    """Write the model-call trace. Best effort: tracing must never fail an upload."""
    if not calls:
        return
    db = SessionLocal()
    try:
        db.add_all([LlmCall(
            report_id=report_id, purpose=c["purpose"], model=c["model"],
            prompt_version=prompts.version_of(c["purpose"]), attempt=c["attempt"], status=c["status"],
            error_class=c["error_class"], latency_ms=c["latency_ms"], input_tokens=c["input_tokens"],
            output_tokens=c["output_tokens"], cost_usd=_call_cost_usd(c["input_tokens"], c["output_tokens"]),
        ) for c in list(calls)])
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("could not record the model-call trace for %s", report_id)
    finally:
        db.close()


def _flag_severity(f: dict) -> str:
    return str(f.get("severity", "")).replace("Severity.", "").upper()


def _save_preliminary(report_id: str, patient_id: str, filename: str | None, report_date: str,
                      source_type: str, pages: int, uploaded_by: str, data: dict) -> bool:
    """Write the deterministic results as a 'preliminary' report. Runs before any model call, so a
    critical value is on a clinician's screen within seconds of the upload and stays there if the
    model provider is down. Idempotent: returns False if a row already exists (a retry)."""
    db = SessionLocal()
    try:
        if db.query(Report.id).filter(Report.id == report_id).first():
            return False
        if not db.query(Patient.id).filter(Patient.id == patient_id).first():
            db.add(Patient(id=patient_id))
        flags = data.get("rule_flags", [])
        entities = data.get("entities", {})
        db.add(Report(
            id=report_id, patient_id=patient_id, filename=filename, report_date=report_date,
            source_type=source_type, pages=pages, entities=entities, rule_flags=flags,
            trend_flags=data.get("trend_flags", []), baseline_flags=data.get("baseline_flags", []),
            risk_summary=data.get("risk_summary", {}), lab_values=entities.get("lab_values", []),
            ingestion_warnings=list(data.get("ingestion_warnings", [])) + [
                "PRELIMINARY: lab flags and drug interactions only. The summaries have not been "
                "generated yet and this report cannot be approved until they are."],
            security_warnings=[], drug_interactions=data.get("drug_interactions", {}) or {},
            # Never released to a patient from here, whatever REQUIRE_CLINICIAN_REVIEW says.
            review_status="pending_review", analysis_status="preliminary",
            has_critical=any(_flag_severity(f) == "CRITICAL" for f in flags),
            uploaded_by=uploaded_by,
        ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return False
        obs.inc("reports_preliminary_total")
        return True
    finally:
        db.close()


def _defer_if_ai_down(job: jobq.JobSnapshot, report_id: str) -> None:
    """If the model provider is considered down, wait for it instead of producing a report with
    no summaries. The preliminary report is already visible meanwhile. After JOB_DEFER_MAX_SECONDS
    we stop waiting and finish with the degraded (clearly labelled) output rather than never."""
    wait = llm_unavailable_for()
    if wait <= 0:
        return
    limit = _env_num("JOB_DEFER_MAX_SECONDS", 3600.0)
    age = (jobq.utcnow() - job.created_at).total_seconds() if job.created_at else 0.0
    if age >= limit:
        logger.warning("AI service still unavailable after %.0fs; finishing %s without waiting", age, report_id)
        return
    raise jobq.DeferJob(min(max(wait, 5.0), 300.0) + secrets.randbelow(5), "AI service unavailable")


async def _process_upload(job: jobq.JobSnapshot, report_progress) -> None:
    """The analysis pipeline for one queued upload, run by a JobWorker with its own DB session.

    Failure handling lives in the worker: this raises, the worker retries with backoff and finally
    fails the job. Delivery is at-least-once, so this must be safe to run twice for the same
    upload: it returns early if an earlier attempt already saved the report, and the final write
    treats a primary-key clash (a second worker got there first) as success."""
    report_id, patient_id, ext = job.id, job.patient_id, job.ext or ""
    filename, report_date, language = job.filename, job.report_date or "", job.language
    tmp_path, ip = job.spool_path, job.ip or "unknown"
    principal = Principal(subject=job.owner, role=ROLE_CLINICIAN, patient_id=None)
    started = time.monotonic()
    models_used = begin_model_tracking()   # every model that answers during this upload
    llm_trace = begin_call_tracking(max_calls=_env_num("LLM_MAX_CALLS_PER_REPORT", 12, int))
    caller_fp = fingerprint(principal.subject)
    ws_callback = _make_job_callback(report_id, report_progress)
    db = SessionLocal()
    try:
        existing = db.query(Report.id, Report.analysis_status).filter(Report.id == report_id).first()
        if existing and (existing.analysis_status or "complete") != "preliminary":
            logger.info("report already saved by an earlier attempt; nothing to redo")
            return

        # ── Step 1: Ingest ─────────────────────────────────────────────────
        try:
            doc = await asyncio.to_thread(load_document, tmp_path)
        except Exception as exc:
            # An unreadable file will be just as unreadable on the next attempt: fail now.
            raise jobq.PermanentJobError("document could not be read") from exc
        raw_text = doc["raw_text"]
        raw_text = doc["raw_text"]
        clean_date = str(report_date).split(" ")[0].split("T")[0] if report_date else ""

        ingestion_extra: list[str] = []
        security_extra: list[str] = []
        conf = doc.get("ocr_confidence")
        if conf is not None and conf < LOW_OCR_CONFIDENCE:
            ingestion_extra.append(
                f"Low OCR confidence ({conf:.0%}) - digits may have been misread. "
                f"Verify every value against the original document."
            )
        if ext in PDF_EXTS:
            security_extra += await asyncio.to_thread(detect_hidden_pdf_text, tmp_path)

        # ── Step 2: Orchestrator agent (tool calling + ReAct) ──────────────
        ws_callback({"type": "pipeline_start", "report_id": report_id})

        agent_data = await run_orchestrator(
            report_text=raw_text,
            patient_id=patient_id,
            report_date=clean_date,
            on_step=ws_callback,
            on_preliminary=lambda snap: asyncio.to_thread(
                _save_preliminary, report_id, patient_id, filename, clean_date,
                doc["source_type"], doc["pages"], principal.subject, snap),
        )

        # Pop the redaction map out immediately — it exists only to keep
        # patient identifiers out of Gemini calls (see deidentification.py)
        # and must never be logged, persisted, or included in agent_data
        # past this point. It's threaded explicitly into the next agent
        # calls, then consumed once by rehydrate() below.
        redaction_map = agent_data.pop("_redaction_map", {})

        # Log pipeline shape only — never patient content (names, lab
        # values, diagnoses, raw report text).
        logger.info(
            "orchestrator complete report_id=%s entity_keys=%s rule_flags=%d "
            "trend_flags=%d tools=%s",
            report_id,
            list(agent_data.get("entities", {}).keys()),
            len(agent_data.get("rule_flags", [])),
            len(agent_data.get("trend_flags", [])),
            [t["tool"] for t in agent_data.get("tool_calls_log", [])],
        )

        entities    = agent_data.get("entities", {})
        rule_flags  = agent_data.get("rule_flags", [])
        trend_flags = agent_data.get("trend_flags", [])
        baseline_flags = agent_data.get("baseline_flags", [])
        forecast_flags = agent_data.get("forecast_flags", [])
        similar_history = agent_data.get("similar_history", [])
        risk_summary = agent_data.get("risk_summary", {})
        pubmed_refs = agent_data.get("pubmed_references", [])
        drug_safety = agent_data.get("drug_safety", [])
        interactions = agent_data.get("drug_interactions", {}) or {}
        interaction_list = interactions.get("interactions", [])
        ingestion_warnings = list(agent_data.get("ingestion_warnings", [])) + ingestion_extra
        security_warnings = list(agent_data.get("security_warnings", [])) + security_extra
        llm_calls = int(agent_data.get("llm_calls", 0))

        # Provider down: keep the preliminary report, wait, and do the model steps later.
        _defer_if_ai_down(job, report_id)

        # ── Step 3: Generate summaries ─────────────────────────────────────
        # NOTE: summaries returned here still contain redaction placeholders
        # (e.g. "[REDACTED_NAME_1]") in place of the patient's real name —
        # see deidentification.py. Do not use these strings user-facing
        # until rehydrate() has run, below.
        ws_callback({"type": "generating_summaries"})
        summaries = await generate_summaries(
            entities, rule_flags, trend_flags, pubmed_refs, drug_safety, baseline_flags,
            redaction_map=redaction_map, drug_interactions=interaction_list,
        )
        llm_calls += 1

        # ── Step 4: Verification agent ─────────────────────────────────────
        ws_callback({"type": "verifier_start"})
        verified = await verify_and_correct(
            clinical_summary=summaries["clinical_summary"],
            patient_summary=summaries["patient_summary"],
            entities=entities,
            flags=rule_flags,
            trend_flags=trend_flags,
            redaction_map=redaction_map,
            drug_interactions=interaction_list,
        )
        llm_calls += 1
        ws_callback({"type": "verifier_complete",
                     "clinical_corrected": verified["clinical_corrected"],
                     "patient_corrected": verified["patient_corrected"]})

        # The provider may have failed during the steps above (they degrade instead of raising).
        _defer_if_ai_down(job, report_id)

        # ── Step 4b: deterministic numeric-grounding check ─────────────────
        # Independent of the (LLM) Verifier: every number in a summary must exist
        # somewhere in the structured ground truth.
        allowed = collect_allowed_numbers(
            entities, rule_flags, trend_flags, baseline_flags, forecast_flags,
            pubmed_refs, interaction_list, extra_texts=[clean_date],
        )
        grounding_warnings: list[str] = []
        for label, text in (("Clinical summary", verified["clinical_summary"]),
                            ("Patient summary", verified["patient_summary"]),
                            ("Longitudinal narrative", summaries.get("longitudinal_narrative", ""))):
            bad = find_ungrounded_numbers(text, allowed)
            if bad:
                grounding_warnings.append(
                    f"{label} contains number(s) not found in the extracted data: {', '.join(bad)}"
                )

        # ── Step 4c: optional translation of the (audited) patient summary ─
        translation = None
        if language != "en":
            ws_callback({"type": "translating", "language": language})
            translation = await translate_patient_summary(verified["patient_summary"], language)
            llm_calls += 1
            if not translation["ok"]:
                ingestion_warnings.append(
                    f"{LANGUAGES[language]['name']} translation was rejected by automated checks "
                    f"({'; '.join(translation['issues'])}) - the English summary is shown instead."
                )

        # Rehydrate — the ONE point in the whole pipeline where redaction
        # placeholders are converted back to the patient's real identifiers,
        # now that every Gemini call for this report is done. Everything
        # from here on (DB row, API response) uses real values again — Gemini
        # itself just never saw them.
        if redaction_map:
            verified["clinical_summary"] = rehydrate(verified["clinical_summary"], redaction_map)
            verified["patient_summary"] = rehydrate(verified["patient_summary"], redaction_map)
            summaries["longitudinal_narrative"] = rehydrate(
                summaries.get("longitudinal_narrative", ""), redaction_map
            )
            summaries["followup_actions"] = [
                rehydrate(a, redaction_map) for a in summaries.get("followup_actions", [])
            ]
            summaries["explained_flags"] = [
                {**f, "explanation": rehydrate(f.get("explanation", ""), redaction_map)}
                for f in summaries.get("explained_flags", [])
            ]
            if translation and translation.get("text"):
                translation["text"] = rehydrate(translation["text"], redaction_map)

        # Output-side guardrail: scan the (now-verified) summaries for signs
        # the model complied with an injected instruction. Independent of
        # the Verifier agent's hallucination audit above — different threat
        # model (adversarial input vs. ordinary model error).
        security_warnings += [
            f"Clinical summary: {w}" for w in scan_output_for_leakage(verified["clinical_summary"])
        ]
        security_warnings += [
            f"Patient summary: {w}" for w in scan_output_for_leakage(verified["patient_summary"])
        ]
        if security_warnings:
            logger.warning(
                "security_warnings report_id=%s caller=%s count=%d",
                report_id, caller_fp, len(security_warnings),
            )
            audit.record(actor=principal.subject, role=principal.role, action="security.injection_suspected",
                         resource=f"report:{report_id}", ip=ip, outcome="success",
                         detail={"warnings": len(security_warnings)})

        # ── Evidence: link each extracted fact to its source line ──────────
        evidence = build_evidence(
            raw_text, entities, rule_flags,
            summaries={"clinical": verified["clinical_summary"], "patient": verified["patient_summary"]},
        )

        # ── Step 5: Persist ────────────────────────────────────────────────
        ws_callback({"type": "saving"})
        patient =db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            patient = Patient(id=patient_id)
            db.add(patient)

        review_required = _require_review()
        lab_values = entities.get("lab_values", [])
        ver_blob = {
            "clinical_corrected": verified["clinical_corrected"],
            "patient_corrected": verified["patient_corrected"],
            "clinical_unverified": verified.get("clinical_unverified", False),
            "patient_unverified": verified.get("patient_unverified", False),
            "clinical_issues": verified.get("clinical_issues", ""),
            "patient_issues": verified.get("patient_issues", ""),
        }
        # The preliminary row (saved earlier in this attempt or in a deferred one) is replaced in
        # place: same id, original timestamp.
        prelim = db.query(Report.created_at).filter(
            Report.id == report_id, Report.analysis_status == "preliminary").first()
        prelim_created_at = prelim.created_at if prelim else None
        if prelim:
            db.query(Report).filter(Report.id == report_id, Report.analysis_status == "preliminary").delete(
                synchronize_session=False)
            db.flush()
        report = Report(
            id=report_id, created_at=prelim_created_at or datetime.utcnow(),
            analysis_status="complete",
            has_critical=any(_flag_severity(f) == "CRITICAL" for f in rule_flags),
            patient_id=patient_id, filename=filename, report_date=clean_date,
            source_type=doc["source_type"], has_hindi=str(doc["has_hindi"]), pages=doc["pages"],
            entities=entities, rule_flags=rule_flags, trend_flags=trend_flags,
            baseline_flags=baseline_flags, forecast_flags=forecast_flags,
            similar_history=similar_history, risk_summary=risk_summary, lab_values=lab_values,
            ingestion_warnings=ingestion_warnings, security_warnings=security_warnings,
            clinical_summary=verified["clinical_summary"], patient_summary=verified["patient_summary"],
            explained_flags=summaries["explained_flags"], followup_actions=summaries["followup_actions"],
            longitudinal_narrative=summaries.get("longitudinal_narrative", ""),
            verification=ver_blob,
            review_status="pending_review" if review_required else "approved",
            reviewed_by=None if review_required else "auto (review not required)",
            reviewed_at=None if review_required else datetime.utcnow(),
            uploaded_by=principal.subject,
            evidence=evidence, grounding_warnings=grounding_warnings,
            drug_interactions=interactions, drug_safety=drug_safety, pubmed_references=pubmed_refs,
            patient_summary_translated=translation["text"] if translation and translation.get("text") else None,
            translation_language=language if translation and translation.get("text") else None,
            translation_check=({"ok": translation["ok"], "issues": translation["issues"]}
                               if translation and translation.get("text") else None),
        )
        db.add(report)
        db.flush()  # agent_runs.report_id is a FK; no ORM relationship orders the inserts
        outbox.add(db, kind=outbox.HISTORY_INDEX, report_id=report_id, patient_id=patient_id, payload={
            # Redacted text only: names, phone numbers, MRNs etc. never reach the vector store.
            "text": _history_text(redact(raw_text, redaction_map)), "lab_values": lab_values,
            "report_date": clean_date})
        db.add(AgentRun(
            report_id=report_id, agent_name="orchestrator+summarizer+verifier",
            tool_calls=agent_data.get("tool_calls_log", []),
            was_corrected=str(verified["clinical_corrected"] or verified["patient_corrected"]),
            caller_fingerprint=caller_fp,
            duration_ms=int((time.monotonic() - started) * 1000), llm_calls=llm_calls,
            # e.g. "gemini-3.8-flash" or, if the fallback answered any call, both models
            model=",".join(sorted(models_used)) or (agent_data.get("token_usage") or {}).get("model"),
            input_tokens=sum(c["input_tokens"] for c in llm_trace),
            output_tokens=sum(c["output_tokens"] for c in llm_trace),
            prompt_versions={k: f"{k}@{v}" for k, v in prompts.current_versions().items()},
        ))
        try:
            db.commit()
        except IntegrityError:
            # Another worker saved this report first (lease handed over mid-run). That is a success.
            db.rollback()
            if db.query(Report.id).filter(Report.id == report_id).first():
                logger.warning("report saved concurrently by another worker; finishing")
                return
            raise

        # ── Step 6: apply the history-store write staged in the outbox ──────
        # Best effort right now; if it fails the event stays in the outbox and the relay retries it
        # (a failure here must never become a 500: the report is saved and a re-upload would
        # duplicate it). _outbox_result records the outcome on the report.
        cfg = _outbox_config()
        await asyncio.to_thread(
            outbox.drain, SessionLocal, _apply_history_event, lease_seconds=cfg["lease_seconds"],
            max_attempts=cfg["max_attempts"], base_seconds=cfg["base_seconds"], report_id=report_id,
            on_result=_outbox_result)

        audit.record(actor=principal.subject, role=principal.role, action="report.upload",
                     resource=f"report:{report_id}", ip=ip,
                     detail={"patient_id": patient_id, "source_type": doc["source_type"],
                             "pages": doc["pages"], "llm_calls": llm_calls,
                             "review_status": report.review_status})
        obs.inc("reports_analysed_total")
    finally:
        _persist_llm_calls(report_id, llm_trace)    # also on failure, deferral and cancellation
        db.close()


# ── Review (human-in-the-loop release) ─────────────────────────────────────────

class ReviewIn(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=1000)
    # Approve-with-edits: the clinician's corrected text replaces the model's (which is kept).
    edited_clinical_summary: str | None = Field(default=None, max_length=5000)
    edited_patient_summary: str | None = Field(default=None, max_length=5000)
    # Required when the report carries any warning (see _review_warnings).
    acknowledge_warnings: bool = False


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


def _review_warnings(r: Report) -> list[str]:
    """Everything a clinician must explicitly accept before a report can be released.
    Without this gate the warnings were logged but approval was one click either way —
    the shape that turns review into rubber-stamping."""
    ver = r.verification or {}
    out = []
    if ver.get("clinical_unverified") or ver.get("patient_unverified"):
        out.append("The AI Verifier could not run: the summaries have NOT been audited.")
    out += [f"Grounding: {w}" for w in (r.grounding_warnings or [])]
    out += [f"Security: {w}" for w in (r.security_warnings or [])]
    out += [f"Extraction: {w}" for w in (r.ingestion_warnings or [])]
    return out


@app.post("/report/{report_id}/review")
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

    previous = _effective_status(r)
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
    warnings = _review_warnings(r)
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
                 resource=f"report:{report_id}", ip=_ip(request), critical=True,
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
            "reviewed_at": _iso(r.reviewed_at), "edited": sorted(edits)}


@app.get("/reviews/pending")
def pending_reviews(db: Session = Depends(get_db),
                    principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    rows = (db.query(Report)
            .filter((Report.review_status == "pending_review") | (Report.review_status.is_(None)))
            .order_by(Report.created_at.desc()).limit(200).all())
    rows.sort(key=lambda r: not r.has_critical)   # critical results first; stable, so newest-first within
    return {"count": len(rows), "reports": [{
        "analysis_status": r.analysis_status or "complete", "has_critical": bool(r.has_critical),
        "report_id": r.id, "patient_id": r.patient_id, "filename": r.filename,
        "report_date": r.report_date, "created_at": _iso(r.created_at),
        "overall_severity": (r.risk_summary or {}).get("overall_severity"),
        "security_warnings": len(r.security_warnings or []),
        "grounding_warnings": len(r.grounding_warnings or []),
        "interaction_alerts": len((r.drug_interactions or {}).get("interactions", [])),
    } for r in rows]}


@app.get("/reviews/pending/export.pdf")
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
            "overall_severity": (r.risk_summary or {}).get("overall_severity"),
            "flags": flags,
            "interactions": (r.drug_interactions or {}).get("interactions", []),
            "warnings": [clean(w) for w in _review_warnings(r)],
            "clinical_summary": clean(r.clinical_summary),
            "patient_summary": clean(r.patient_summary),
        })
    pdf = build_queue_pdf(items, generated_by=principal.subject, include_names=include_names)
    audit.record(actor=principal.subject, role=principal.role, action="reviews.export_pdf",
                 resource="reviews:pending", ip=_ip(request), critical=True,
                 detail={"reports": len(items), "include_names": include_names})
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    return Response(content=pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="review-queue-{stamp}.pdf"',
        "Cache-Control": "no-store",
    })


# ── Report access ──────────────────────────────────────────────────────────────

@app.get("/report/{report_id}")
def get_report(report_id: str, request: Request, db: Session = Depends(get_db),
               principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    r = _load_report_for(principal, report_id, db)
    audit.record(actor=principal.subject, role=principal.role, action="report.view",
                 resource=f"report:{report_id}", ip=_ip(request), detail={"patient_id": r.patient_id})
    if principal.role == ROLE_PATIENT:
        return _report_patient_view(r)
    return _report_full(r, db)


@app.post("/report/{report_id}/translate")
async def translate_report(report_id: str, request: Request,
                           language: str = Query(..., min_length=2, max_length=5),
                           db: Session = Depends(get_db),
                           principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    """Translate the plain-language patient summary on demand. The text is
    de-identified again before it reaches Gemini, exactly like the main pipeline."""
    language = language.lower()
    if not is_supported(language):
        raise HTTPException(400, f"Unsupported language. Use one of: {', '.join(LANGUAGES)}.")
    r = _load_report_for(principal, report_id, db)
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
                 resource=f"report:{report_id}", ip=_ip(request),
                 outcome="success" if text else "failure", detail={"language": language})
    if not text:
        raise HTTPException(502, "Translation could not be produced or failed automated safety checks: "
                                 + "; ".join(result.get("issues", [])))
    return {"report_id": report_id, "language": language, "text": text}


@app.get("/report/{report_id}/fhir")
def export_fhir(report_id: str, request: Request, include_name: bool = False,
                db: Session = Depends(get_db),
                principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    r = db.query(Report).filter(Report.id == report_id).first()
    if not r:
        raise HTTPException(404, "Report not found.")
    audit.record(actor=principal.subject, role=principal.role, action="report.export_fhir",
                 resource=f"report:{report_id}", ip=_ip(request),
                 detail={"include_name": include_name, "review_status": _effective_status(r)})
    bundle = build_bundle({
        "report_id": r.id, "patient_id": r.patient_id, "report_date": r.report_date,
        "review_status": _effective_status(r), "entities": r.entities or {},
        "rule_flags": r.rule_flags or [], "clinical_summary": r.clinical_summary,
        "filename": r.filename,
    }, include_name=include_name)
    return bundle


# ── Patient endpoints ──────────────────────────────────────────────────────────

def _visible_reports(principal: Principal, patient_id: str, db: Session) -> list[Report]:
    q = db.query(Report).filter(Report.patient_id == patient_id)
    if principal.role == ROLE_PATIENT:
        q = q.filter(Report.review_status == "approved")
    return q.order_by(Report.report_date).all()


@app.get("/patient/{patient_id}/history")
def patient_history(patient_id: str, request: Request, db: Session = Depends(get_db),
                    principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    _assert_patient_scope(principal, patient_id)
    reports = _visible_reports(principal, patient_id, db)
    audit.record(actor=principal.subject, role=principal.role, action="patient.history_view",
                 resource=f"patient:{patient_id}", ip=_ip(request))
    return {
        "patient_id": patient_id,
        "report_count": len(reports),
        "reports": [
            {
                "report_id": r.id,
                "filename": r.filename,
                "report_date": r.report_date,
                "risk_summary": r.risk_summary,
                "lab_values": r.lab_values,
                "review_status": _effective_status(r),
                "diagnoses": r.entities.get("clinical", {}).get("diagnoses", []) if r.entities else [],
            }
            for r in reports
        ],
    }


@app.get("/patient/{patient_id}/trends")
def patient_trends(patient_id: str, db: Session = Depends(get_db),
                   principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_PATIENT))):
    """Chart-ready lab series (with the ICMR reference band) across a patient's reports."""
    _assert_patient_scope(principal, patient_id)
    reports = _visible_reports(principal, patient_id, db)
    series = build_trend_series([{
        "report_id": r.id, "report_date": r.report_date, "lab_values": r.lab_values or [],
        "gender": (r.entities or {}).get("patient", {}).get("gender"),
    } for r in reports])
    return {"patient_id": patient_id, "report_count": len(reports), "series": series}


@app.get("/me/reports")
def my_reports(db: Session = Depends(get_db),
               principal: Principal = Depends(require_roles(ROLE_PATIENT))):
    reports = _visible_reports(principal, principal.patient_id, db)
    return {"patient_id": principal.patient_id, "reports": [{
        "report_id": r.id, "report_date": r.report_date, "filename": r.filename,
        "overall_severity": (r.risk_summary or {}).get("overall_severity"),
    } for r in reports]}


@app.get("/patients")
def list_patients(db: Session = Depends(get_db),
                  principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    patients = db.query(Patient).all()
    return {"patients": [p.id for p in patients]}


@app.delete("/patient/{patient_id}")
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
                 resource=f"patient:{patient_id}", ip=_ip(request), critical=True,
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


# ── Operations: metrics + audit ────────────────────────────────────────────────

def _percentile(values: list[float], pct: float) -> int | None:
    if not values:
        return None
    s = sorted(values)
    return int(s[min(len(s) - 1, int(round(pct * (len(s) - 1))))])


@app.get("/admin/ai-quality")
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


@app.get("/admin/llm-usage")
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


@app.get("/admin/metrics")
def metrics(db: Session = Depends(get_db),
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
        status_counts[_effective_status(r)] = status_counts.get(_effective_status(r), 0) + 1
        v = r.verification or {}
        corrected += bool(v.get("clinical_corrected") or v.get("patient_corrected"))
        unverified += bool(v.get("clinical_unverified") or v.get("patient_unverified"))
        with_security += bool(r.security_warnings)
        with_grounding += bool(r.grounding_warnings)
        with_interactions += bool((r.drug_interactions or {}).get("interactions"))

    # ── Product: is review real, and is the tool saving clinicians time? ──────────
    # approve_without_edit_rate high AND minutes_to_review very low is the rubber-stamp
    # signature (automation bias); a high edit rate means the summaries aren't good enough yet.
    reviewed = [r for r in reports if _effective_status(r) in ("approved", "rejected") and r.reviewed_at]
    approved = [r for r in reviewed if _effective_status(r) == "approved"]
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


@app.get("/admin/audit")
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


@app.get("/admin/audit/verify")
def audit_verify(db: Session = Depends(get_db),
                 principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    result = audit.verify_chain(db)
    audit.record(actor=principal.subject, role=principal.role, action="audit.verify",
                 outcome="success" if result["ok"] else "failure",
                 detail={"checked": result["checked"], "ok": result["ok"]})
    return result
