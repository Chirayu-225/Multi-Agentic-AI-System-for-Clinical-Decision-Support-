"""
main.py
Agentic Medical Assistant — FastAPI backend.

Endpoints
  Auth        POST /auth/login · GET /auth/me
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
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# Load .env by an explicit path next to this file, before any of the
# src.* imports below — several of them (src.agents.orchestrator,
# summarizer, verifier) read GEMINI_API_KEY at MODULE IMPORT TIME to
# configure the Gemini client, so .env must be loaded before those imports
# run, not just before the imports are used.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Query, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.session import get_db, create_tables, SessionLocal
from src.db.models import Patient, Report, AgentRun, User, AuditEvent
from src.ingestion.document_loader import (
    load_document, ALLOWED_EXTS, PDF_EXTS, sniff_file_type, extension_matches_content,
)
from src.agents.orchestrator import run_orchestrator
from src.agents.summarizer import generate_summaries
from src.agents.verifier import verify_and_correct
from src.agents.translator import (LANGUAGES, is_supported, translate_patient_summary)
from src.tools.history_store import store_report as chroma_store
from src.tools.trend_series import build_trend_series
from src.api import auth as authmod
from src.api.auth import (Principal, ROLE_CLINICIAN, ROLE_PATIENT, ROLE_SERVICE,
                          get_principal, require_roles, verify_ws_auth, fingerprint)
from src.security import audit
from src.security.passwords import (DUMMY_HASH, WeakPassword, hash_password,
                                    validate_password_strength, verify_password)
from src.security.rate_limit import allow as rate_limit_allow
from src.security.llm_guardrails import scan_output_for_leakage, detect_hidden_pdf_text
from src.security.deidentification import rehydrate, redact, build_redaction_map
from src.evidence.grounding import build_evidence, collect_allowed_numbers, find_ungrounded_numbers
from src.interop.fhir import build_bundle

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:4200").split(","),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Authorization", "Content-Type"],
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
    required = ["GEMINI_API_KEY"]
    if os.getenv("ALLOW_API_KEY_AUTH", "true").lower() != "false":
        required.append("API_KEY")
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"See backend/.env.example."
        )
    authmod.validate_auth_config()
    create_tables()
    _bootstrap_users()


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
    # Deliberately minimal and unauthenticated (load balancers hit this).
    return {"status": "ok", "version": APP_VERSION}


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
    token, ttl = authmod.create_access_token(username=user.username, role=user.role,
                                             patient_id=user.patient_id)
    audit.record(actor=user.username, role=user.role, action="auth.login", ip=ip)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl,
            "role": user.role, "username": user.username,
            "display_name": user.display_name or user.username, "patient_id": user.patient_id}


@app.get("/auth/me")
def whoami(principal: Principal = Depends(get_principal)):
    return {"username": principal.subject, "role": principal.role,
            "patient_id": principal.patient_id, "via": principal.via}


# ── WebSocket — real-time agent streaming ──────────────────────────────────────

@app.websocket("/ws/{report_id}")
async def websocket_endpoint(websocket: WebSocket, report_id: str):
    """
    WebSocket endpoint for real-time agent step streaming (staff only — patient
    accounts have no business watching the analysis pipeline).

    Auth: browsers' native WebSocket API can't set custom headers, so the credential
    travels as a query param instead: `?token=<JWT>` or `?api_key=...`. Use wss://
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
                   "reviewed_at": _iso(r.reviewed_at), "note": r.review_note},
        "llm_output": {
            "clinical_summary": r.clinical_summary, "patient_summary": r.patient_summary,
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
    principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_SERVICE)),
):
    started = time.monotonic()
    ip = _ip(request)
    caller_fp = fingerprint(principal.subject)
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
    if db.query(Report).filter(Report.id == report_id).first():
        raise HTTPException(409, "A report with this id already exists.")

    # Save to temp file, enforcing a size cap (previously unbounded — a large
    # upload could tie up OCR/NER/LLM processing indefinitely). The original
    # extension is kept so the loader can tell a PDF from a photo.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
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

        # ── Step 1: Ingest ─────────────────────────────────────────────────
        doc = await asyncio.to_thread(load_document, tmp_path)
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
            security_extra += detect_hidden_pdf_text(tmp_path)

        # ── Step 2: Orchestrator agent (tool calling + ReAct) ──────────────
        ws_callback = _make_ws_callback(report_id)
        ws_callback({"type": "pipeline_start", "report_id": report_id})

        agent_data = await run_orchestrator(
            report_text=raw_text,
            patient_id=patient_id,
            report_date=clean_date,
            on_step=ws_callback,
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
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
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
        report = Report(
            id=report_id, patient_id=patient_id, filename=filename, report_date=clean_date,
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
        db.add(AgentRun(
            report_id=report_id, agent_name="orchestrator+summarizer+verifier",
            tool_calls=agent_data.get("tool_calls_log", []),
            was_corrected=str(verified["clinical_corrected"] or verified["patient_corrected"]),
            caller_fingerprint=caller_fp,
            duration_ms=int((time.monotonic() - started) * 1000), llm_calls=llm_calls,
        ))
        db.commit()

        # ── Step 6: Store in ChromaDB for longitudinal tracking ─────────────
        chroma_store(
            patient_id=patient_id, report_text=raw_text, lab_values=lab_values,
            report_date=clean_date, filename=filename,
        )

        audit.record(actor=principal.subject, role=principal.role, action="report.upload",
                     resource=f"report:{report_id}", ip=ip,
                     detail={"patient_id": patient_id, "source_type": doc["source_type"],
                             "pages": doc["pages"], "llm_calls": llm_calls,
                             "review_status": report.review_status})
        ws_callback({"type": "pipeline_complete", "report_id": report_id})

        db.refresh(report)
        return _report_full(report, db)

    except HTTPException:
        raise
    except Exception:
        incident_id = str(uuid.uuid4())[:8]
        # Full detail (including traceback) is logged server-side only —
        # the client gets a generic message plus a correlation id, not the
        # raw exception string, which previously could reflect internal
        # paths/data back to the caller.
        logger.exception("Upload failed [incident=%s] report_id=%s", incident_id, report_id)
        audit.record(actor=principal.subject, role=principal.role, action="report.upload",
                     resource=f"report:{report_id}", outcome="failure", ip=ip,
                     detail={"incident": incident_id})
        raise HTTPException(
            500,
            detail=f"Processing failed. Reference: {incident_id}. Check server logs for detail.",
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Review (human-in-the-loop release) ─────────────────────────────────────────

class ReviewIn(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=1000)


@app.post("/report/{report_id}/review")
def review_report(report_id: str, body: ReviewIn, request: Request, db: Session = Depends(get_db),
                  principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Approve or reject an AI-generated report. Clinician accounts only: the shared
    service key is deliberately NOT accepted, so every release decision is attributable
    to a named person."""
    r = db.query(Report).filter(Report.id == report_id).first()
    if not r:
        raise HTTPException(404, "Report not found.")
    if body.decision == "reject" and len(body.note.strip()) < 3:
        raise HTTPException(422, "A reason is required when rejecting a report.")

    previous = _effective_status(r)
    new_status = "approved" if body.decision == "approve" else "rejected"
    # Audit FIRST and fail closed: a release decision that cannot be recorded must not happen.
    audit.record(actor=principal.subject, role=principal.role, action=f"report.{body.decision}",
                 resource=f"report:{report_id}", ip=_ip(request), critical=True,
                 detail={"patient_id": r.patient_id, "previous_status": previous,
                         "had_security_warnings": bool(r.security_warnings),
                         "had_grounding_warnings": bool(r.grounding_warnings)})
    r.review_status = new_status
    r.reviewed_by = principal.subject
    r.reviewed_at = datetime.utcnow()
    r.review_note = body.note.strip() or None
    db.commit()
    return {"report_id": report_id, "status": new_status, "reviewed_by": principal.subject,
            "reviewed_at": _iso(r.reviewed_at)}


@app.get("/reviews/pending")
def pending_reviews(db: Session = Depends(get_db),
                    principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    rows = (db.query(Report)
            .filter((Report.review_status == "pending_review") | (Report.review_status.is_(None)))
            .order_by(Report.created_at.desc()).limit(200).all())
    return {"count": len(rows), "reports": [{
        "report_id": r.id, "patient_id": r.patient_id, "filename": r.filename,
        "report_date": r.report_date, "created_at": _iso(r.created_at),
        "overall_severity": (r.risk_summary or {}).get("overall_severity"),
        "security_warnings": len(r.security_warnings or []),
        "grounding_warnings": len(r.grounding_warnings or []),
        "interaction_alerts": len((r.drug_interactions or {}).get("interactions", [])),
    } for r in rows]}


# ── Report access ──────────────────────────────────────────────────────────────

@app.get("/report/{report_id}")
def get_report(report_id: str, request: Request, db: Session = Depends(get_db),
               principal: Principal = Depends(get_principal)):
    r = _load_report_for(principal, report_id, db)
    audit.record(actor=principal.subject, role=principal.role, action="report.view",
                 resource=f"report:{report_id}", ip=_ip(request), detail={"patient_id": r.patient_id})
    if principal.role == ROLE_PATIENT:
        return _report_patient_view(r)
    return _report_full(r, db)


@app.post("/report/{report_id}/translate")
async def translate_report(report_id: str, request: Request,
                           language: str = Query(..., min_length=2, max_length=5),
                           db: Session = Depends(get_db), principal: Principal = Depends(get_principal)):
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
                principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_SERVICE))):
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
                    principal: Principal = Depends(get_principal)):
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
                   principal: Principal = Depends(get_principal)):
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
                  principal: Principal = Depends(require_roles(ROLE_CLINICIAN, ROLE_SERVICE))):
    patients = db.query(Patient).all()
    return {"patients": [p.id for p in patients]}


@app.delete("/patient/{patient_id}")
def delete_patient(patient_id: str, request: Request, db: Session = Depends(get_db),
                   principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found.")
    n_reports = len(patient.reports)
    audit.record(actor=principal.subject, role=principal.role, action="patient.delete",
                 resource=f"patient:{patient_id}", ip=_ip(request), critical=True,
                 detail={"reports_deleted": n_reports})
    db.delete(patient)
    db.commit()
    logger.warning("Patient record deleted: patient_id=%s by=%s", patient_id, principal.subject)
    return {"message": f"Patient {patient_id} and all reports deleted."}


# ── Operations: metrics + audit ────────────────────────────────────────────────

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

    durations = [x.duration_ms for x in runs if x.duration_ms]
    llm = [x.llm_calls for x in runs if x.llm_calls]
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
