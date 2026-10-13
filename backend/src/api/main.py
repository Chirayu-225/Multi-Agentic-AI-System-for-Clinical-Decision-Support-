"""
src/api/main.py
MedSight v2 — FastAPI backend.
Endpoints: report upload, patient history, WebSocket agent streaming.
"""

import logging
import os
import uuid
import tempfile
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Depends, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.db.session import get_db, create_tables
from src.db.models import Patient, Report, AgentRun
from src.ingestion.document_loader import load_document, chunk_text
from src.agents.orchestrator import run_orchestrator
from src.agents.summarizer import generate_summaries
from src.agents.verifier import verify_and_correct
from src.tools.history_store import store_report as chroma_store
from src.api.auth import require_api_key, verify_ws_api_key, fingerprint
from src.security.rate_limit import allow as rate_limit_allow
from src.security.llm_guardrails import scan_output_for_leakage

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("medsight")

# Every AI-generated summary carries this disclaimer end-to-end (API + UI) —
# this tool assists a clinician's review, it does not replace it.
AI_DISCLAIMER = (
    "AI-generated analysis for clinician review only. Not a diagnosis and not "
    "a substitute for professional medical judgment. Verify all findings "
    "against the source report before acting on them."
)

app = FastAPI(
    title="MedSight API v2",
    description="Multi-Agent Clinical Intelligence Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:4200").split(","),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Active WebSocket connections per report_id
_ws_connections: dict[str, WebSocket] = {}


@app.on_event("startup")
async def startup():
    # Fail fast and loud on missing secrets rather than 500ing on first
    # request — much easier to diagnose in a demo / interview setting.
    missing = [v for v in ("OPENROUTER_API_KEY", "API_KEY") if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"See backend/.env.example."
        )
    create_tables()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "agents": ["orchestrator", "verifier"]}


# ── WebSocket — real-time agent streaming ──────────────────────────────────────

@app.websocket("/ws/{report_id}")
async def websocket_endpoint(websocket: WebSocket, report_id: str):
    """
    WebSocket endpoint for real-time agent step streaming.
    Angular frontend connects here to display live agent activity.

    Auth: browsers' native WebSocket API can't set custom headers, so the
    shared secret travels as an `api_key` query param instead
    (`/ws/{report_id}?api_key=...`). Use wss:// in any non-local deployment
    so the query string is encrypted in transit.
    """
    if not await verify_ws_api_key(websocket):
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
                pass
    import asyncio
    def callback(step: dict):
        asyncio.create_task(_send(step))
    return callback


# ── Report upload & analysis ───────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# The pipeline behind this endpoint is expensive (OCR/NER + 2 LLM calls via
# OpenRouter + external API calls) and was previously unlimited. Limits are per caller
# (API-key fingerprint if present, else client IP) — see src/security/rate_limit.py
# for why this is process-local and what to replace it with at multi-instance scale.
UPLOAD_RATE_LIMIT_MAX = int(os.getenv("UPLOAD_RATE_LIMIT_MAX", "10"))
UPLOAD_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("UPLOAD_RATE_LIMIT_WINDOW_SECONDS", "600"))


@app.post("/report/upload")
async def upload_report(
    request: Request,
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    report_date: str = Form(default=""),
    report_id: str = Form(default=""),
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    caller_fp = fingerprint(api_key)
    limiter_key = caller_fp or (request.client.host if request.client else "unknown")
    ok, retry_after = rate_limit_allow(
        limiter_key, UPLOAD_RATE_LIMIT_MAX, UPLOAD_RATE_LIMIT_WINDOW_SECONDS
    )
    if not ok:
        raise HTTPException(
            429,
            detail=f"Rate limit exceeded ({UPLOAD_RATE_LIMIT_MAX} uploads per "
                   f"{UPLOAD_RATE_LIMIT_WINDOW_SECONDS}s). Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Use the client-supplied report_id (so the WebSocket connection the
    # frontend opened BEFORE upload matches the ID used for step callbacks).
    # Fall back to generating one if the client didn't send it.
    report_id = report_id or str(uuid.uuid4())

    # Save to temp file, enforcing a size cap (previously unbounded — a large
    # upload could tie up OCR/NER/LLM processing indefinitely).
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
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
        # ── Step 1: Ingest ─────────────────────────────────────────────────
        doc = load_document(tmp_path)
        raw_text = doc["raw_text"]
        clean_date = str(report_date).split(" ")[0].split("T")[0] if report_date else ""

        # ── Step 2: Orchestrator agent (tool calling + ReAct) ──────────────
        ws_callback = _make_ws_callback(report_id)
        ws_callback({"type": "pipeline_start", "report_id": report_id})

        agent_data = await run_orchestrator(
            report_text=raw_text,
            patient_id=patient_id,
            report_date=clean_date,
            on_step=ws_callback,
        )

        # Log pipeline shape only — never patient content (names, lab
        # values, diagnoses, raw report text). Previously this printed raw
        # lab values and a report-text snippet straight to stdout, which is
        # exactly the kind of PHI that shouldn't land in application logs.
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
        ingestion_warnings = agent_data.get("ingestion_warnings", [])
        security_warnings = list(agent_data.get("security_warnings", []))

        # ── Step 3: Generate summaries ─────────────────────────────────────
        ws_callback({"type": "generating_summaries"})
        summaries = await generate_summaries(
            entities, rule_flags, trend_flags, pubmed_refs, drug_safety, baseline_flags
        )

        # ── Step 4: Verification agent ─────────────────────────────────────
        ws_callback({"type": "verifier_start"})
        verified = await verify_and_correct(
            clinical_summary=summaries["clinical_summary"],
            patient_summary=summaries["patient_summary"],
            entities=entities,
            flags=rule_flags,
            trend_flags=trend_flags,
        )
        ws_callback({"type": "verifier_complete",
                     "clinical_corrected": verified["clinical_corrected"],
                     "patient_corrected": verified["patient_corrected"]})

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
                "security_warnings report_id=%s caller=%s warnings=%s",
                report_id, caller_fp, security_warnings,
            )

        # ── Step 5: Persist to PostgreSQL ──────────────────────────────────
        # Ensure patient exists
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            patient = Patient(id=patient_id)
            db.add(patient)

        lab_values = entities.get("lab_values", [])
        report = Report(
            id=report_id,
            patient_id=patient_id,
            filename=file.filename,
            report_date=clean_date,
            source_type=doc["source_type"],
            has_hindi=str(doc["has_hindi"]),
            pages=doc["pages"],
            entities=entities,
            rule_flags=rule_flags,
            trend_flags=trend_flags,
            baseline_flags=baseline_flags,
            forecast_flags=forecast_flags,
            similar_history=similar_history,
            risk_summary=risk_summary,
            lab_values=lab_values,
            ingestion_warnings=ingestion_warnings,
            security_warnings=security_warnings,
            clinical_summary=verified["clinical_summary"],
            patient_summary=verified["patient_summary"],
            explained_flags=summaries["explained_flags"],
            followup_actions=summaries["followup_actions"],
            longitudinal_narrative=summaries.get("longitudinal_narrative", ""),
            verification={
                "clinical_corrected": verified["clinical_corrected"],
                "patient_corrected": verified["patient_corrected"],
                "clinical_unverified": verified.get("clinical_unverified", False),
                "patient_unverified": verified.get("patient_unverified", False),
                "clinical_issues": verified.get("clinical_issues", ""),
                "patient_issues": verified.get("patient_issues", ""),
            },
        )
        db.add(report)

        # Log agent run
        agent_run = AgentRun(
            report_id=report_id,
            agent_name="orchestrator+verifier",
            tool_calls=agent_data.get("tool_calls_log", []),
            was_corrected=str(
                verified["clinical_corrected"] or verified["patient_corrected"]
            ),
            caller_fingerprint=caller_fp,
        )
        db.add(agent_run)
        db.commit()

        # ── Step 6: Store in ChromaDB for longitudinal tracking ─────────────
        chroma_store(
            patient_id=patient_id,
            report_text=raw_text,
            lab_values=lab_values,
            report_date=clean_date,
            filename=file.filename,
        )

        ws_callback({"type": "pipeline_complete", "report_id": report_id})

        return {
            "report_id": report_id,
            "patient_id": patient_id,
            "filename": file.filename,
            "source_type": doc["source_type"],
            "has_hindi": doc["has_hindi"],
            "pages": doc["pages"],
            "entities": entities,
            "rule_flags": rule_flags,
            "trend_flags": trend_flags,
            "baseline_flags": baseline_flags,
            "forecast_flags": forecast_flags,
            "similar_history": similar_history,
            "risk_summary": risk_summary,
            "pubmed_references": pubmed_refs,
            "drug_safety": drug_safety,
            "ingestion_warnings": ingestion_warnings,
            "security_warnings": security_warnings,
            "disclaimer": AI_DISCLAIMER,
            "llm_output": {
                "clinical_summary": verified["clinical_summary"],
                "patient_summary": verified["patient_summary"],
                "explained_flags": summaries["explained_flags"],
                "followup_actions": summaries["followup_actions"],
                "longitudinal_narrative": summaries.get("longitudinal_narrative", ""),
                "verification": {
                    "clinical_summary_corrected": verified["clinical_corrected"],
                    "patient_summary_corrected": verified["patient_corrected"],
                    "clinical_unverified": verified.get("clinical_unverified", False),
                    "patient_unverified": verified.get("patient_unverified", False),
                },
            },
            "agent_metadata": {
                "tool_calls": agent_data.get("tool_calls_log", []),
                "pubmed_found": len(pubmed_refs),
                "drug_safety_checked": len(drug_safety),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        incident_id = str(uuid.uuid4())[:8]
        # Full detail (including traceback) is logged server-side only —
        # the client gets a generic message plus a correlation id, not the
        # raw exception string, which previously could reflect internal
        # paths/data back to the caller.
        logger.exception("Upload failed [incident=%s] report_id=%s", incident_id, report_id)
        raise HTTPException(
            500,
            detail=f"Processing failed. Reference: {incident_id}. Check server logs for detail.",
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Patient endpoints ──────────────────────────────────────────────────────────

@app.get("/patient/{patient_id}/history")
def patient_history(patient_id: str, db: Session = Depends(get_db), _auth: str = Depends(require_api_key)):
    reports = (
        db.query(Report)
        .filter(Report.patient_id == patient_id)
        .order_by(Report.report_date)
        .all()
    )
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
                "diagnoses": r.entities.get("clinical", {}).get("diagnoses", []) if r.entities else [],
            }
            for r in reports
        ],
    }


@app.get("/report/{report_id}")
def get_report(report_id: str, db: Session = Depends(get_db), _auth: str = Depends(require_api_key)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(404, "Report not found.")
    return report


@app.get("/patients")
def list_patients(db: Session = Depends(get_db), _auth: str = Depends(require_api_key)):
    patients = db.query(Patient).all()
    return {"patients": [p.id for p in patients]}


@app.delete("/patient/{patient_id}")
def delete_patient(patient_id: str, db: Session = Depends(get_db), _auth: str = Depends(require_api_key)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found.")
    db.delete(patient)
    db.commit()
    logger.warning("Patient record deleted: patient_id=%s", patient_id)
    return {"message": f"Patient {patient_id} and all reports deleted."}