"""
src/api/main.py
MedSight v2 — FastAPI backend.
Endpoints: report upload, patient history, WebSocket agent streaming.
"""

import os
import uuid
import tempfile
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.db.session import get_db, create_tables
from src.db.models import Patient, Report, AgentRun
from src.ingestion.document_loader import load_document, chunk_text
from src.agents.orchestrator import run_orchestrator
from src.agents.summarizer import generate_summaries
from src.agents.verifier import verify_and_correct
from src.tools.history_store import store_report as chroma_store

app = FastAPI(
    title="MedSight API v2",
    description="Multi-Agent Clinical Intelligence Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:4200").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections per report_id
_ws_connections: dict[str, WebSocket] = {}


@app.on_event("startup")
async def startup():
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
    """
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

@app.post("/report/upload")
async def upload_report(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    report_date: str = Form(default=""),
    report_id: str = Form(default=""),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Use the client-supplied report_id (so the WebSocket connection the
    # frontend opened BEFORE upload matches the ID used for step callbacks).
    # Fall back to generating one if the client didn't send it.
    report_id = report_id or str(uuid.uuid4())

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

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

        # Debug — print what the orchestrator returned
        import json as _json
        print(f"\n[DEBUG] Orchestrator returned:")
        print(f"  entities keys: {list(agent_data.get('entities', {}).keys())}")
        print(f"  lab_values: {agent_data.get('entities', {}).get('lab_values', [])}")
        print(f"  rule_flags count: {len(agent_data.get('rule_flags', []))}")
        print(f"  trend_flags count: {len(agent_data.get('trend_flags', []))}")
        print(f"  tool_calls: {[t['tool'] for t in agent_data.get('tool_calls_log', [])]}")
        print(f"  raw_text snippet: {raw_text[:200]}\n")

        entities    = agent_data.get("entities", {})
        rule_flags  = agent_data.get("rule_flags", [])
        trend_flags = agent_data.get("trend_flags", [])
        risk_summary = agent_data.get("risk_summary", {})
        pubmed_refs = agent_data.get("pubmed_references", [])
        drug_safety = agent_data.get("drug_safety", [])

        # ── Step 3: Generate summaries ─────────────────────────────────────
        ws_callback({"type": "generating_summaries"})
        summaries = await generate_summaries(
            entities, rule_flags, trend_flags, pubmed_refs, drug_safety
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
            risk_summary=risk_summary,
            lab_values=lab_values,
            clinical_summary=verified["clinical_summary"],
            patient_summary=verified["patient_summary"],
            explained_flags=summaries["explained_flags"],
            followup_actions=summaries["followup_actions"],
            verification={
                "clinical_corrected": verified["clinical_corrected"],
                "patient_corrected": verified["patient_corrected"],
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
            "risk_summary": risk_summary,
            "pubmed_references": pubmed_refs,
            "drug_safety": drug_safety,
            "llm_output": {
                "clinical_summary": verified["clinical_summary"],
                "patient_summary": verified["patient_summary"],
                "explained_flags": summaries["explained_flags"],
                "followup_actions": summaries["followup_actions"],
                "verification": {
                    "clinical_summary_corrected": verified["clinical_corrected"],
                    "patient_summary_corrected": verified["patient_corrected"],
                },
            },
            "agent_metadata": {
                "tool_calls": agent_data.get("tool_calls_log", []),
                "pubmed_found": len(pubmed_refs),
                "drug_safety_checked": len(drug_safety),
            },
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[ERROR] Upload failed for patient {patient_id}:\n{tb}")
        raise HTTPException(500, detail=f"{type(e).__name__}: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ── Patient endpoints ──────────────────────────────────────────────────────────

@app.get("/patient/{patient_id}/history")
def patient_history(patient_id: str, db: Session = Depends(get_db)):
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
def get_report(report_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(404, "Report not found.")
    return report


@app.get("/patients")
def list_patients(db: Session = Depends(get_db)):
    patients = db.query(Patient).all()
    return {"patients": [p.id for p in patients]}


@app.delete("/patient/{patient_id}")
def delete_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found.")
    db.delete(patient)
    db.commit()
    return {"message": f"Patient {patient_id} and all reports deleted."}