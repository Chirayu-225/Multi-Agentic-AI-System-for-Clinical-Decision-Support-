"""Report upload, upload-job status and the live progress WebSocket."""
import os
import re
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.agents.translator import LANGUAGES, is_supported
from src.api.auth import Principal, ROLE_CLINICIAN, require_roles, verify_ws_auth
from src.api.deps import client_ip, safe_filename, valid_patient_id
from src.core.config import env_num, spool_dir
from src.db.models import Report
from src.db.session import get_db
from src.ingestion.document_loader import ALLOWED_EXTS, extension_matches_content, sniff_file_type
from src.jobs import queue as jobq
from src.monitoring import observability as obs
from src.pipeline import workers
from src.pipeline.progress import ws_connections
from src.security import audit
from src.security.rate_limit import allow as rate_limit_allow

router = APIRouter()


# ── WebSocket — real-time agent streaming ──────────────────────────────────────

@router.websocket("/ws/{report_id}")
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
    ws_connections[report_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        ws_connections.pop(report_id, None)


@router.get("/report/{report_id}/job")
def get_upload_job(report_id: str, db: Session = Depends(get_db),
                   principal: Principal = Depends(require_roles(ROLE_CLINICIAN))):
    """Job status; also the poll fallback for the websocket (late connect, dropped socket,
    or a job being processed by another process). Uploader only."""
    row = jobq.get(db, report_id)
    if not row or row.owner != principal.subject:
        raise HTTPException(404, "No such upload job.")
    return jobq.job_view(row)


# ── Report upload & analysis ───────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


# The pipeline behind this endpoint is expensive (OCR/NER + 2-3 Gemini calls +
# external API calls). Limits are per caller (user / service key) — see
# src/security/rate_limit.py for why this is process-local and what to replace it
# with at multi-instance scale.
UPLOAD_RATE_LIMIT_MAX = int(os.getenv("UPLOAD_RATE_LIMIT_MAX", "10"))


UPLOAD_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("UPLOAD_RATE_LIMIT_WINDOW_SECONDS", "600"))


@router.post("/report/upload")
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
    ip = client_ip(request)
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

    patient_id = valid_patient_id(patient_id)
    language = (language or "en").lower()
    if language != "en" and not is_supported(language):
        raise HTTPException(400, f"Unsupported language '{language}'. Use 'en' or one of: {', '.join(LANGUAGES)}.")

    filename = safe_filename(file.filename)
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
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=spool_dir()) as tmp:
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
    if jobq.active_count(db) >= env_num("JOB_QUEUE_MAX", 200, int):
        os.unlink(tmp_path)
        raise HTTPException(503, "Too many uploads in progress. Retry shortly.",
                            headers={"Retry-After": "30"})
    try:
        row = jobq.enqueue(
            db, job_id=report_id, owner=principal.subject, ip=ip, patient_id=patient_id, filename=filename,
            ext=ext, report_date=report_date, language=language, spool_path=tmp_path,
            max_attempts=env_num("JOB_MAX_ATTEMPTS", 3, int))
    except jobq.DuplicateJob:
        os.unlink(tmp_path)
        raise HTTPException(409, "A report with this id already exists.")
    except BaseException:
        os.unlink(tmp_path)
        raise
    audit.record(actor=principal.subject, role=principal.role, action="report.upload_queued",
                 resource=f"report:{report_id}", ip=ip, detail={"patient_id": patient_id})
    obs.inc("uploads_accepted_total")
    if workers.job_worker is not None:
        workers.job_worker.wake()
    return JSONResponse(
        status_code=202,
        content={**jobq.job_view(row), "status_url": f"/report/{report_id}/job"},
        headers={"Location": f"/report/{report_id}/job"},
    )
