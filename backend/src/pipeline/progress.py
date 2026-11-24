"""Live progress: WebSocket connections and the callbacks that report pipeline steps."""
import asyncio
import json
import logging

from fastapi import WebSocket

logger = logging.getLogger("medsight")


# Active WebSocket connections per report_id
ws_connections: dict[str, WebSocket] = {}


def make_ws_callback(report_id: str):
    """Return a callback that sends agent steps to the connected WebSocket."""
    async def _send(step: dict):
        ws = ws_connections.get(report_id)
        if ws:
            try:
                await ws.send_text(json.dumps(step))
            except Exception:
                logger.debug("websocket send failed for report_id=%s", report_id)

    def callback(step: dict):
        asyncio.create_task(_send(step))
    return callback


# Stage -> percent. Tool calls inside the orchestrator advance 10 -> 50 in small steps.
_STAGE_PERCENT = {
    "queued": 0, "pipeline_start": 10, "orchestrator_complete": 50, "generating_summaries": 60,
    "verifier_start": 75, "verifier_complete": 85, "translating": 90, "saving": 95, "pipeline_complete": 100,
}


def make_job_callback(job_id: str, report_progress):
    """Forward agent steps to the websocket, add `progress` events, and persist progress on the job
    row (via `report_progress`) so the polling endpoint works even if the websocket never connects."""
    ws = make_ws_callback(job_id)
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
