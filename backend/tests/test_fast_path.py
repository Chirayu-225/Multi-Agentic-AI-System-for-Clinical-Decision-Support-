"""Critical-value fast path and graceful degradation.

  * The deterministic results are saved as a 'preliminary' report BEFORE the first model call.
  * If the model provider is down the job waits (deferred, no attempt used) while the preliminary
    report stays visible to clinicians, critical results first, and cannot be approved.
  * When the provider recovers the same row becomes 'complete'.
  * A broken preliminary hook can never break the analysis.
"""
import asyncio
from datetime import timedelta

import pytest

import main
from src.db.models import Report
from src.db.session import SessionLocal
from src.jobs import queue as jobq
from src.jobs.queue import DeferJob
from src.db.models import UploadJob

from tests import test_api_flow as flow
from tests.test_api_flow import (  # noqa: F401
    APPROVE, CLIN, client, fresh_state, login, upload, upload_raw, wait_for_job,
)

CRITICAL_LABS = [
    {"test": "Potassium", "value": 7.1, "unit": "mmol/L"},
    {"test": "Hemoglobin", "value": 9.2, "unit": "g/dL"},
]


@pytest.fixture()
def critical(monkeypatch):
    monkeypatch.setitem(flow.ENTITIES, "lab_values", CRITICAL_LABS)


def _report(report_id):
    with SessionLocal() as db:
        r = db.query(Report).filter(Report.id == report_id).first()
        return None if r is None else {
            "status": r.analysis_status, "critical": r.has_critical, "created_at": r.created_at,
            "summary": r.clinical_summary, "review": r.review_status, "warnings": list(r.ingestion_warnings or []),
        }


def _wait_for_stage(report_id, stage, timeout=10.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with SessionLocal() as db:
            j = db.query(UploadJob).filter(UploadJob.id == report_id).first()
            if j and j.stage == stage and j.status == "queued":
                return
        time.sleep(0.02)
    raise AssertionError(f"job {report_id} never reached stage {stage}")


def _make_due(report_id):
    """Skip the deferral delay instead of sleeping through it."""
    with SessionLocal() as db:
        db.query(UploadJob).filter(UploadJob.id == report_id).update(
            {"next_attempt_at": jobq.utcnow() - timedelta(seconds=1)})
        db.commit()


# ── ordering: preliminary is saved before any model step ──────────────────────

def test_preliminary_report_exists_before_the_first_model_step(client, critical, monkeypatch):
    seen = {}
    real = main.generate_summaries

    async def spy(*a, **k):
        seen["row"] = _report("fp-1")           # the first model step is about to run
        return await real(*a, **k)

    monkeypatch.setattr(main, "generate_summaries", spy)
    h = login(client, CLIN)
    assert upload(client, h, report_id="fp-1").status_code == 200
    assert seen["row"]["status"] == "preliminary" and seen["row"]["critical"] is True
    assert seen["row"]["summary"] is None
    final = _report("fp-1")
    assert final["status"] == "complete" and final["critical"] is True
    assert final["created_at"] == seen["row"]["created_at"]       # same row, original timestamp


def test_non_critical_report_is_flagged_as_such(client):
    h = login(client, CLIN)
    upload(client, h, report_id="fp-2")
    assert _report("fp-2")["critical"] is False


# ── provider down: wait, keep the preliminary visible, do not burn attempts ───

def test_provider_outage_defers_the_job_and_keeps_the_critical_result_visible(client, critical, monkeypatch):
    monkeypatch.setattr(main, "llm_unavailable_for", lambda: 120.0)
    h = login(client, CLIN)
    assert upload_raw(client, h, report_id="fp-3").status_code == 202
    _wait_for_stage("fp-3", "waiting_for_ai")

    with SessionLocal() as db:
        job = db.query(UploadJob).filter(UploadJob.id == "fp-3").first()
        assert job.attempts == 0 and job.status == "queued"       # the deferral cost no attempt
        assert job.next_attempt_at > jobq.utcnow() + timedelta(seconds=4)
    row = _report("fp-3")
    assert row["status"] == "preliminary" and row["critical"] is True and row["review"] == "pending_review"
    assert any("PRELIMINARY" in w for w in row["warnings"])

    # visible to clinicians while the model is still unavailable
    pending = client.get("/reviews/pending", headers=h).json()["reports"]
    mine = [p for p in pending if p["report_id"] == "fp-3"][0]
    assert mine["analysis_status"] == "preliminary" and mine["has_critical"] is True
    assert client.get("/report/fp-3", headers=h).json()["analysis_status"] == "preliminary"

    # and it cannot be released
    r = client.post("/report/fp-3/review", headers=h, json=APPROVE)
    assert r.status_code == 409 and "preliminary" in r.json()["detail"]["message"]
    assert _report("fp-3")["review"] == "pending_review"


def test_job_completes_when_the_provider_recovers_and_the_report_becomes_approvable(client, critical, monkeypatch):
    state = {"down": True}
    monkeypatch.setattr(main, "llm_unavailable_for", lambda: 60.0 if state["down"] else 0.0)
    h = login(client, CLIN)
    upload_raw(client, h, report_id="fp-5")
    _wait_for_stage("fp-5", "waiting_for_ai")
    created = _report("fp-5")["created_at"]

    state["down"] = False
    _make_due("fp-5")
    job = wait_for_job(client, h, "fp-5")
    assert job["status"] == "complete" and job["attempt"] == 1
    row = _report("fp-5")
    assert row["status"] == "complete" and row["summary"] and row["created_at"] == created
    assert row["critical"] is True
    assert client.post("/report/fp-5/review", headers=h, json=APPROVE).status_code == 200


def test_waiting_has_a_limit_after_which_the_degraded_report_is_finished(client, monkeypatch):
    monkeypatch.setattr(main, "llm_unavailable_for", lambda: 120.0)
    monkeypatch.setenv("JOB_DEFER_MAX_SECONDS", "0")
    h = login(client, CLIN)
    assert upload(client, h, report_id="fp-6").status_code == 200
    assert _report("fp-6")["status"] == "complete"


def test_critical_reports_are_listed_first_in_the_review_queue(client, monkeypatch):
    h = login(client, CLIN)
    upload(client, h, report_id="old-critical")                   # default labs: not critical
    with SessionLocal() as db:
        db.query(Report).filter(Report.id == "old-critical").update({"has_critical": True})
        db.commit()
    upload(client, h, report_id="newer-normal")
    ids = [p["report_id"] for p in client.get("/reviews/pending", headers=h).json()["reports"]]
    assert ids.index("old-critical") < ids.index("newer-normal")


def test_a_second_attempt_does_not_duplicate_the_preliminary_row(critical):
    data = {"entities": {"lab_values": CRITICAL_LABS}, "rule_flags": [{"severity": "Severity.CRITICAL"}]}
    assert main._save_preliminary("fp-7", "P001", "r.pdf", "2026-10-01", "digital", 1, "dr.rao", data) is True
    assert main._save_preliminary("fp-7", "P001", "r.pdf", "2026-10-01", "digital", 1, "dr.rao", data) is False
    with SessionLocal() as db:
        assert db.query(Report).filter(Report.id == "fp-7").count() == 1


# ── jobs: DeferJob semantics ───────────────────────────────────────────────────

def test_defer_requeues_without_charging_the_attempt():
    # No `client` fixture on purpose: it starts the app's own job worker, which would claim this
    # directly-enqueued job before the test does.
    with SessionLocal() as db:
        jobq.enqueue(db, job_id="d-1", owner="dr.rao", ip=None, patient_id="P001", filename="r.pdf", ext=".pdf",
                     report_date="", language="en", spool_path="/x", max_attempts=3)
        job = jobq.claim_next(db, "w1", 60)
        assert job.attempts == 1
        assert jobq.defer(db, "d-1", "w1", 30) is True
        row = jobq.get(db, "d-1")
        assert row.status == "queued" and row.attempts == 0 and row.stage == "waiting_for_ai"
        assert jobq.claim_next(db, "w2", 60) is None               # not due yet
        assert jobq.defer(db, "d-1", "w1", 30) is False            # no longer the owner


# ── orchestrator hook ──────────────────────────────────────────────────────────

def _orch_run(monkeypatch, on_preliminary, order_log=None):
    pytest.importorskip("google.genai.types")
    from tests.test_orchestrator_tool_loop import DETERMINISTIC, FakeChat, FakeClient
    import src.agents.orchestrator as orch

    order = order_log if order_log is not None else []

    async def fake_execute(name, args):
        order.append(name)
        return DETERMINISTIC.get(name, {"matches": []})

    class Client(FakeClient):
        def __init__(self):
            super().__init__(FakeChat([[]]))
            outer = self

            class _Chats:
                def create(self, model, config):
                    order.append("MODEL")
                    return outer.chat
            self.aio.chats = _Chats()

    monkeypatch.setattr(orch, "_execute_tool", fake_execute)
    monkeypatch.setattr(orch, "get_client", lambda: Client())
    out = asyncio.run(orch.run_orchestrator("Hemoglobin 10.2 g/dL", "P001", "2026-10-01",
                                            on_preliminary=on_preliminary))
    return out, order


def test_hook_runs_after_the_deterministic_steps_and_before_the_first_model_call(monkeypatch):
    got = {}
    order_log = []

    def hook(snap):
        got["snap"] = snap
        got["model_calls_so_far"] = order_log.count("MODEL")

    pytest.importorskip("google.genai.types")
    out, order = _orch_run(monkeypatch, hook, order_log=order_log)
    assert got["model_calls_so_far"] == 0                         # no model call had happened yet
    assert order.index("detect_trends") < order.index("MODEL")
    assert {"entities", "rule_flags", "risk_summary", "drug_interactions"} <= set(got["snap"])
    assert "_redaction_map" not in got["snap"]                    # the map never leaves the orchestrator


def test_a_failing_hook_does_not_stop_the_analysis(monkeypatch):
    def boom(snap):
        raise RuntimeError("database down")

    out, order = _orch_run(monkeypatch, boom)
    assert "risk_summary" in out and "MODEL" in order
