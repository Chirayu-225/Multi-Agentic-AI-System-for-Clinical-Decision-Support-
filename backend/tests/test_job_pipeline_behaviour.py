"""Upload-job behaviour seen through the API: retries, permanent failures, idempotent re-runs,
back-pressure, and the operational endpoints (/ready, /metrics, request ids)."""
import asyncio
import os
import re

import main
from src.db.models import Report
from src.db.session import SessionLocal
from src.jobs import queue as jobq

from tests.test_api_flow import (  # noqa: F401
    CLIN, client, fresh_state, login, upload, upload_raw, wait_for_job,
)


def test_transient_failure_is_retried_and_the_report_is_saved_exactly_once(client, monkeypatch):
    real = main.run_orchestrator
    calls = {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("503 model overloaded")
        return await real(*a, **k)

    monkeypatch.setattr(main, "run_orchestrator", flaky)
    h = login(client, CLIN)
    assert upload_raw(client, h, report_id="p-1").status_code == 202
    job = wait_for_job(client, h, "p-1")
    assert job["status"] == "complete" and job["attempt"] == 2
    with SessionLocal() as db:
        assert db.query(Report).filter(Report.id == "p-1").count() == 1


def test_unreadable_document_fails_at_once_without_burning_retries(client, monkeypatch):
    def broken(path):
        raise ValueError("corrupt pdf")

    monkeypatch.setattr(main, "load_document", broken)
    h = login(client, CLIN)
    upload_raw(client, h, report_id="p-2")
    job = wait_for_job(client, h, "p-2")
    assert job["status"] == "failed" and job["attempt"] == 1
    assert "corrupt" not in (job["error"] or "")


def test_rerunning_a_job_whose_report_is_already_saved_does_nothing(client, monkeypatch):
    h = login(client, CLIN)
    assert upload(client, h, report_id="p-3").status_code == 200

    async def must_not_run(*a, **k):
        raise AssertionError("the pipeline ran again for an already-saved report")

    monkeypatch.setattr(main, "run_orchestrator", must_not_run)
    snap = jobq.JobSnapshot(id="p-3", owner="dr.rao", ip="127.0.0.1", patient_id="P001", filename="r.pdf",
                            ext=".pdf", report_date="", language="en", spool_path="/nonexistent.pdf",
                            attempts=2, max_attempts=3, worker_id="w")
    asyncio.run(main._process_upload(snap, lambda stage, pct: None))      # returns quietly


def test_uploads_are_refused_with_retry_after_when_the_queue_is_full(client, monkeypatch):
    monkeypatch.setenv("JOB_QUEUE_MAX", "0")
    before = set(os.listdir(main._spool_dir()))
    h = login(client, CLIN)
    r = upload_raw(client, h, report_id="p-4")
    assert r.status_code == 503 and r.headers["retry-after"] == "30"
    assert set(os.listdir(main._spool_dir())) == before                   # the rejected file was not kept


def test_spool_file_is_gone_once_the_job_is_done(client):
    h = login(client, CLIN)
    before = set(os.listdir(main._spool_dir()))
    upload(client, h, report_id="p-5")
    assert set(os.listdir(main._spool_dir())) == before


# ── operational endpoints ──────────────────────────────────────────────────────

def test_health_is_liveness_and_ready_reports_database_worker_and_queue(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ready"
    assert body["database"] == "ok" and body["worker"] == "ok"
    assert set(body["queue"]) == {"queued", "running", "complete", "failed"}


def test_request_id_is_returned_and_a_hostile_one_is_replaced(client):
    r = client.get("/health")
    assert re.fullmatch(r"[0-9a-f]{16}", r.headers["x-request-id"])
    assert client.get("/health", headers={"X-Request-ID": "trace-12345678"}).headers["x-request-id"] == "trace-12345678"
    hostile = "x" * 500
    assert client.get("/health", headers={"X-Request-ID": hostile}).headers["x-request-id"] != hostile


def test_metrics_are_off_without_a_token_and_hold_no_ids_when_on(client, monkeypatch):
    assert client.get("/metrics").status_code == 404
    monkeypatch.setenv("METRICS_TOKEN", "s3cret-metrics-token")
    assert client.get("/metrics").status_code == 404
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 404

    h = login(client, CLIN)
    upload(client, h, report_id="m-1")
    r = client.get("/metrics", headers={"Authorization": "Bearer s3cret-metrics-token"})
    assert r.status_code == 200
    text = r.text
    assert "uploads_accepted_total" in text
    assert 'jobs_finished_total{status="complete"}' in text
    assert 'http_requests_total{method="POST",route="/report/upload",status="202"}' in text
    assert 'upload_jobs{status="complete"}' in text
    assert "m-1" not in text and "P001" not in text and "dr.rao" not in text      # no ids, no names


def test_job_log_lines_carry_the_job_id(client, caplog):
    caplog.set_level("INFO", logger="medsight.jobs")
    h = login(client, CLIN)
    upload(client, h, report_id="log-1")
    started = [r for r in caplog.records if r.name == "medsight.jobs" and r.getMessage().startswith("job started")]
    assert started and all(r.job_id == "log-1" for r in started)
