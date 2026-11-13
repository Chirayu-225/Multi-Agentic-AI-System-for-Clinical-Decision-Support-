"""POST /report/upload is a background job: 202 + job id, progress over the websocket and a
polling endpoint, failures reported on the job (never a hung request), temp file cleaned up."""
import os

import main
from src.db.models import Report
from src.db.session import SessionLocal

from tests.test_api_flow import (  # noqa: F401  (fresh_state is autouse; client is a fixture)
    CLIN, PAT, client, fresh_state, login, upload, upload_raw, wait_for_job,
)


def test_upload_returns_202_with_job_id_before_pipeline_runs(client, monkeypatch):
    seen = {}
    real = main.run_orchestrator

    async def spy(*a, **k):
        seen["ran"] = True
        return await real(*a, **k)

    monkeypatch.setattr(main, "run_orchestrator", spy)
    h = login(client, CLIN)
    r = upload_raw(client, h, report_id="job-1")
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"] == body["report_id"] == "job-1"
    assert body["status_url"] == "/report/job-1/job"
    assert r.headers["location"] == "/report/job-1/job"
    assert "clinical_summary" not in body                       # no report payload in the 202
    assert wait_for_job(client, h, "job-1")["status"] == "complete"
    assert seen["ran"]


def test_completed_job_saved_report_pending_review_and_is_fetchable(client):
    h = login(client, CLIN)
    r = upload(client, h, report_id="job-2")
    assert r.status_code == 200
    assert r.json()["review"]["status"] == "pending_review"
    job = client.get("/report/job-2/job", headers=h).json()
    assert job["status"] == "complete" and job["percent"] == 100 and job["error"] is None


def test_pipeline_failure_marks_job_failed_without_leaking_detail(client, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("secret internal path C:\\x\\y")

    monkeypatch.setattr(main, "run_orchestrator", boom)
    h = login(client, CLIN)
    r = upload_raw(client, h, report_id="job-3")
    assert r.status_code == 202
    job = wait_for_job(client, h, "job-3")
    assert job["status"] == "failed"
    assert "Reference:" in job["error"] and "secret" not in job["error"]
    with SessionLocal() as db:                                  # nothing half-saved
        assert db.query(Report).filter(Report.id == "job-3").first() is None


def test_temp_file_removed_after_success_and_after_failure(client, monkeypatch):
    paths = []
    real_load = main.load_document

    def spy_load(path):
        paths.append(path)
        return real_load(path)

    monkeypatch.setattr(main, "load_document", spy_load)
    h = login(client, CLIN)
    upload(client, h, report_id="job-4")

    async def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(main, "run_orchestrator", boom)
    upload(client, h, report_id="job-5")
    # The failing job is attempted several times (all from the same spooled file), so compare
    # distinct files: one per upload, and none may be left behind once the jobs are finished.
    assert len(set(paths)) == 2 and not any(os.path.exists(p) for p in set(paths))


def test_progress_is_streamed_over_websocket_in_order(client):
    h = login(client, CLIN)
    token = h["Authorization"].split()[1]
    percents = []
    with client.websocket_connect(f"/ws/job-6?token={token}") as ws:
        assert upload_raw(client, h, report_id="job-6").status_code == 202
        while True:
            msg = ws.receive_json()
            if msg["type"] == "progress":
                percents.append(msg["percent"])
            if msg["type"] == "pipeline_complete":
                break
    assert percents == sorted(percents)                         # never goes backwards
    assert 60 in percents and 75 in percents and 95 in percents


def test_failure_is_announced_over_websocket(client, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(main, "run_orchestrator", boom)
    h = login(client, CLIN)
    token = h["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/job-7?token={token}") as ws:
        upload_raw(client, h, report_id="job-7")
        while (msg := ws.receive_json())["type"] != "pipeline_error":
            pass
    assert "Reference:" in msg["detail"]


def test_synchronous_rejections_still_happen_before_a_job_exists(client):
    h = login(client, CLIN)
    r = client.post("/report/upload", headers=h, data={"patient_id": "P001", "report_id": "job-8"},
                    files={"file": ("r.pdf", b"<html>x</html>", "application/pdf")})
    assert r.status_code == 400
    assert client.get("/report/job-8/job", headers=h).status_code == 404
    assert upload_raw(client, h, language="xx").status_code == 400


def test_duplicate_report_id_conflicts_while_job_is_in_flight(client):
    h = login(client, CLIN)
    assert upload_raw(client, h, report_id="job-9").status_code == 202
    assert upload_raw(client, h, report_id="job-9").status_code == 409


def test_job_status_is_private_to_the_uploader_and_clinicians(client):
    h = login(client, CLIN)
    upload(client, h, report_id="job-10")
    assert client.get("/report/job-10/job").status_code == 401
    assert client.get("/report/job-10/job", headers=login(client, PAT)).status_code == 403
    assert client.get("/report/nope/job", headers=h).status_code == 404


def test_finished_jobs_are_pruned_after_retention(client):
    from datetime import timedelta
    from src.jobs import queue as jobq
    h = login(client, CLIN)
    upload(client, h, report_id="job-11")
    with SessionLocal() as db:
        assert jobq.prune(db, timedelta(hours=1)) == 0             # too recent
        row = jobq.get(db, "job-11")
        row.finished_at = jobq.utcnow() - timedelta(hours=2)
        db.commit()
        assert jobq.prune(db, timedelta(hours=1)) == 1
    assert client.get("/report/job-11/job", headers=h).status_code == 404
