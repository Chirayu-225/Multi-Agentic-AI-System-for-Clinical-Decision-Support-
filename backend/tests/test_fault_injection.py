"""Fault injection: break one thing at a time in the middle of the pipeline and check the
invariants that matter - nothing lost, nothing duplicated, nothing released early, nothing
claimed as audited that was not. Runs on SQLite and, in CI, on PostgreSQL."""
import threading

import pytest
from sqlalchemy.exc import OperationalError

import main
from tests.patching import patch_used
from src.security import audit
from src.jobs import outbox
from src.db.models import OutboxEvent, Report, UploadJob
from src.db.session import SessionLocal, engine
from src.jobs.queue import PermanentJobError

from tests import test_api_flow as flow
from tests.test_api_flow import (  # noqa: F401
    APPROVE, CLIN, client, fresh_state, login, upload, upload_raw, wait_for_job,
)


@pytest.fixture()
def critical(monkeypatch):
    monkeypatch.setitem(flow.ENTITIES, "lab_values", [{"test": "Potassium", "value": 7.1, "unit": "mmol/L"}])


def rows(report_id):
    with SessionLocal() as db:
        return db.query(Report).filter(Report.id == report_id).all()


# ── the database fails during the final save ───────────────────────────────────

def test_database_error_at_the_final_save_is_retried_and_leaves_exactly_one_report(client, monkeypatch):
    real_add, n = outbox.add, {"i": 0}

    def flaky_add(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            raise OperationalError("INSERT", {}, Exception("connection reset"))
        return real_add(*a, **k)

    monkeypatch.setattr(outbox, "add", flaky_add)
    h = login(client, CLIN)
    upload_raw(client, h, report_id="f-1")
    assert wait_for_job(client, h, "f-1")["status"] == "complete"
    (r,) = rows("f-1")
    assert r.analysis_status == "complete" and r.clinical_summary
    with SessionLocal() as db:
        assert db.query(OutboxEvent).filter(OutboxEvent.report_id == "f-1").count() == 1   # no duplicate event


# ── the analysis dies for good after the preliminary save ─────────────────────

def test_a_permanently_failed_analysis_still_leaves_the_critical_result_visible_and_unapprovable(client, critical, monkeypatch):
    async def dies(*a, **k):
        raise PermanentJobError("model output unusable")

    patch_used(monkeypatch, "generate_summaries", dies)
    h = login(client, CLIN)
    upload_raw(client, h, report_id="f-2")
    assert wait_for_job(client, h, "f-2")["status"] == "failed"
    (r,) = rows("f-2")
    assert r.analysis_status == "preliminary" and r.has_critical is True
    pending = client.get("/reviews/pending", headers=h).json()["reports"]
    assert any(p["report_id"] == "f-2" and p["has_critical"] for p in pending)
    assert client.post("/report/f-2/review", headers=h, json=APPROVE).status_code == 409


# ── the history store is down ──────────────────────────────────────────────────

def test_history_store_outage_never_loses_the_report_or_the_pending_write(client, monkeypatch):
    monkeypatch.setenv("OUTBOX_RETRY_BASE_SECONDS", "3600")
    patch_used(monkeypatch, "chroma_store", lambda **k: (_ for _ in ()).throw(OSError("chroma down")))
    h = login(client, CLIN)
    assert upload(client, h, report_id="f-3").status_code == 200
    (r,) = rows("f-3")
    assert r.analysis_status == "complete"
    with SessionLocal() as db:
        ev = db.query(OutboxEvent).filter(OutboxEvent.report_id == "f-3").one()
        assert ev.status == "pending" and ev.payload                      # still deliverable later


# ── the verifier is unusable ───────────────────────────────────────────────────

def test_an_unusable_verifier_cannot_produce_a_silently_approvable_report(client, monkeypatch):
    async def unverified(clinical_summary, patient_summary, *a, **k):
        return {"clinical_summary": clinical_summary, "patient_summary": patient_summary,
                "clinical_corrected": False, "patient_corrected": False,
                "clinical_unverified": True, "patient_unverified": True,
                "clinical_issues": "x", "patient_issues": "x"}

    patch_used(monkeypatch, "verify_and_correct", unverified)
    h = login(client, CLIN)
    upload(client, h, report_id="f-4")
    r = client.post("/report/f-4/review", headers=h, json={"decision": "approve"})
    assert r.status_code == 409 and any("NOT been audited" in w for w in r.json()["detail"]["warnings"])


# ── the audit store is unwritable at release time ─────────────────────────────

def test_release_decision_fails_closed_when_the_audit_log_cannot_be_written(client, monkeypatch):
    h = login(client, CLIN)
    upload(client, h, report_id="f-5")

    def broken(*a, **k):
        if k.get("critical"):
            raise RuntimeError("audit store down")

    monkeypatch.setattr(audit, "record", broken)
    from fastapi.testclient import TestClient
    r = TestClient(main.app, raise_server_exceptions=False).post("/report/f-5/review", headers=h, json=APPROVE)
    assert r.status_code == 500                              # the approval must not silently go through
    with SessionLocal() as db:
        assert db.query(Report).filter(Report.id == "f-5").one().review_status == "pending_review"


# ── load: many at once ────────────────────────────────────────────────────────

def test_twenty_simultaneous_uploads_all_complete_exactly_once(client):
    h = login(client, CLIN)
    for i in range(20):
        assert upload_raw(client, h, report_id=f"l-{i}", patient_id=f"P{i % 4}").status_code == 202
    for i in range(20):
        assert wait_for_job(client, h, f"l-{i}", timeout=60)["status"] == "complete"
    with SessionLocal() as db:
        assert db.query(Report).filter(Report.id.like("l-%")).count() == 20
        assert db.query(UploadJob).filter(UploadJob.id.like("l-%"), UploadJob.status != "complete").count() == 0
        done = db.query(OutboxEvent).filter(OutboxEvent.report_id.like("l-%")).all()
        assert len(done) == 20 and all(e.status in ("done", "pending") for e in done)


@pytest.mark.skipif(engine.dialect.name != "postgresql",
                    reason="row locking (SELECT ... FOR UPDATE) only exists on PostgreSQL; SQLite serialises writers")
def test_two_clinicians_deciding_at_once_exactly_one_decision_wins(client):
    h = login(client, CLIN)
    upload(client, h, report_id="f-6")
    results = []

    def decide(decision):
        body = APPROVE if decision == "approve" else {"decision": "reject", "note": "disagree with values"}
        results.append(client.post("/report/f-6/review", headers=h, json=body).status_code)

    threads = [threading.Thread(target=decide, args=(d,)) for d in ("approve", "reject")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    with SessionLocal() as db:
        final = db.query(Report).filter(Report.id == "f-6").one().review_status
    # approve->reject is a legal sequence (retraction), so either order is valid; what must never
    # happen is a lost update or an error.
    assert sorted(results) in ([200, 200], [200, 409]) and final in ("approved", "rejected")
