"""History-store outbox: the report and its ChromaDB write are made consistent without a shared
transaction. The event is staged with the report, applied with retries, idempotent, cancelled by
rejection and erased with the patient."""
import time
from datetime import timedelta

import pytest

from tests.patching import patch_used
from src.pipeline.history_sync import HISTORY_PENDING
from src.jobs import outbox
from src.db.models import OutboxEvent, Report
from src.db.session import SessionLocal

from tests.test_api_flow import (  # noqa: F401
    APPROVE, CLIN, client, fresh_state, login, upload, upload_raw, wait_for_job,
)


def _events(report_id=None):
    with SessionLocal() as db:
        q = db.query(OutboxEvent)
        if report_id:
            q = q.filter(OutboxEvent.report_id == report_id)
        return [(e.status, e.attempts, e.last_error) for e in q.order_by(OutboxEvent.id).all()]


def _warnings(report_id):
    with SessionLocal() as db:
        return list(db.query(Report).filter(Report.id == report_id).first().ingestion_warnings or [])


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_event_is_written_with_the_report_and_applied_once(client, monkeypatch):
    calls = []
    patch_used(monkeypatch, "chroma_store", lambda **k: calls.append(k))
    h = login(client, CLIN)
    upload(client, h, report_id="o-1")
    assert [c["report_id"] for c in calls] == ["o-1"]
    assert _events("o-1") == [("done", 1, None)]
    with SessionLocal() as db:                                    # applied events keep no payload
        assert db.query(OutboxEvent).filter(OutboxEvent.report_id == "o-1").first().payload is None


def test_payload_holds_redacted_text_only(client, monkeypatch):
    seen = {}
    patch_used(monkeypatch, "chroma_store", lambda **k: seen.update(k))
    h = login(client, CLIN)
    upload(client, h, report_id="o-2")
    assert "Priya" not in seen["report_text"] and "9876543210" not in seen["report_text"]
    assert "Hemoglobin" in seen["report_text"]


def test_failed_write_stays_queued_with_a_visible_warning_then_succeeds_on_retry(client, monkeypatch):
    monkeypatch.setenv("OUTBOX_RETRY_BASE_SECONDS", "3600")        # no automatic retry during the first part
    state = {"fail": True, "calls": 0}

    def store(**k):
        state["calls"] += 1
        if state["fail"]:
            raise OSError("embedding model unavailable")

    patch_used(monkeypatch, "chroma_store", store)
    h = login(client, CLIN)
    r = upload(client, h, report_id="o-3")
    assert r.status_code == 200                                    # the upload itself succeeded
    assert _events("o-3")[0][0] == "pending" and "OSError" in _events("o-3")[0][2]
    warn = _warnings("o-3")
    assert any(w.startswith(HISTORY_PENDING) for w in warn)

    state["fail"] = False                                          # the store recovers
    with SessionLocal() as db:
        db.query(OutboxEvent).update({"next_attempt_at": outbox.utcnow() - timedelta(seconds=1)})
        db.commit()
    assert _wait(lambda: _events("o-3")[0][0] == "done")
    assert _wait(lambda: not any(w.startswith(HISTORY_PENDING) for w in _warnings("o-3")))


def test_gives_up_after_max_attempts_and_says_so(client, monkeypatch):
    monkeypatch.setenv("OUTBOX_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("OUTBOX_RETRY_BASE_SECONDS", "0")

    def boom(**k):
        raise OSError("down")

    patch_used(monkeypatch, "chroma_store", boom)
    h = login(client, CLIN)
    upload(client, h, report_id="o-4")
    assert _wait(lambda: _events("o-4")[0][0] == "dead")
    assert _events("o-4")[0][1] == 2
    assert _wait(lambda: any("after repeated attempts" in w for w in _warnings("o-4")))
    assert not any(w.startswith(HISTORY_PENDING) for w in _warnings("o-4"))


# The three tests below deliberately do not use the `client` fixture: it starts the app's own outbox
# relay, which would claim these hand-made events before the test does.

def test_two_relays_never_apply_the_same_event_twice():
    with SessionLocal() as db:
        outbox.add(db, kind=outbox.HISTORY_INDEX, report_id="x", patient_id="P001", payload={})
        db.commit()
        a = outbox.claim_due(db, lease_seconds=60)
        b = outbox.claim_due(db, lease_seconds=60)
    assert len(a) == 1 and b == []                                 # the lease hides it from the second


def test_claim_cannot_be_won_twice_under_a_race():
    """Simulate the losing side of a race: the row was claimed between SELECT and UPDATE."""
    with SessionLocal() as db:
        outbox.add(db, kind=outbox.HISTORY_INDEX, report_id="y", patient_id="P001", payload={})
        db.commit()
        winner = outbox.claim_due(db, lease_seconds=60)
    assert len(winner) == 1
    with SessionLocal() as db:
        assert outbox.claim_due(db, lease_seconds=60) == []


def test_a_crashed_worker_s_event_becomes_due_again_after_the_lease():
    with SessionLocal() as db:
        outbox.add(db, kind=outbox.HISTORY_INDEX, report_id="z", patient_id="P001", payload={})
        db.commit()
        assert len(outbox.claim_due(db, lease_seconds=60)) == 1       # claimed, then the process "dies"
        assert outbox.claim_due(db, lease_seconds=60) == []
        later = outbox.utcnow() + timedelta(seconds=61)
        again = outbox.claim_due(db, lease_seconds=60, now=later)
    assert len(again) == 1 and again[0]["attempts"] == 2


def test_rejection_cancels_a_pending_write_and_nothing_is_stored(client, monkeypatch):
    monkeypatch.setenv("OUTBOX_RETRY_BASE_SECONDS", "3600")
    calls = []

    def flaky(**k):
        calls.append(k)
        raise OSError("down")

    patch_used(monkeypatch, "chroma_store", flaky)
    patch_used(monkeypatch, "delete_report_history", lambda *a, **k: None)
    h = login(client, CLIN)
    upload(client, h, report_id="o-5")
    assert _events("o-5")[0][0] == "pending"
    r = client.post("/report/o-5/review", headers=h, json={"decision": "reject", "note": "misread values"})
    assert r.status_code == 200
    assert _events("o-5")[0][0] == "cancelled"
    n = len(calls)
    with SessionLocal() as db:
        db.query(OutboxEvent).update({"next_attempt_at": outbox.utcnow() - timedelta(seconds=1)})
        db.commit()
    time.sleep(0.3)
    assert len(calls) == n                                         # the relay did not write it later


def test_write_that_races_with_a_rejection_is_undone(client, monkeypatch):
    undone = []
    h = login(client, CLIN)
    patch_used(monkeypatch, "delete_report_history", lambda pid, rid: undone.append((pid, rid)))

    def store(**k):                                                # the clinician rejects mid-write
        with SessionLocal() as db:
            db.query(Report).filter(Report.id == k["report_id"]).update({"review_status": "rejected"})
            db.commit()

    patch_used(monkeypatch, "chroma_store", store)
    upload(client, h, report_id="o-6")
    assert undone == [("P001", "o-6")]


def test_patient_erasure_removes_outbox_events(client, monkeypatch):
    monkeypatch.setenv("OUTBOX_RETRY_BASE_SECONDS", "3600")
    patch_used(monkeypatch, "chroma_store", lambda **k: (_ for _ in ()).throw(OSError("down")))
    h = login(client, CLIN)
    upload(client, h, report_id="o-7")
    assert _events("o-7")
    assert client.delete("/patient/P001", headers=h).status_code == 200
    assert _events("o-7") == []


def test_ready_and_metrics_report_the_outbox(client, monkeypatch):
    body = client.get("/ready").json()
    assert body["outbox_relay"] == "ok" and set(body["outbox"]) == {"pending", "done", "dead", "cancelled"}
    monkeypatch.setenv("METRICS_TOKEN", "tok-tok-tok-tok")
    text = client.get("/metrics", headers={"Authorization": "Bearer tok-tok-tok-tok"}).text
    assert 'outbox_events{status="pending"}' in text
