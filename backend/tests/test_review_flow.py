"""
Clinician review: acknowledge-before-release, approve-with-edits, reject-with-reason.

The fixture upload's clinical summary contains "8.4", a number absent from the extracted
data, so every fixture report carries a grounding warning.
"""
import main
from src.db.session import SessionLocal
from src.db.models import Report
from tests.test_api_flow import (CLIN, PAT, client, fresh_state, login, upload)  # noqa: F401


def _upload(client):
    ch = login(client, CLIN)
    return ch, upload(client, ch).json()["report_id"]


def test_approval_blocked_until_warnings_acknowledged(client):
    ch, rid = _upload(client)
    r = client.post(f"/report/{rid}/review", headers=ch, json={"decision": "approve"})
    assert r.status_code == 409
    assert any("Grounding" in w for w in r.json()["detail"]["warnings"])
    assert client.get(f"/report/{rid}", headers=ch).json()["review"]["status"] == "pending_review"

    r = client.post(f"/report/{rid}/review", headers=ch,
                    json={"decision": "approve", "acknowledge_warnings": True})
    assert r.status_code == 200
    rev = client.get(f"/report/{rid}", headers=ch).json()["review"]
    assert rev["status"] == "approved" and rev["warnings_acknowledged"] is True


def test_report_lists_what_needs_acknowledging(client):
    ch, rid = _upload(client)
    need = client.get(f"/report/{rid}", headers=ch).json()["review"]["requires_acknowledgement"]
    assert need and any("8.4" in w for w in need)


def test_clean_report_approves_without_acknowledgement(client):
    ch, rid = _upload(client)
    with SessionLocal() as db:
        r = db.get(Report, rid)
        r.grounding_warnings, r.ingestion_warnings, r.security_warnings = [], [], []
        db.commit()
    r = client.post(f"/report/{rid}/review", headers=ch, json={"decision": "approve"})
    assert r.status_code == 200


def test_approve_with_edits_keeps_ai_original_and_patient_sees_edit(client):
    ch, rid = _upload(client)
    original = client.get(f"/report/{rid}", headers=ch).json()["llm_output"]["patient_summary"]
    edited = "Your haemoglobin is low (9.2 g/dL). Please see your doctor this week."
    r = client.post(f"/report/{rid}/review", headers=ch, json={
        "decision": "approve", "acknowledge_warnings": True,
        "edited_patient_summary": edited,
    })
    assert r.status_code == 200 and r.json()["edited"] == ["patient_summary"]

    full = client.get(f"/report/{rid}", headers=ch).json()
    assert full["llm_output"]["patient_summary"] == edited
    assert full["llm_output"]["ai_original"]["patient_summary"] == original
    assert full["review"]["edited"] is True

    ph = login(client, PAT)
    assert client.get(f"/report/{rid}", headers=ph).json()["patient_summary"] == edited


def test_unchanged_edit_is_not_counted_as_edit(client):
    ch, rid = _upload(client)
    same = client.get(f"/report/{rid}", headers=ch).json()["llm_output"]["clinical_summary"]
    r = client.post(f"/report/{rid}/review", headers=ch, json={
        "decision": "approve", "acknowledge_warnings": True, "edited_clinical_summary": same + "  "})
    assert r.json()["edited"] == []
    assert client.get(f"/report/{rid}", headers=ch).json()["review"]["edited"] is False


def test_reject_still_requires_reason_and_ignores_edits(client):
    ch, rid = _upload(client)
    assert client.post(f"/report/{rid}/review", headers=ch,
                       json={"decision": "reject"}).status_code == 422
    r = client.post(f"/report/{rid}/review", headers=ch, json={
        "decision": "reject", "note": "OCR misread creatinine", "edited_clinical_summary": "x"})
    assert r.status_code == 200 and r.json()["edited"] == []


def test_metrics_report_review_quality(client):
    ch, r1 = _upload(client)
    _, r2 = _upload(client)
    _, r3 = _upload(client)
    client.post(f"/report/{r1}/review", headers=ch, json={"decision": "approve", "acknowledge_warnings": True})
    client.post(f"/report/{r2}/review", headers=ch, json={
        "decision": "approve", "acknowledge_warnings": True, "edited_clinical_summary": "Clinician text."})
    client.post(f"/report/{r3}/review", headers=ch, json={"decision": "reject", "note": "wrong patient"})
    m = client.get("/admin/metrics", headers=ch).json()["review"]
    assert m["reviewed"] == 3
    assert m["approve_without_edit_rate"] == 0.5 and m["edit_rate"] == 0.5
    assert round(m["rejection_rate"], 2) == 0.33
    assert m["approved_with_acknowledged_warnings"] == 2
    assert m["median_minutes_to_review"] is not None

# ── State machine: a decision can't be silently flipped or repeated ─────────────────────

def test_cannot_approve_twice(client):
    ch, rid = _upload(client)
    assert client.post(f"/report/{rid}/review", headers=ch, json={
        "decision": "approve", "acknowledge_warnings": True}).status_code == 200
    again = client.post(f"/report/{rid}/review", headers=ch, json={
        "decision": "approve", "acknowledge_warnings": True,
        "edited_patient_summary": "Quietly changed after release."})
    assert again.status_code == 409 and again.json()["detail"]["current_status"] == "approved"
    # the released text was not altered by the refused request
    assert client.get(f"/report/{rid}", headers=ch).json()["review"]["edited"] is False


def test_rejection_is_final(client):
    ch, rid = _upload(client)
    assert client.post(f"/report/{rid}/review", headers=ch,
                       json={"decision": "reject", "note": "wrong patient"}).status_code == 200
    for body in ({"decision": "approve", "acknowledge_warnings": True},
                 {"decision": "reject", "note": "again"}):
        r = client.post(f"/report/{rid}/review", headers=ch, json=body)
        assert r.status_code == 409 and r.json()["detail"]["current_status"] == "rejected"
    assert client.get(f"/report/{rid}", headers=ch).json()["review"]["status"] == "rejected"


def test_approved_report_can_still_be_retracted(client):
    ch, rid = _upload(client)
    client.post(f"/report/{rid}/review", headers=ch, json={"decision": "approve", "acknowledge_warnings": True})
    r = client.post(f"/report/{rid}/review", headers=ch, json={"decision": "reject", "note": "retracted"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
