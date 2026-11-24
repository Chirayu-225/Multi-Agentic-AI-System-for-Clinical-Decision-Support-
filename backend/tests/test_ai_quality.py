"""Clinician-edit feedback loop: edit measurement, aggregation, and the API."""
import pytest

import main
from src.monitoring import ai_quality as aq
from src.db.models import AgentRun, Report
from src.db.session import SessionLocal

from tests.test_api_flow import (  # noqa: F401
    APPROVE, CLIN, client, fresh_state, login, upload, upload_raw, wait_for_job,
)


# ── measuring an edit ──────────────────────────────────────────────────────────

def test_identical_text_is_no_change_and_a_rewrite_is_a_large_change():
    assert aq.word_diff("Hb is low", "Hb is low") == (0.0, 0)
    ratio, words = aq.word_diff("Hemoglobin is 9.2 g/dL and low", "Completely different sentence here now")
    assert ratio > 0.8 and words >= 8


def test_a_typo_fix_is_small_next_to_a_rewrite():
    base = "Hemoglobin is 9.2 g/dL which is below the normal range for an adult female patient."
    small, _ = aq.word_diff(base, base.replace("below", "under"))
    large, _ = aq.word_diff(base, "Patient needs iron studies and a repeat count within two weeks.")
    assert 0 < small < 0.15 < large


def test_combined_stats_weight_by_length():
    long_a = " ".join(["word"] * 100)
    ratio, words = aq.edit_stats([(long_a, long_a), ("short text", "other thing")])
    assert words == 4 and ratio < 0.1                      # a short rewrite beside a long untouched text


def test_empty_inputs_do_not_crash():
    assert aq.word_diff("", "") == (0.0, 0)
    assert aq.word_diff("", "new text")[1] == 2


# ── aggregation ────────────────────────────────────────────────────────────────

def rec(status="approved", edited=False, ratio=None, vc=False, vu=False, sv="summarizer@A", vv="verifier@X", model="m"):
    return {"status": status, "edited": edited, "edit_ratio": ratio, "verifier_corrected": vc,
            "verifier_unverified": vu, "summarizer_version": sv, "verifier_version": vv, "model": model}


def test_overall_counts_and_rates():
    s = aq.quality_summary([rec(), rec(), rec(edited=True, ratio=0.2), rec(edited=True, ratio=0.4), rec("rejected")])
    assert (s["reviewed"], s["approved"], s["rejected"]) == (5, 4, 1)
    assert s["approved_unedited"] == 2 and s["approved_edited"] == 2
    assert s["unedited_rate"] == 0.4 and s["edit_rate"] == 0.5 and s["rejection_rate"] == 0.2
    assert s["median_edit_ratio"] == pytest.approx(0.3)
    assert s["enough_data"] is False                          # 5 < 20


def test_verifier_misses_are_edits_on_summaries_it_passed():
    s = aq.quality_summary([rec(), rec(edited=True, ratio=0.1), rec(vc=True), rec(vu=True), rec(edited=True, ratio=0.1, vc=True)])
    v = s["verifier"]
    assert v["corrected_rate"] == 0.4 and v["unverified_rate"] == 0.2
    assert v["missed_by_verifier_rate"] == 0.5                # of 2 approved+passed, 1 was edited anyway


def test_prompt_versions_are_compared_separately_and_small_groups_are_flagged():
    rows = [rec(sv="summarizer@A")] * 25 + [rec(edited=True, ratio=0.3, sv="summarizer@B")] * 3
    s = aq.quality_summary(rows)
    a, b = s["by_summarizer_version"]["summarizer@A"], s["by_summarizer_version"]["summarizer@B"]
    assert a["enough_data"] is True and a["unedited_rate"] == 1.0
    assert b["enough_data"] is False and b["edit_rate"] == 1.0
    assert "anecdotes" in s["note"]


def test_empty_input_is_all_none_not_a_crash():
    s = aq.quality_summary([])
    assert s["reviewed"] == 0 and s["unedited_rate"] is None and s["verifier"]["corrected_rate"] is None


# ── through the API ────────────────────────────────────────────────────────────

def test_edit_ratio_is_stored_when_a_clinician_edits(client):
    h = login(client, CLIN)
    upload(client, h, report_id="q-1")
    body = {**APPROVE, "edited_clinical_summary": "Hemoglobin low; start iron and repeat CBC in two weeks."}
    assert client.post("/report/q-1/review", headers=h, json=body).status_code == 200
    with SessionLocal() as db:
        r = db.query(Report).filter(Report.id == "q-1").first()
        assert r.review_edited is True and r.edit_ratio > 0.3 and r.edit_words_changed > 5


def test_an_unedited_approval_has_no_edit_ratio(client):
    h = login(client, CLIN)
    upload(client, h, report_id="q-2")
    client.post("/report/q-2/review", headers=h, json=APPROVE)
    with SessionLocal() as db:
        assert db.query(Report).filter(Report.id == "q-2").first().edit_ratio is None


def test_ai_quality_endpoint_groups_by_the_prompt_version_the_report_was_made_with(client):
    h = login(client, CLIN)
    for rid in ("q-3", "q-4", "q-5"):
        upload(client, h, report_id=rid)
    with SessionLocal() as db:                                  # pretend q-5 was made by an older prompt
        db.query(AgentRun).filter(AgentRun.report_id == "q-5").update(
            {"prompt_versions": {"summarizer": "summarizer@OLD", "verifier": "verifier@OLD"}})
        db.commit()
    client.post("/report/q-3/review", headers=h, json=APPROVE)
    client.post("/report/q-4/review", headers=h, json={**APPROVE, "edited_patient_summary": "A different explanation entirely."})
    client.post("/report/q-5/review", headers=h, json={"decision": "reject", "note": "wrong patient"})
    body = client.get("/admin/ai-quality?days=1", headers=h).json()
    assert (body["reviewed"], body["approved"], body["rejected"], body["approved_edited"]) == (3, 2, 1, 1)
    assert "summarizer@OLD" in body["by_summarizer_version"]
    assert body["by_summarizer_version"]["summarizer@OLD"]["rejection_rate"] == 1.0
    text = str(body)
    assert "wrong patient" not in text and "q-3" not in text and "P001" not in text   # aggregates only


def test_ai_quality_needs_a_clinician(client):
    assert client.get("/admin/ai-quality").status_code == 401
