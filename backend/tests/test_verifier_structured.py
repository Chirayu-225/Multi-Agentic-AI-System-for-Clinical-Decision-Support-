"""Verifier output is JSON validated against a schema, repaired at most once, and fails CLOSED.
The old text format silently reported 'corrected' when the verdict line was missing; these tests
pin that this can no longer happen."""
import asyncio
import json

import pytest

import src.agents.verifier as verifier
from src.agents.verifier import VerifierOutputInvalid, parse_audit, verify_and_correct

ENTS = {"patient": {}, "clinical": {"diagnoses": []}, "medications": []}
FLAGS = [{"test": "Hemoglobin", "value": 9.2, "unit": "g/dL", "severity": "HIGH"}]
CLIN, PAT = "Hemoglobin is 9.2 g/dL.", "Your hemoglobin is a bit low."


def audit(responses, monkeypatch):
    """Run verify_and_correct against scripted model replies; returns (result, calls)."""
    calls = []
    script = list(responses)

    async def fake(prompt, **kw):
        calls.append({"prompt": prompt, **kw})
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(verifier, "generate_text", fake)
    out = asyncio.run(verify_and_correct(CLIN, PAT, ENTS, FLAGS, []))
    return out, calls


def good(c_verdict="PASS", p_verdict="PASS", c_fix=None, p_fix=None, c_issues=(), p_issues=()):
    return json.dumps({"clinical": {"verdict": c_verdict, "issues": list(c_issues), "corrected": c_fix},
                       "patient": {"verdict": p_verdict, "issues": list(p_issues), "corrected": p_fix}})


# ── happy paths ────────────────────────────────────────────────────────────────

def test_clean_audit_passes_both_and_keeps_the_original_text(monkeypatch):
    out, calls = audit([good()], monkeypatch)
    assert len(calls) == 1
    assert out["clinical_summary"] == CLIN and out["patient_summary"] == PAT
    assert not out["clinical_corrected"] and not out["patient_corrected"]
    assert not out["clinical_unverified"] and not out["patient_unverified"]
    assert out["clinical_issues"] == "none"


def test_failure_uses_the_models_correction_and_lists_issues(monkeypatch):
    out, _ = audit([good("FAIL", "PASS", c_fix="Hemoglobin is 9.2 g/dL (low).", c_issues=["mentions HbA1c"])], monkeypatch)
    assert out["clinical_corrected"] is True and out["clinical_summary"] == "Hemoglobin is 9.2 g/dL (low)."
    assert out["clinical_issues"] == "mentions HbA1c"
    assert out["patient_corrected"] is False and out["patient_summary"] == PAT


def test_the_call_requests_json_with_the_schema(monkeypatch):
    _, calls = audit([good()], monkeypatch)
    assert calls[0]["json_schema"] is verifier.AuditResult and calls[0]["label"] == "VERIFIER"


def test_code_fenced_json_is_accepted():
    r = parse_audit("```json\n" + good() + "\n```")
    assert r.clinical.verdict == "PASS"


def test_verdict_is_case_insensitive():
    assert parse_audit(good("pass", "Pass")).patient.verdict == "PASS"


# ── repair, once ───────────────────────────────────────────────────────────────

def test_malformed_output_is_repaired_with_exactly_one_extra_call(monkeypatch):
    out, calls = audit(["Sure! Here is my audit: looks fine", good()], monkeypatch)
    assert [c["label"] for c in calls] == ["VERIFIER", "VERIFIER/repair"]
    assert "not valid" in calls[1]["prompt"] and "looks fine" in calls[1]["prompt"]
    assert not out["clinical_unverified"] and out["clinical_summary"] == CLIN


def test_fail_without_a_correction_is_invalid_and_gets_repaired(monkeypatch):
    """The old parser reported this as 'corrected' with the text unchanged."""
    out, calls = audit([good("FAIL", "PASS", c_fix=None), good("FAIL", "PASS", c_fix="Fixed text.")], monkeypatch)
    assert len(calls) == 2
    assert out["clinical_summary"] == "Fixed text." and out["clinical_corrected"] is True


def test_still_invalid_after_the_repair_fails_closed_and_never_calls_a_third_time(monkeypatch):
    out, calls = audit(["garbage", "more garbage"], monkeypatch)
    assert len(calls) == 2
    assert out["clinical_unverified"] and out["patient_unverified"]
    assert out["clinical_corrected"] is False and out["patient_corrected"] is False   # no silent "corrected"
    assert out["clinical_summary"] == CLIN and out["patient_summary"] == PAT          # text untouched
    assert "NOT been audited" in out["clinical_issues"]


@pytest.mark.parametrize("bad", [
    "", "null", "[]", "{}", '{"clinical": {"verdict": "PASS"}}',                      # patient missing
    '{"clinical": {"verdict": "MAYBE", "issues": []}, "patient": {"verdict": "PASS"}}',
    '{"clinical": {"verdict": "FAIL", "corrected": "   "}, "patient": {"verdict": "PASS"}}',
    '{"clinical": {"verdict": "PASS", "issues": "not a list"}, "patient": {"verdict": "PASS"}}',
])
def test_every_malformed_shape_is_rejected(bad):
    with pytest.raises(VerifierOutputInvalid):
        parse_audit(bad)


# ── fail closed on provider errors ─────────────────────────────────────────────

def test_provider_error_fails_closed(monkeypatch):
    out, calls = audit([RuntimeError("boom")], monkeypatch)
    assert len(calls) == 1
    assert out["clinical_unverified"] and out["patient_unverified"] and not out["clinical_corrected"]


def test_rate_limit_fails_closed_with_a_specific_reason(monkeypatch):
    out, _ = audit([RuntimeError("429 RESOURCE_EXHAUSTED")], monkeypatch)
    assert out["clinical_unverified"] and "rate limit" in out["clinical_issues"]


def test_provider_error_during_the_repair_also_fails_closed(monkeypatch):
    out, calls = audit(["nope", RuntimeError("503 UNAVAILABLE")], monkeypatch)
    assert len(calls) == 2 and out["clinical_unverified"]


def test_circuit_open_fails_closed(monkeypatch):
    from src.resilience import CircuitOpen
    out, _ = audit([CircuitOpen(30)], monkeypatch)
    assert out["clinical_unverified"] and out["patient_unverified"]


# ── the prompt contract ────────────────────────────────────────────────────────

def test_system_prompt_asks_for_json_and_the_version_changed():
    assert "ONLY a JSON object" in verifier.VERIFIER_SYSTEM_PROMPT
    assert "VERDICT_CLINICAL" not in verifier.VERIFIER_SYSTEM_PROMPT
