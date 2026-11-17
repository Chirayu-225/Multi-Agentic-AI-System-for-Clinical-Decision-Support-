"""A critical result must never be shown as "normal", "no significant findings" or "verified".

Found by uploading a synthetic report with hemoglobin 6.2 g/dL and fasting glucose 540 mg/dL: the
flag cards were right, but the banner said "Risk Level NORMAL, 0 Critical", the clinical summary
said "No significant findings to summarize", the action was "Routine monitoring recommended", and
both summaries carried a "verified" badge. Three separate defects, one test group each.
"""
import asyncio
import json

import pytest

import src.agents.orchestrator as orch
import src.agents.summarizer as summarizer
import src.agents.verifier as verifier
from src.flagging.risk_flagging import Severity, compute_risk_summary, normalize_severity

CRITICAL_LABS = [{"test": "Hemoglobin", "value": 6.2, "unit": "g/dL"},
                 {"test": "Glucose (fasting)", "value": 540, "unit": "mg/dL"},
                 {"test": "Total WBC", "value": 7200, "unit": "cells/mcL"}]
ENTITIES = {"patient": {"gender": "Male", "age": "54"}, "clinical": {}, "medications": [],
            "lab_values": CRITICAL_LABS}


def real_flags():
    """Flags exactly as the pipeline holds them: through the tool layer, severities as strings."""
    out = asyncio.run(orch._execute_tool("check_icmr_flags",
                                         {"lab_values": CRITICAL_LABS, "gender": "Male", "age": 54}))
    return out["flags"]


# ── 1. the overall risk level ──────────────────────────────────────────────────────────────────

def test_the_tool_layer_really_does_hand_back_prefixed_strings():
    """The premise of the bug. If this ever changes, the tests below still have to pass."""
    assert {f["severity"] for f in real_flags()} == {"Severity.CRITICAL", "Severity.NORMAL"}


def test_critical_lab_flags_from_the_real_tool_path_make_the_report_critical():
    summary = compute_risk_summary(real_flags(), [], [])
    assert summary["overall_severity"] == Severity.CRITICAL
    assert summary["critical_count"] == 2 and summary["requires_immediate_attention"] is True
    assert sorted(summary["critical_tests"]) == ["Glucose (fasting)", "Hemoglobin"]


@pytest.mark.parametrize("form", [Severity.HIGH, "HIGH", "Severity.HIGH", "severity.high", " high "])
def test_every_form_a_severity_takes_is_read_the_same(form):
    assert normalize_severity(form) == Severity.HIGH
    summary = compute_risk_summary([{"severity": form, "test": "x"}], [], [])
    assert summary["overall_severity"] == Severity.HIGH and summary["high_count"] == 1


@pytest.mark.parametrize("layer", ["trend", "baseline"])
def test_string_severities_count_in_the_trend_and_baseline_layers_too(layer):
    flag = [{"severity": "Severity.HIGH", "test": "x"}]
    summary = compute_risk_summary([], flag if layer == "trend" else [], flag if layer == "baseline" else [])
    assert summary["overall_severity"] == Severity.HIGH


@pytest.mark.parametrize("junk", [None, "", "URGENT", 3])
def test_a_severity_that_cannot_be_read_is_never_treated_as_normal(junk):
    assert normalize_severity(junk) is None
    summary = compute_risk_summary([{"severity": junk, "test": "x"}], [], [])
    assert summary["overall_severity"] == Severity.MODERATE and summary["unrecognized_severity_count"] == 1


def test_all_normal_flags_are_still_normal():
    summary = compute_risk_summary([{"severity": "Severity.NORMAL", "test": "x"}], [], [])
    assert summary["overall_severity"] == Severity.NORMAL and summary["color"] == "green"


def test_the_orchestrator_reports_critical_for_a_critical_report(monkeypatch):
    """End of the real path: deterministic steps with the real lab check, model step unavailable."""
    real = orch._execute_tool

    async def execute(name, args):
        if name == "check_icmr_flags":
            return await real(name, args)
        return {"extract_entities": ENTITIES,
                "detect_trends": {"trend_flags": [], "baseline_flags": [], "forecast_flags": []},
                "get_patient_history": {"result": []}}.get(name, {})

    async def model_down(*a, **k):
        raise RuntimeError("provider down")

    snapshots = []
    monkeypatch.setattr(orch, "_execute_tool", execute)
    monkeypatch.setattr(orch, "with_fallback", model_down)
    out = asyncio.run(orch.run_orchestrator("Hemoglobin 6.2 g/dL  Glucose (fasting) 540 mg/dL", "P1", "2026-11-14",
                                            on_preliminary=snapshots.append))
    for risk in (out["risk_summary"], snapshots[0]["risk_summary"]):     # final and preliminary
        assert risk["overall_severity"] == "Severity.CRITICAL" and risk["critical_count"] == 2


# ── 2. an answer that cannot be read is not an all-clear ───────────────────────────────────────

GOOD = {"CLINICAL_SUMMARY": "Hb 6.2 g/dL, critically low.", "PATIENT_SUMMARY": "Your blood count is very low."}


@pytest.mark.parametrize("answer", [
    "SECTION: CLINICAL_SUMMARY\nHb 6.2 g/dL, critically low.\n\nSECTION: PATIENT_SUMMARY\nYour blood count is very low.",
    "SECTION: CLINICAL_SUMMARY  \nHb 6.2 g/dL, critically low.\nSECTION: PATIENT_SUMMARY\t\nYour blood count is very low.",
    "## SECTION: CLINICAL_SUMMARY\nHb 6.2 g/dL, critically low.\n\n## SECTION: PATIENT_SUMMARY\nYour blood count is very low.\n\n---",
    "**SECTION: CLINICAL_SUMMARY**\nHb 6.2 g/dL, critically low.\n\n**SECTION: PATIENT_SUMMARY**\nYour blood count is very low.",
    "### Clinical Summary\nHb 6.2 g/dL, critically low.\n\n### Patient Summary\nYour blood count is very low.",
    "**Clinical Summary:** Hb 6.2 g/dL, critically low.\n**Patient Summary:** Your blood count is very low.",
    "1. CLINICAL_SUMMARY\nHb 6.2 g/dL, critically low.\n2. PATIENT_SUMMARY\nYour blood count is very low.",
    '```json\n{"clinical_summary": "Hb 6.2 g/dL, critically low.", "patient_summary": "Your blood count is very low."}\n```',
])
def test_the_headers_models_actually_write_are_all_read(answer):
    got = summarizer.split_sections(answer)
    assert {k: got[k] for k in GOOD} == GOOD


def test_a_sentence_that_starts_like_a_header_stays_in_the_body():
    got = summarizer.split_sections("SECTION: CLINICAL_SUMMARY\nPatient summary of labs shows anaemia.\n"
                                    "SECTION: PATIENT_SUMMARY\nLow.")
    assert got["CLINICAL_SUMMARY"] == "Patient summary of labs shows anaemia." and got["PATIENT_SUMMARY"] == "Low."


def summaries_for(answer, monkeypatch, flags=None):
    async def generate(prompt, **kw):
        return answer
    monkeypatch.setattr(summarizer, "generate_text", generate)
    return asyncio.run(summarizer.generate_summaries(ENTITIES, flags if flags is not None else real_flags(), [], [], [], []))


@pytest.mark.parametrize("answer", ["", "I cannot help with that.", "Here is the summary you asked for.",
                                    "SECTION: CLINICAL_SUMMARY\n\nSECTION: PATIENT_SUMMARY\n"])
def test_an_unreadable_answer_becomes_summary_unavailable_never_an_all_clear(answer, monkeypatch):
    out = summaries_for(answer, monkeypatch)
    assert summarizer.is_placeholder(out["clinical_summary"]) and summarizer.is_placeholder(out["patient_summary"])
    everything = json.dumps(out).lower()
    for reassurance in ("no significant findings", "have been reviewed", "routine monitoring"):
        assert reassurance not in everything
    assert out["followup_actions"] == [summarizer.NO_ACTIONS]
    assert len(out["explained_flags"]) == 2                    # the abnormal flags themselves are kept


def test_the_reassuring_placeholders_are_gone_from_the_code():
    import inspect
    source = inspect.getsource(summarizer).lower()
    assert '"no significant findings' not in source and '"your results have been reviewed' not in source
    assert '"routine monitoring recommended' not in source


def test_a_readable_answer_is_used_as_written(monkeypatch):
    out = summaries_for("SECTION: CLINICAL_SUMMARY\nHb 6.2 g/dL and glucose 540 mg/dL, both critical.\n"
                        "SECTION: PATIENT_SUMMARY\nTwo results need urgent attention.\n"
                        "SECTION: FLAG_EXPLANATIONS\nFLAG_0: low hemoglobin.\n**FLAG_1**: high glucose.\n"
                        "SECTION: FOLLOWUP_ACTIONS\n1. Urgent review.\n- Repeat CBC.\n"
                        "SECTION: LONGITUDINAL_NARRATIVE\nN/A", monkeypatch)
    assert out["clinical_summary"].startswith("Hb 6.2") and not summarizer.is_placeholder(out["patient_summary"])
    assert out["followup_actions"] == ["Urgent review.", "Repeat CBC."]
    assert [f["explanation"] for f in out["explained_flags"]] == ["low hemoglobin.", "high glucose."]


def test_half_an_answer_gives_half_a_summary_and_says_so(monkeypatch):
    out = summaries_for("SECTION: CLINICAL_SUMMARY\nHb 6.2 g/dL, glucose 540 mg/dL.", monkeypatch)
    assert out["clinical_summary"].startswith("Hb 6.2") and summarizer.is_placeholder(out["patient_summary"])


# ── 3. what "verified" is allowed to mean ──────────────────────────────────────────────────────

def audit(clinical, patient, monkeypatch, flags=None, corrected=None, calls=None):
    async def generate(prompt, **kw):
        if calls is not None:
            calls.append(kw.get("label"))
        side = {"verdict": "FAIL", "issues": ["x"], "corrected": corrected} if corrected else \
               {"verdict": "PASS", "issues": [], "corrected": None}
        return json.dumps({"clinical": side, "patient": {"verdict": "PASS", "issues": [], "corrected": None}})
    monkeypatch.setattr(verifier, "generate_text", generate)
    return asyncio.run(verifier.verify_and_correct(clinical, patient, ENTITIES,
                                                   flags if flags is not None else real_flags(), []))


def test_the_original_failure_is_no_longer_verified(monkeypatch):
    """The model audit passes it (it contains no false claim). The code check does not."""
    out = audit("No significant findings to summarize.", "Your results have been reviewed.", monkeypatch)
    assert out["clinical_unverified"] is True
    assert sorted(out["clinical_missing_critical"]) == ["Glucose (fasting)", "Hemoglobin"]
    assert "does not mention 2 critical result(s)" in out["clinical_issues"]


def test_a_placeholder_is_never_verified_and_costs_no_model_call(monkeypatch):
    calls = []
    out = audit(summarizer.UNREADABLE_CLINICAL, summarizer.UNREADABLE_PATIENT, monkeypatch, calls=calls)
    assert out["clinical_unverified"] and out["patient_unverified"] and calls == []
    assert out["clinical_issues"] == verifier.NO_SUMMARY_REASON


def test_a_missing_patient_summary_is_unverified_while_a_good_clinical_one_passes(monkeypatch):
    out = audit("Hemoglobin 6.2 g/dL and fasting glucose 540 mg/dL are critical.",
                summarizer.UNREADABLE_PATIENT, monkeypatch)
    assert out["patient_unverified"] is True and out["clinical_unverified"] is False


@pytest.mark.parametrize("clinical", [
    "Hemoglobin 6.2 g/dL and fasting glucose 540 mg/dL are both critical.",     # by value and name
    "Two critical values: 6.2 and 540.",                                        # by value only
    "Hb is critically low and FBS critically high; act today.",                 # by alias only
    "Haemoglobin is critically low; glucose is critically high.",               # another spelling
])
def test_a_summary_that_mentions_every_critical_result_can_be_verified(clinical, monkeypatch):
    out = audit(clinical, "Two results need urgent attention.", monkeypatch)
    assert out["clinical_unverified"] is False and out["clinical_missing_critical"] == []


def test_one_critical_result_left_out_is_named(monkeypatch):
    out = audit("Hemoglobin is 6.2 g/dL, critically low.", "One result needs attention.", monkeypatch)
    assert out["clinical_unverified"] is True and out["clinical_missing_critical"] == ["Glucose (fasting)"]


def test_a_correction_that_drops_a_critical_result_is_not_verified_either(monkeypatch):
    out = audit("Hemoglobin 6.2 g/dL, glucose 999 mg/dL.", "ok text here", monkeypatch,
                corrected="Hemoglobin is 6.2 g/dL.")
    assert out["clinical_corrected"] is True and out["clinical_unverified"] is True


def test_high_and_moderate_flags_do_not_have_to_be_mentioned(monkeypatch):
    flags = [{"test": "TSH", "value": 6.1, "severity": "Severity.MODERATE"},
             {"test": "ALT", "value": 90, "severity": "Severity.HIGH"}]
    out = audit("Thyroid and liver values are mildly raised.", "Some values are a little raised.",
                monkeypatch, flags=flags)
    assert out["clinical_unverified"] is False


def test_a_value_written_with_a_thousands_separator_still_counts():
    flags = [{"test": "Total WBC", "value": 35307.0, "severity": "Severity.CRITICAL"}]
    assert verifier.missing_critical_results("White cell count is 35,307 cells/mcL.", flags) == []
    assert verifier.missing_critical_results("White cell count is raised.", flags) == ["Total WBC"]


# ── 4. a summary that lists the values but says nothing about them (seen live, 2026-11-17) ─────
# Groq's answer, after the verifier's rewrite, was "A 54-year-old male has a hemoglobin of 6.2 g/dL and a
# glucose level of 540 mg/dL." and, for the patient, the two values again. No false claim, every critical
# value named - and neither reader is told that anything is wrong.

LIVE_CLINICAL = "A 54-year-old male has a hemoglobin of 6.2 g/dL and a glucose level of 540 mg/dL."
LIVE_PATIENT = "Your hemoglobin is 6.2 g/dL and your glucose is 540 mg/dL."
GOOD_PATIENT = "Two of your results need attention today. Please contact your doctor today."


def test_the_live_answer_is_no_longer_verified(monkeypatch):
    out = audit(LIVE_CLINICAL, LIVE_PATIENT, monkeypatch)
    assert out["clinical_unverified"] and out["patient_unverified"]
    assert out["clinical_issues"] == verifier.NO_URGENCY
    assert out["patient_issues"] == verifier.NO_DOCTOR_ADVICE
    assert out["content_issues"] == [verifier.NO_URGENCY, verifier.NO_DOCTOR_ADVICE]
    assert out["clinical_audit_unverified"] is False          # the audit itself ran and passed


def test_saying_the_values_are_critical_and_pointing_to_the_doctor_is_verified(monkeypatch):
    out = audit("Hemoglobin 6.2 g/dL is critically low and glucose 540 mg/dL critically high; "
                "both need same-day review.", GOOD_PATIENT, monkeypatch)
    assert not out["clinical_unverified"] and not out["patient_unverified"] and out["content_issues"] == []


def test_a_correction_that_drops_the_urgency_is_not_verified(monkeypatch):
    out = audit("Hemoglobin 6.2 g/dL is critically low, glucose 540 mg/dL critically high; urgent.",
                GOOD_PATIENT, monkeypatch, corrected=LIVE_CLINICAL)
    assert out["clinical_corrected"] and out["clinical_unverified"]
    assert out["clinical_issues"] == verifier.NO_URGENCY


def test_without_a_critical_result_neither_urgency_nor_doctor_advice_is_required(monkeypatch):
    flags = [{"test": "TSH", "value": 6.1, "severity": "Severity.MODERATE"}]
    out = audit("TSH is 6.1 mIU/L, mildly raised.", "Your thyroid value is a little high.", monkeypatch, flags=flags)
    assert not out["clinical_unverified"] and not out["patient_unverified"] and out["content_issues"] == []


def test_nothing_written_gives_one_message_not_two(monkeypatch):
    out = audit(summarizer.UNREADABLE_CLINICAL, summarizer.UNREADABLE_PATIENT, monkeypatch)
    assert out["content_issues"] == [verifier.NO_SUMMARY_REASON]


def test_the_prompts_ask_for_urgency_doctor_advice_and_suggestions():
    prompt = summarizer._build_prompt("ctx", "FLAG_0: x", False)
    assert "how urgent" in prompt and "contact their doctor today" in prompt
    assert 'starts with "Consider"' in prompt and "Do not" in prompt
    assert "how urgent it is" in " ".join(verifier.VERIFIER_SYSTEM_PROMPT.split())


# ── 5. follow-up actions worded as treatment orders (SAFETY.md H18) ──────────────────────────────

@pytest.mark.parametrize("action", [
    "Admit the patient for emergency management of severe anemia - arrange cross-matched blood.",
    "Start diabetic emergency protocol: IV insulin infusion, isotonic saline.",
    "Urgently transfuse 2 units of packed red cells.",
    "Prescribe metformin 500 mg twice daily.",
    "Immediately stop the statin.",
])
def test_treatment_orders_are_found(action):
    assert summarizer.directive_actions([action]) == [action]


@pytest.mark.parametrize("action", [
    "Consider admission for evaluation of severe anaemia.",
    "Consider starting insulin after confirming the glucose.",
    "Order comprehensive labs (CBC, HbA1c, renal profile).",
    "Repeat fasting glucose within 24 hours.",
    "Review the patient the same day.",
    "Started on metformin last year - review adherence.",
])
def test_suggestions_and_tests_are_not_orders(action):
    assert summarizer.directive_actions([action]) == []


def test_urgency_written_with_a_non_breaking_hyphen_counts():
    """Groq writes "same‑day" with U+2011; the check must not call that summary silent."""
    flags = [{"test": "Hemoglobin", "value": 6.2, "severity": "CRITICAL"}]
    out = verifier.check_summaries("Hemoglobin 6.2 g/dL needs same‑day review.",
                                   "Please contact your doctor.", flags)
    assert out == {"clinical": None, "patient": None, "missing_critical": []}
