"""
Reference-range selection and the lab's printed range.

Regression for: "F"/"M" (how most Indian lab reports write sex) did not match the old
"male"/"female" check and fell through to the FIRST range on file — for haemoglobin the
adult-male range — so a healthy woman with Hb 12.5 g/dL was flagged LOW. Age was ignored
entirely, so children were graded against adult ranges.
"""
import pytest

from src.flagging.risk_flagging import flag_lab_values, _get_range
from src.flagging.demographics import parse_demographics, normalize_sex


def _hb(value, **kw):
    lab = {"test": "Hemoglobin", "value": value, "unit": "g/dL", **kw.pop("lab", {})}
    return flag_lab_values([lab], **kw)[0]


@pytest.mark.parametrize("sex", ["F", "f", "Female", "FEMALE", "32Y/F", "Mrs"])
def test_female_written_any_way_gets_female_range(sex):
    f = _hb(12.5, gender=sex, age=32)
    assert f["direction"] == "normal"
    assert f["normal_range"]["min"] == 12.0
    assert "female" in f["range_basis"]


@pytest.mark.parametrize("sex", ["M", "Male", "54 Years / Male"])
def test_male_written_any_way_gets_male_range(sex):
    f = _hb(12.5, gender=sex, age=54)
    assert f["direction"] == "low"
    assert f["normal_range"]["min"] == 13.0


def test_unknown_sex_never_silently_uses_male_range():
    f = _hb(12.5, gender=None, age=40)
    assert f["direction"] == "normal"                 # 12.0-17.0, the widest adult range
    assert "sex not stated" in f["range_basis"]


def test_child_gets_paediatric_range():
    f = _hb(11.5, gender="M", age=8)
    assert f["direction"] == "normal"                 # adult male would call this low
    assert f["range_basis"] == "paediatric range"


def test_child_without_paediatric_range_is_labelled():
    ref = _get_range("alt", "F", 10)
    assert "no paediatric range" in ref["basis"]


def test_elderly_range_when_sex_unknown():
    assert _get_range("hemoglobin", None, 70)["basis"] == "elderly range"


def test_printed_range_discordance_is_flagged():
    # ICMR adult female 12.0-15.0 says normal; this lab's analyser range says low.
    f = _hb(12.5, gender="F", age=30, lab={"ref_low": 12.8, "ref_high": 16.0})
    assert f["direction"] == "normal" and f["lab_direction"] == "low"
    assert f["range_discordant"] is True
    assert "printed range" in f["message"]


def test_printed_range_agreement_is_not_flagged():
    f = _hb(10.0, gender="F", age=30, lab={"ref_low": 12.0, "ref_high": 15.0})
    assert f["direction"] == f["lab_direction"] == "low"
    assert f["range_discordant"] is False


def test_demographics_fallback_parses_common_headers():
    assert parse_demographics("Age/Gender: 54 Years / Male") == {"age_years": 54.0, "sex": "male"}
    assert parse_demographics("Age / Sex : 32Y/F") == {"age_years": 32.0, "sex": "female"}
    assert parse_demographics("Age: 6 months\nSex: F")["age_years"] == 0.5
    assert normalize_sex("M/F") is None               # contradictory -> don't guess


def test_lab_parser_reads_printed_ranges():
    from src.ner.medical_ner import _extract_lab_values_regex
    labs = {l["test"]: l for l in _extract_lab_values_regex(
        "Hemoglobin        12.5    g/dL   12.0 - 15.0\n"
        "HbA1c: 7.8 % (< 5.7)\n"
        "Serum Creatinine - 1.9 mg/dL 0.7 to 1.3\n"
    )}
    assert (labs["hemoglobin"]["ref_low"], labs["hemoglobin"]["ref_high"]) == (12.0, 15.0)
    assert labs["hba1c"]["ref_high"] == 5.7 and "ref_low" not in labs["hba1c"]
    assert labs["creatinine"]["ref_high"] == 1.3


# ── Normalised demographics flow into prompts and FHIR ────────────────────────
# Regression: the orchestrator stores age_years as a number; the summarizer joined patient
# fields as strings and crashed (found by a real end-to-end upload, not by the stubbed suite).

PATIENT = {"name": "Asha Rao", "age": "34 Years / F", "gender": "34 Years / F",
           "age_years": 34.0, "sex": "female"}


def test_summarizer_context_handles_numeric_age():
    from src.agents.summarizer import _build_context
    ctx = _build_context({"patient": PATIENT}, [], [], [], [], [], {}, [])
    assert "age 34" in ctx and "female" in ctx


def test_verifier_ground_truth_handles_numeric_age():
    from src.agents.verifier import _build_ground_truth
    gt = _build_ground_truth({"patient": PATIENT}, [], [])
    assert "Age: 34" in gt and "Sex: female" in gt


def test_infant_age_rendered_in_months():
    from src.agents.summarizer import _build_context
    ctx = _build_context({"patient": {"age_years": 0.5, "sex": "male"}}, [], [], [], [], [], {}, [])
    assert "age 6 months" in ctx


def test_fhir_gender_from_raw_report_text():
    from src.interop.fhir import build_bundle
    for patient, expected in (({"gender": "34 Years / F"}, "female"),
                              ({"sex": "male", "gender": "M"}, "male"),
                              ({"gender": "unknown"}, None)):
        b = build_bundle({"patient_id": "P1", "report_id": "r", "entities": {"patient": patient}})
        pat = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Patient")
        assert pat.get("gender") == expected
