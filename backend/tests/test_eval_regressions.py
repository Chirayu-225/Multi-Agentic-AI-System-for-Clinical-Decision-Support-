"""Pins the real bugs the evaluation harness found, plus a smoke test of the harness itself."""
import pytest

from src.flagging.risk_flagging import _canonical_name, flag_lab_values
from src.ner.medical_ner import _extract_lab_values_regex, _parse_number, _clean_lab_name
from src.security.deidentification import build_redaction_map, redact, rehydrate


@pytest.mark.parametrize("name", ["Post Prandial Glucose", "postprandial glucose", "PPBS", "Post-Prandial Glucose", "PP Glucose"])
def test_post_prandial_glucose_is_not_graded_against_fasting_range(name):
    assert _canonical_name(name) == "glucose_postprandial"
    assert _canonical_name(_clean_lab_name(name)) == "glucose_postprandial"


def test_post_prandial_value_in_its_own_normal_range_is_not_flagged():
    flag = flag_lab_values([{"test": "Post Prandial Glucose", "value": 130, "unit": "mg/dL"}])[0]
    assert str(flag["severity"]).endswith("NORMAL")      # 130 is normal post-prandial (<=140), high if fasting (>100)


@pytest.mark.parametrize("raw,expected", [("1,50,000", 150000), ("150,000", 150000), ("12,562", 12562),
                                          ("9,2", 9.2), ("9.2", 9.2), ("4,50", 4.5), ("1,23,45,678", 12345678)])
def test_number_parsing_handles_indian_and_decimal_comma(raw, expected):
    assert _parse_number(raw) == expected


@pytest.mark.parametrize("line,test,value", [
    ("Hemoglobin: 9.2 g/dL", "hemoglobin", 9.2),
    ("Creatinine - 1.9 mg/dL", "creatinine", 1.9),
    ("Haemoglobin                 14.0      g/dL        13.0 - 17.0", "haemoglobin", 14.0),
    ("Platelet Count: 1,50,000 cells/mcL", "platelets", 150000),
    ("Potassium                  5,4       mEq/L       3.5 - 5.0", "potassium", 5.4),
    ("Sodium was found to be 139 mEq/L on this sample.", "sodium", 139),
])
def test_extraction_layouts(line, test, value):
    got = _extract_lab_values_regex(line)
    assert len(got) == 1 and got[0]["value"] == value
    assert _canonical_name(got[0]["test"]) == _canonical_name(test)


@pytest.mark.parametrize("line", [
    "MRN: 8429141456", "Phone: 9876543210", "Aadhaar: 9785 3045 7201", "Age/Sex: 67 / Male",
    "Report Date: 2026-02-10", "Bed No: 12", "Page 2 of 3",
])
def test_identifiers_and_demographics_are_not_extracted_as_labs(line):
    assert _extract_lab_values_regex(line) == []


def test_uppercase_and_wrapped_names_are_redacted():
    ents = {"patient": {"name": "Priya Sharma"}, "providers": {"doctor": "Dr. Meena Nair"}}
    text = "PRIYA SHARMA\nPriya\nSharma was seen by Meena Nair and Dr Meena Nair. Priyanka is unrelated."
    out = redact(text, build_redaction_map(ents, text))
    for leaked in ("PRIYA", "Priya\n", "Sharma was", "Meena", "Nair"):
        assert leaked not in out, leaked
    assert "Priyanka" in out                       # word-bounded: a different name survives


def test_label_fallback_catches_name_when_ner_found_nothing():
    text = "Sunrise General Hospital\nPatient Name: Anita Rao\nReferred by: Dr. Kavya Nair\n"
    m = build_redaction_map({}, text)
    out = redact(text, m)
    for s in ("Anita", "Rao", "Kavya", "Nair", "Sunrise General Hospital"):
        assert s not in out


@pytest.mark.parametrize("text", ["Patient Name: Not Available", "Name: Male", "Patient: Report"])
def test_label_fallback_ignores_non_names(text):
    assert build_redaction_map({}, text) == {}


def test_harness_runs_and_meets_its_own_thresholds(monkeypatch):
    import eval.run_eval as ev
    monkeypatch.setattr(ev, "N_CASES", 60)
    result = ev.run()
    assert ev.check(result) == []
    assert "Evaluation results" in ev.to_markdown(result)
