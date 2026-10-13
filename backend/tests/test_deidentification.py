from src.security.deidentification import build_redaction_map, redact, rehydrate

ENT = {
    "patient": {"name": "Priya Sharma", "age": "42", "gender": "Female"},
    "providers": {"doctor": "Dr. Anita Rao", "hospital": "Sunrise General Hospital"},
}
TEXT = (
    "Patient: Priya Sharma  Ph: 9876543210  priya.s@example.com\n"
    "MRN 20240011223  Aadhaar 1234 5678 9012\nHb 9.2 g/dL. Priya's fever has resolved."
)


def test_roundtrip_restores_everything():
    m = build_redaction_map(ENT, TEXT)
    # A bare "Priya" rehydrates to the full name by design (documented in
    # deidentification.py) - every other character must round-trip exactly.
    assert rehydrate(redact(TEXT, m), m) == TEXT.replace("Priya's", "Priya Sharma's")


def test_no_identifier_survives_redaction():
    m = build_redaction_map(ENT, TEXT)
    out = redact(TEXT, m)
    for ident in ["Priya", "Sharma", "9876543210", "priya.s@example.com",
                  "20240011223", "1234 5678 9012", "Anita Rao", "Sunrise General Hospital"]:
        assert ident not in out, f"{ident} leaked"


def test_clinical_content_untouched():
    m = build_redaction_map(ENT, TEXT)
    out = redact(TEXT, m)
    assert "Hb 9.2 g/dL" in out


def test_generic_words_not_overredacted():
    m = build_redaction_map(ENT, TEXT)
    sample = "Return to the hospital for follow-up care."
    assert redact(sample, m) == sample


def test_empty_map_is_noop():
    assert redact("abc", {}) == "abc"
    assert rehydrate("abc", {}) == "abc"


def test_title_words_not_registered_as_name_parts():
    m = build_redaction_map({"patient": {"name": "Dr Ravi Kumar"}}, "")
    assert "Dr" not in m
