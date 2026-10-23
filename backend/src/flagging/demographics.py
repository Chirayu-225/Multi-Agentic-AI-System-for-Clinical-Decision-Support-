"""
src/flagging/demographics.py

Age and sex normalisation for reference-range selection.

Indian lab reports write these many ways: "Age/Gender: 54 Years / Male", "Age/Sex: 32Y/F",
"Sex: F", "54 Yrs M", "Gender : Female". GLiNER often returns the raw span or nothing. The
old range lookup only matched the words "male"/"female", so "F" or "M" fell through to the
FIRST range in the table — for haemoglobin that is the adult-male range, which flagged a
healthy woman with Hb 12.5 as anaemic.
"""

import re

MALE, FEMALE = "male", "female"

_SEX_WORDS = {
    "m": MALE, "male": MALE, "man": MALE, "boy": MALE, "mr": MALE, "purush": MALE,
    "f": FEMALE, "female": FEMALE, "woman": FEMALE, "girl": FEMALE, "mrs": FEMALE, "ms": FEMALE,
    "miss": FEMALE, "mahila": FEMALE, "stree": FEMALE,
}


def normalize_sex(raw) -> str | None:
    """'M', 'Male', '54 Years / Male', 'F', 'female ' -> 'male' | 'female' | None."""
    if not raw:
        return None
    tokens = re.findall(r"[a-z]+", str(raw).lower())
    found = {_SEX_WORDS[t] for t in tokens if t in _SEX_WORDS}
    # "Mrs" plus "M" etc. is contradictory, as is a string naming both — don't guess.
    return found.pop() if len(found) == 1 else None


def normalize_age(raw) -> float | None:
    """'54', '54 Years', '6 months', '54Y/M', '3 yrs 4 mths' -> age in years, or None."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if 0 <= raw < 130 else None
    s = str(raw).lower()
    # A bare "M"/"F" after the number is sex ("54M"), never months — months must be spelt out.
    m = re.search(r"(\d{1,3}(?:\.\d+)?)(?![\d.])(?:\s*(years?|yrs?|yo|y|months?|mths?|mos?|days?)(?![a-z]))?", s)
    if not m:
        return None
    n, unit = float(m.group(1)), (m.group(2) or "")
    if unit.startswith(("month", "mth", "mo")):
        n /= 12.0
    elif unit.startswith("day"):
        n /= 365.0
    return n if 0 <= n < 130 else None


# "Age/Gender: 54 Years / Male", "Age / Sex : 32Y/F", "Age: 54  Sex: M", "Gender: Female"
_COMBINED = re.compile(
    r"age\s*/\s*(?:sex|gender)\s*[:\-]?\s*(?P<age>\d{1,3}(?:\.\d+)?\s*[a-z]*)\s*[/,|]\s*(?P<sex>[a-z]+)",
    re.I)
_AGE = re.compile(r"\bage\s*[:\-]\s*(?P<age>\d{1,3}(?:\.\d+)?\s*(?:years?|yrs?|y|months?|mths?)?)", re.I)
_SEX = re.compile(r"\b(?:sex|gender)\s*[:\-]\s*(?P<sex>[a-z]+)", re.I)


def parse_demographics(text: str) -> dict:
    """Regex fallback for when NER misses age/sex. Returns {'age_years', 'sex'} (None if absent)."""
    age = sex = None
    head = (text or "")[:3000]  # demographics are in the header block
    m = _COMBINED.search(head)
    if m:
        age, sex = normalize_age(m.group("age")), normalize_sex(m.group("sex"))
    if age is None:
        m = _AGE.search(head)
        age = normalize_age(m.group("age")) if m else None
    if sex is None:
        m = _SEX.search(head)
        sex = normalize_sex(m.group("sex")) if m else None
    return {"age_years": age, "sex": sex}


def resolve_demographics(ner_patient: dict, text: str) -> dict:
    """Prefer NER output when it normalises cleanly, fall back to the regex parse."""
    ner_patient = ner_patient or {}
    parsed = parse_demographics(text)
    sex = normalize_sex(ner_patient.get("gender")) or parsed["sex"]
    age = normalize_age(ner_patient.get("age")) if ner_patient.get("age") else None
    if age is None:
        age = parsed["age_years"]
    return {"age_years": age, "sex": sex}
