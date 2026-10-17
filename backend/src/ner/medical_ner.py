"""
src/ner/medical_ner.py

Medical Named Entity Recognition using GLiNER.
Extracts: patient info, diagnoses, medications,
lab values with units, dates, doctor names.
"""

from gliner import GLiNER
import re

# Medical entity labels GLiNER will extract
MEDICAL_LABELS = [
    "patient name",
    "patient age",
    "patient gender",
    "date",
    "diagnosis",
    "symptom",
    "medication",
    "dosage",
    "lab test",
    "lab value",
    "lab unit",
    "doctor name",
    "hospital name",
    "body part",
    "procedure",
]

# Regex patterns for numeric lab values (catches "Hb: 9.2 g/dL" style)
# Anchored to start of word boundary, max 5 words for test name
LAB_VALUE_PATTERN = re.compile(
    r"(?<!\w)(?P<test>[A-Za-z][A-Za-z0-9\-/]*(?:\s+[A-Za-z][A-Za-z0-9\-/]*){0,4}?)\s*[:\-=]\s*"
    r"(?P<value>\d+\.?\d*)\s*"
    r"(?P<unit>g/dL|mg/dL|mEq/L|U/L|mIU/L|%|cells/mcL|mmol/L|IU/L|ng/mL|pg/mL)?",
    re.IGNORECASE,
)

# Known noise words to strip from extracted lab names
_NOISE_WORDS = {
    "male", "female", "laboratory", "results", "lab", "test",
    "report", "patient", "date", "value", "level", "count",
    "serum", "blood", "urine", "plasma",
}

# Known lab test names to match against extracted text
KNOWN_LAB_TESTS = [
    "hemoglobin", "haemoglobin", "glucose", "creatinine", "wbc",
    "platelets", "sodium", "potassium", "bilirubin", "alt", "ast",
    "sgpt", "sgot", "tsh", "hba1c", "hb", "fbs", "ppbs", "tlc",
]


def _clean_lab_name(raw: str) -> str:
    """
    Clean a raw lab name extracted by GLiNER or regex.
    - Strips noise words from the beginning
    - Extracts the last meaningful word if name is too long
    - Returns cleaned name for alias lookup
    """
    words = raw.lower().strip().split()

    # Post-prandial glucose must be recognised BEFORE the generic "glucose" shortcut below,
    # or it is collapsed to "glucose" and graded against the fasting range.
    joined = " ".join(words)
    if ("prandial" in joined or "ppbs" in words or "pp" in words or "ppg" in words) and \
            ("glucose" in words or "sugar" in words or "ppbs" in words or "ppg" in words or "prandial" in joined):
        return "post prandial glucose"

    # If a known lab test name appears anywhere in the string, extract it
    for known in KNOWN_LAB_TESTS:
        if known in words:
            return known

    # Strip leading noise words
    cleaned = [w for w in words if w not in _NOISE_WORDS]
    if cleaned:
        # If still more than 3 words, take the last meaningful word
        if len(cleaned) > 3:
            return cleaned[-1]
        return " ".join(cleaned)

    return raw.lower().strip()

_model = None


def _get_model():
    global _model
    if _model is None:
        # urchade/gliner_mediumv2.1 is a good balance of size and accuracy
        _model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
    return _model


def extract_entities(text: str, threshold: float = 0.4) -> dict:
    """
    Run GLiNER NER on text and return categorised entities.

    Returns:
        {
            "patient": { name, age, gender },
            "clinical": { diagnoses, symptoms, procedures, body_parts },
            "medications": [ { name, dosage } ],
            "lab_values": [ { test, value, unit } ],
            "dates": [],
            "providers": { doctor, hospital },
            "raw_entities": []   # all GLiNER outputs
        }
    """
    model = _get_model()

    # GLiNER works best on chunks under ~512 tokens
    # Truncate if needed
    if len(text.split()) > 400:
        # keep line breaks: the lab parser below is line-based
        kept, n = [], 0
        for ln in text.split("\n"):
            w = len(ln.split())
            if n + w > 400:
                break
            kept.append(ln)
            n += w
        text = "\n".join(kept)

    raw = model.predict_entities(text, MEDICAL_LABELS, threshold=threshold)

    result = {
        "patient": {"name": None, "age": None, "gender": None},
        "clinical": {
            "diagnoses": [],
            "symptoms": [],
            "procedures": [],
            "body_parts": [],
        },
        "medications": [],
        "lab_values": [],
        "dates": [],
        "providers": {"doctor": None, "hospital": None},
        "raw_entities": raw,
    }

    for ent in raw:
        label = ent["label"].lower()
        text_val = ent["text"].strip()

        if label == "patient name":
            result["patient"]["name"] = text_val
        elif label == "patient age":
            result["patient"]["age"] = text_val
        elif label == "patient gender":
            result["patient"]["gender"] = text_val
        elif label == "diagnosis":
            if text_val not in result["clinical"]["diagnoses"]:
                result["clinical"]["diagnoses"].append(text_val)
        elif label == "symptom":
            if text_val not in result["clinical"]["symptoms"]:
                result["clinical"]["symptoms"].append(text_val)
        elif label == "procedure":
            if text_val not in result["clinical"]["procedures"]:
                result["clinical"]["procedures"].append(text_val)
        elif label == "body part":
            if text_val not in result["clinical"]["body_parts"]:
                result["clinical"]["body_parts"].append(text_val)
        elif label == "medication":
            # NER sometimes tags only the dosage-form prefix ("Tab", "Inj.") as a drug
            if re.fullmatch(r"(tab|tabs|tablet|tablets|cap|caps|capsule|inj|injection|syp|syrup|drops|oint)\.?", text_val, re.I):
                continue
            result["medications"].append({"name": text_val, "dosage": None})
        elif label == "dosage":
            # Attach dosage to the last medication if available
            if result["medications"]:
                result["medications"][-1]["dosage"] = text_val
        elif label in ("lab test", "lab value"):
            cleaned = _clean_lab_name(text_val)
            result["lab_values"].append({"test": cleaned, "value": None, "unit": None})
        elif label == "date":
            if text_val not in result["dates"]:
                result["dates"].append(text_val)
        elif label == "doctor name":
            result["providers"]["doctor"] = text_val
        elif label == "hospital name":
            result["providers"]["hospital"] = text_val

    # Supplement with line-by-line lab value extraction
    regex_labs = _extract_lab_values_regex(text)
    for lab in regex_labs:
        existing_tests = [l["test"].lower() for l in result["lab_values"]]
        if lab["test"].lower() in existing_tests:
            # Update existing entry that has null value
            for existing in result["lab_values"]:
                if existing["test"].lower() == lab["test"].lower() and existing["value"] is None:
                    existing["value"] = lab["value"]
                    existing["unit"]  = lab["unit"]
        else:
            result["lab_values"].append(lab)

    return result


# Units recognised in report text (case-insensitive).
_UNIT_ALT = r"g/dL|mg/dL|mEq/L|U/L|mIU/L|%|cells/mcL|cells/cumm|/cumm|mmol/L|IU/L|ng/mL|pg/mL|mcg/dL|uIU/mL|µIU/mL"
# Numbers as they appear on Indian lab reports: plain (9.2), decimal comma (9,2), and
# thousands grouping in either Western (150,000) or lakh style (1,50,000). The grouped
# forms are tried FIRST: otherwise "1,50,000" parsed as 1 - a silent, dangerous misread
# (a platelet count of 1 would be graded CRITICAL).
_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:[.,]\d+)?"


def _parse_number(raw: str) -> float:
    """'1,50,000' -> 150000 ; '150,000' -> 150000 ; '9,2' -> 9.2 ; '9.2' -> 9.2.
    A comma followed by exactly 1-2 digits at the end is a decimal comma; groups of 3
    (optionally preceded by lakh-style pairs) are thousands separators."""
    raw = raw.strip()
    if re.fullmatch(r"\d{1,3}(?:,\d{2})*,\d{3}(?:\.\d+)?", raw) or re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        return float(raw.replace(",", ""))
    if re.fullmatch(r"\d+,\d{1,2}", raw):
        return float(raw.replace(",", "."))
    return float(raw.replace(",", ""))

# Lines that look like "Label: number" but are identifiers/demographics, not lab results. Without
# this, MRNs, phone numbers and ages were being extracted as "lab values" (eval/run_eval.py
# measured 25% extraction precision), shown in the UI, and flagged "no ICMR range".
_NON_LAB_LABELS = re.compile(
    r"\b(mrn|uhid|ip ?no|op ?no|reg(?:istration)?|bed|ward|room|phone|mobile|tel|contact|fax|pin ?code?|zip|"
    r"aadhaar|aadhar|pan|id|age|age/sex|sex|dob|date|time|page|sample|specimen|barcode|lab no|invoice|bill|"
    r"receipt|ref(?:erred)? ?no|accession|visit|episode|height|weight|bmi|license|licence|nabl)\b", re.I)

_TABLE_RE = re.compile(
    rf"^(?P<test>[A-Za-z][A-Za-z0-9 /().%+\-]{{1,40}}?)\s{{2,}}(?P<value>{_NUM})\s*(?P<unit>{_UNIT_ALT})?(?:\s|$)", re.I)
_COLON_RE = re.compile(
    rf"^(?P<test>[A-Za-z][A-Za-z0-9 /().%+\-]{{1,40}}?)\s*:\s*(?P<value>{_NUM})\s*(?P<unit>{_UNIT_ALT})?", re.I)
_DASH_RE = re.compile(
    rf"^(?P<test>[A-Za-z][A-Za-z0-9 /().%+]{{1,40}}?)\s+[-\u2013=]\s*(?P<value>{_NUM})\s*(?P<unit>{_UNIT_ALT})?", re.I)
# Loose single-space row: "Hemoglobin 10.2 g/dL 13.0 - 17.0". Unit REQUIRED and the test name
# is letters only, so ranges and stray numbers cannot be mistaken for a result.
_LOOSE_RE = re.compile(
    rf"^(?P<test>[A-Za-z][A-Za-z /().%+\-]{{1,40}}?)\s+(?P<value>{_NUM})\s*(?P<unit>{_UNIT_ALT})(?:\s|$)", re.I)
_NUMBER_TOKEN = re.compile(r"(?:^|\s)\d+(?:[.,]\d+)?(?:\s|$)")
_SENTENCE_RE = re.compile(
    rf"(?P<test>[A-Za-z][A-Za-z0-9 /().%+\-]{{1,40}}?)\s+(?:was found to be|was measured at|was|is|of|measured|measures|came to|=)\s+(?P<value>{_NUM})\s*(?P<unit>{_UNIT_ALT})", re.I)


def _extract_lab_values_regex(text: str) -> list[dict]:
    """
    Structured lab value extraction, line by line, trying four real-world layouts in order:
      1. "Test: 9.2 g/dL"                      (colon)
      2. "Test - 9.2 g/dL"                      (dash / equals)
      3. "Test        9.2    g/dL   12 - 15"    (table: columns separated by 2+ spaces)
      4. "Test was found to be 9.2 g/dL ..."    (sentence; unit required)
    A match is kept only if it carries a recognised unit OR the test name maps to a known
    ICMR test - this is what keeps identifiers, ages and dates out. For table rows the FIRST
    number is the result (later numbers are the reference range).
    """
    from src.flagging.risk_flagging import _canonical_name, REFERENCE_RANGES

    matches, seen = [], set()
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) > 200:
            continue
        m = None
        for rx in (_COLON_RE, _DASH_RE, _TABLE_RE, _LOOSE_RE, _SENTENCE_RE):
            cand = rx.match(line) if rx is not _SENTENCE_RE else rx.search(line)
            # A "test name" that contains a stand-alone number is really a result row that
            # swallowed its own value ("Hemoglobin 10.2 g/dL 13.0 - 17.0" read by the dash
            # pattern as test="Hemoglobin 10.2 g/dL 13.0", value=17.0). Reject and try the next layout.
            if cand and not _NUMBER_TOKEN.search(cand.group("test")):
                m = cand
                break
        if not m:
            continue

        raw_test = m.group("test").strip()
        if _NON_LAB_LABELS.search(raw_test):
            continue
        test = _clean_lab_name(raw_test)
        if not test or test in _NOISE_WORDS or len(test) < 2:
            continue
        unit = (m.group("unit") or "").strip()
        known = _canonical_name(test) in REFERENCE_RANGES
        if not unit and not known:
            continue
        try:
            value = _parse_number(m.group("value"))
        except ValueError:
            continue
        if test not in seen:
            seen.add(test)
            matches.append({"test": test, "value": value, "unit": unit})

    # Layout 5: one cell per line (common when a PDF table is extracted):
    #   Hemoglobin / 10.2 / g/dL / 13.0 - 17.0.   A recognised unit on the line after the
    #   number is REQUIRED here, which keeps this pass from picking up stray numbers.
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    num_only = re.compile(rf"^(?P<value>{_NUM})$")
    unit_only = re.compile(rf"^(?P<unit>{_UNIT_ALT})$", re.I)
    for i in range(len(lines) - 2):
        name = lines[i]
        if not (2 <= len(name) <= 40) or not re.match(r"^[A-Za-z][A-Za-z0-9 /().%+\-]*$", name):
            continue
        mv, mu = num_only.match(lines[i + 1]), unit_only.match(lines[i + 2])
        if not (mv and mu) or _NON_LAB_LABELS.search(name):
            continue
        test = _clean_lab_name(name)
        if not test or test in _NOISE_WORDS or test in seen:
            continue
        try:
            value = _parse_number(mv.group("value"))
        except ValueError:
            continue
        seen.add(test)
        matches.append({"test": test, "value": value, "unit": mu.group("unit")})
    return matches


def merge_chunk_entities(chunk_results: list[dict]) -> dict:
    """
    Merge NER results from multiple text chunks into one unified result.
    Deduplicates across chunks.
    """
    merged = {
        "patient": {"name": None, "age": None, "gender": None},
        "clinical": {
            "diagnoses": [],
            "symptoms": [],
            "procedures": [],
            "body_parts": [],
        },
        "medications": [],
        "lab_values": [],
        "dates": [],
        "providers": {"doctor": None, "hospital": None},
        "raw_entities": [],
    }

    for res in chunk_results:
        # Patient info — take first non-null
        for key in ("name", "age", "gender"):
            if not merged["patient"][key] and res["patient"].get(key):
                merged["patient"][key] = res["patient"][key]

        # Clinical lists — deduplicate
        for key in ("diagnoses", "symptoms", "procedures", "body_parts"):
            for item in res["clinical"].get(key, []):
                if item not in merged["clinical"][key]:
                    merged["clinical"][key].append(item)

        # Medications
        existing_meds = [m["name"].lower() for m in merged["medications"]]
        for med in res.get("medications", []):
            if med["name"].lower() not in existing_meds:
                merged["medications"].append(med)
                existing_meds.append(med["name"].lower())

        # Lab values
        existing_labs = [l["test"].lower() for l in merged["lab_values"]]
        for lab in res.get("lab_values", []):
            if lab["test"].lower() not in existing_labs:
                merged["lab_values"].append(lab)
                existing_labs.append(lab["test"].lower())

        # Dates
        for d in res.get("dates", []):
            if d not in merged["dates"]:
                merged["dates"].append(d)

        # Providers
        for key in ("doctor", "hospital"):
            if not merged["providers"][key] and res["providers"].get(key):
                merged["providers"][key] = res["providers"][key]

        merged["raw_entities"].extend(res.get("raw_entities", []))

    return merged
