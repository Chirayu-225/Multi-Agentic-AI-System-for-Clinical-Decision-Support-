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
        text = " ".join(text.split()[:400])

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


def _extract_lab_values_regex(text: str) -> list[dict]:
    """
    Structured lab value extraction using two approaches:
    1. Line-by-line parser — primary method, catches "Test: value unit" format
    2. Regex pattern — fallback for inline formats
    """
    matches = []
    seen_tests = set()

    # ── Method 1: Line-by-line parser (most reliable) ─────────────────────
    for line in text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue

        # Split on first colon only
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue

        raw_test = parts[0].strip()
        rest     = parts[1].strip()

        # Extract numeric value from rest of line
        num_match = re.search(
            r"(\d+\.?\d*)\s*(g/dL|mg/dL|mEq/L|U/L|mIU/L|%|cells/mcL|mmol/L|IU/L|ng/mL|pg/mL)?",
            rest
        )
        if not num_match:
            continue

        value = float(num_match.group(1))
        unit  = num_match.group(2) or ""

        # Clean the test name
        test = _clean_lab_name(raw_test)

        # Skip noise-only names
        if not test or test in _NOISE_WORDS or len(test) < 2:
            continue

        if test not in seen_tests:
            seen_tests.add(test)
            matches.append({"test": test, "value": value, "unit": unit.strip()})

    # ── Method 2: Regex fallback for inline formats ────────────────────────
    if not matches:
        for m in LAB_VALUE_PATTERN.finditer(text):
            raw_test = m.group("test").strip()
            value    = m.group("value")
            unit     = m.group("unit") or ""
            test     = _clean_lab_name(raw_test)

            if len(test) < 2 or not value or test in _NOISE_WORDS:
                continue
            if test not in seen_tests:
                seen_tests.add(test)
                try:
                    matches.append({
                        "test": test,
                        "value": float(value),
                        "unit": unit.strip(),
                    })
                except ValueError:
                    pass

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
