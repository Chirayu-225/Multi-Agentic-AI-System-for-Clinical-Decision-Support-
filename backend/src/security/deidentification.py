"""
src/security/deidentification.py

De-identification for LLM calls — mitigates OWASP LLM02:2026 (Sensitive Information
Disclosure) for the specific case of sending clinical text to a third-party model API
(Gemini) rather than keeping it inside our own infrastructure.

Direct identifiers (patient full name, treating doctor, hospital/facility name, phone
numbers, email addresses, and government/medical ID-number-shaped sequences) are replaced
with stable placeholder tokens — e.g. "Priya Sharma" -> "[REDACTED_NAME_1]" — BEFORE any
prompt is built for Gemini. Gemini's response will then naturally reference the placeholder
instead of the real value. Only on the server, after all Gemini calls for a report are done,
do we substitute the placeholders back to their real values (rehydrate) for storage and for
the API response the UI renders. The real values never leave the process in a Gemini request.

Call sequence for one report upload (see run_orchestrator / generate_summaries /
verify_and_correct / main.py's upload_report):
  1. orchestrator.run_orchestrator() extracts entities locally (no LLM — see that module's
     docstring), then calls build_redaction_map() and redact()s the raw report text before
     it's included in the one Gemini prompt that still carries free text.
  2. The same redaction_map is threaded into generate_summaries() and verify_and_correct(),
     which redact() the patient-identifying line of their own prompts before calling Gemini.
     Their returned summaries therefore still contain the placeholder tokens, not real
     values — this is intentional, not a bug, and MUST stay that way until step 3.
  3. main.py's upload_report(), after all three agents are done, calls rehydrate() exactly
     once on every user-facing text field before building the API response / DB row. This
     is the ONLY point where placeholders are converted back to real values.

Deliberately NOT redacted: age, gender, diagnoses, medications, lab values, symptoms. These
are the actual clinical content the agents need to reason over and write about — redacting
them wouldn't protect anyone, it would just break the feature. Age/gender aren't direct
identifiers on their own either. This mirrors the HIPAA Safe Harbor distinction between
direct identifiers and clinical content; it is NOT a claim of full Safe Harbor compliance
(see the repo root SECURITY.md for the explicit scope and limits of this mitigation).

The phone/email/ID patterns below are heuristic pattern matching, same spirit as
llm_guardrails.py's injection patterns: deliberately over-inclusive, advisory-grade, not a
formally verified PII detector. A production system handling real PHI would use a dedicated
de-identification library (e.g. Microsoft Presidio) instead of hand-rolled regex — noted
here rather than overstated.
"""

import re
from functools import lru_cache

# Indian mobile-style: optional +91/91 prefix, then a 10-digit number starting 6-9.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
# 12-digit grouped ID (Aadhaar-shaped) or a long unbroken digit run typical of an MRN/
# patient-ID field — long enough that it won't collide with an ordinary lab value.
_ID_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{8,}\b")

RedactionMap = dict[str, str]


_TITLE_WORDS = {"dr", "mr", "mrs", "ms", "prof", "smt", "shri"}

# Label-based fallback. The NER model (GLiNER) is the primary source of the patient's and
# doctor's names, but it can miss them - and a name it misses would otherwise go to Gemini in
# the clear. Report headers are highly regular ("Patient Name: ...", "Referred by: ..."), so a
# labelled-line regex is a cheap, high-precision second net. Measured in eval/run_eval.py
# ("Regex only" column): it takes name leakage from ~100% to ~0% when the NER finds nothing.
_NAME_CHARS = r"[A-Za-z][A-Za-z.'\-]*(?:[ \t]+[A-Za-z][A-Za-z.'\-]*){0,3}"
_PATIENT_LABEL_RE = re.compile(rf"(?im)^[ \t]*(?:patient(?:'s)?[ \t]*name|pt\.?[ \t]*name|name[ \t]*of[ \t]*(?:the[ \t]*)?patient|mr[ \t]*/[ \t]*(?:mrs|ms)\.?|name|patient)[ \t]*[:\-][ \t]*({_NAME_CHARS})")
_DOCTOR_LABEL_RE = re.compile(rf"(?im)(?:referred[ \t]*by|ref\.?[ \t]*by|consultant|doctor|physician|pathologist|signature)[ \t]*[:\-][ \t]*({_NAME_CHARS})")
# Letterhead: the facility name sits in the first few lines and contains a facility word.
_FACILITY_WORD_RE = re.compile(
    r"\b(hospital|clinic|diagnostics?|laborator(?:y|ies)|labs?|medical cent(?:re|er)|health ?care|"
    r"health city|nursing home|polyclinic|institute of|path ?labs?)\b", re.I)
_NOT_A_NAME = {"not", "available", "na", "n/a", "nil", "none", "unknown", "male", "female", "other",
               "self", "report", "patient", "name", "date", "age", "sex", "test", "result", "the"}


def _plausible_name(candidate: str) -> bool:
    words = [w for w in re.split(r"[ \t]+", candidate.strip()) if w]
    if not words or any(re.search(r"\d", w) for w in words):
        return False
    return not any(w.lower().strip(".") in _NOT_A_NAME for w in words)


def build_redaction_map(entities: dict, raw_text: str) -> RedactionMap:
    """
    Build a {real_value: placeholder} map from GLiNER-extracted entities (patient name,
    treating doctor, hospital) plus regex-detected phone/email/ID-shaped strings in the
    raw report text. Call this once per report, right after (local, non-LLM) entity
    extraction, before any Gemini call is constructed.
    """
    mapping: RedactionMap = {}
    counters: dict[str, int] = {}

    def _add(value, label: str) -> str | None:
        if not value or not isinstance(value, str):
            return None
        value = value.strip()
        if len(value) < 2:
            return mapping.get(value)
        if value not in mapping:
            counters[label] = counters.get(label, 0) + 1
            mapping[value] = f"[REDACTED_{label}_{counters[label]}]"
        return mapping[value]

    def _add_name(value) -> None:
        """Register the full patient name AND each individual name word
        (>=3 letters, titles like "Dr" excluded) under the SAME placeholder,
        so a bare first or last name mentioned elsewhere in the report
        (separately from the full name GLiNER extracted) is also caught —
        not just an exact match on the full string. Deliberately NOT done
        for doctor/hospital below: their individual words ("Care",
        "General", "Hospital") are common clinical vocabulary, and
        redacting them globally would garble ordinary report text — a
        precision/recall tradeoff documented in the module docstring."""
        placeholder = _add(value, "NAME")
        if not placeholder or not value:
            return
        for part in re.findall(r"[A-Za-z]{3,}", value):
            if part.lower() in _TITLE_WORDS:
                continue
            mapping.setdefault(part, placeholder)

    patient = entities.get("patient", {}) or {}
    _add_name(patient.get("name"))

    providers = entities.get("providers", {}) or {}
    doctor = providers.get("doctor")
    placeholder = _add(doctor, "DOCTOR")
    if placeholder and isinstance(doctor, str):
        # Also register the name WITHOUT its title ("Dr. Meena Nair" -> "Meena Nair"):
        # reports mention the doctor both ways. Whole-string only - never the individual
        # words, for the same precision reason documented in _add_name().
        bare = re.sub(r"^(?:(?:dr|prof|mr|mrs|ms)\.?\s+)+", "", doctor.strip(), flags=re.I)
        if len(bare) >= 4 and bare != doctor.strip():
            mapping.setdefault(bare, placeholder)
    _add(providers.get("hospital"), "HOSPITAL")

    text = raw_text or ""
    # Label-based fallback (see _PATIENT_LABEL_RE): catches names the NER missed or truncated.
    for m in _PATIENT_LABEL_RE.finditer(text):
        if _plausible_name(m.group(1)):
            _add_name(m.group(1).strip())
    for m in _DOCTOR_LABEL_RE.finditer(text):
        cand = m.group(1).strip()
        if _plausible_name(cand):
            ph = _add(cand, "DOCTOR")
            bare = re.sub(r"^(?:(?:dr|prof|mr|mrs|ms)\.?\s+)+", "", cand, flags=re.I)
            if ph and len(bare) >= 4 and bare != cand:
                mapping.setdefault(bare, ph)
    # Letterhead facility name (first 6 non-empty lines) when the NER did not supply one.
    if not providers.get("hospital"):
        head = [ln.strip() for ln in text.splitlines() if ln.strip()][:6]
        for ln in head:
            if 4 <= len(ln) <= 80 and not re.search(r"\d", ln) and _FACILITY_WORD_RE.search(ln) \
                    and "department" not in ln.lower():
                _add(ln, "HOSPITAL")
                break
    for match in _EMAIL_RE.findall(text):
        _add(match, "EMAIL")
    for match in _PHONE_RE.findall(text):
        _add(match, "PHONE")
    for match in _ID_RE.findall(text):
        _add(match, "ID")

    return mapping


@lru_cache(maxsize=2048)
def _pattern_for(real_value: str) -> "re.Pattern[str]":
    """Case-insensitive, whitespace-flexible, word-bounded matcher for one identifier.
    - case-insensitive: printed lab headers are routinely UPPERCASE ("PRIYA SHARMA"); the
      original case-sensitive replace left those through (found by eval/run_eval.py).
    - whitespace-flexible: a name or ID split across a line break still matches.
    - word-bounded (only where the value starts/ends with a word character): "Priya" must not
      be carved out of "Priyanka"."""
    parts = [re.escape(p) for p in real_value.split()]
    body = r"\s+".join(parts)
    prefix = r"(?<!\w)" if real_value[:1].isalnum() else ""
    suffix = r"(?!\w)" if real_value[-1:].isalnum() else ""
    return re.compile(prefix + body + suffix, re.IGNORECASE)


def redact(text: str, redaction_map: RedactionMap) -> str:
    """Replace every real value in `text` with its placeholder. Longest values first, so a
    short identifier that's a substring of a longer one (e.g. a first name inside a full
    name) can't partially clobber the longer replacement."""
    if not text or not redaction_map:
        return text
    for real_value in sorted(redaction_map, key=len, reverse=True):
        text = _pattern_for(real_value).sub(lambda _m, ph=redaction_map[real_value]: ph, text)
    return text


def rehydrate(text: str, redaction_map: RedactionMap) -> str:
    """Replace every placeholder in `text` back with its real value. The inverse of
    redact() — call this exactly once, server-side, after all Gemini calls for a report
    are complete, on every field that will reach storage or the API response."""
    if not text or not redaction_map:
        return text
    for real_value, placeholder in redaction_map.items():
        text = text.replace(placeholder, real_value)
    return text
