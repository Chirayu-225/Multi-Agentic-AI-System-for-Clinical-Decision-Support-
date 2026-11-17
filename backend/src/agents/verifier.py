"""
src/agents/verifier.py

Verification Agent — the second agent in the Agentic Medical Assistant's multi-agent system.

Distinct from the Orchestrator in:
  - Role: critical auditor, not generator
  - Memory: only the current report's extracted data (no history)
  - Tools: none — structured evaluation only
  - Authority: can reject and trigger correction of the orchestrator's output
  - System prompt: adversarial perspective, looking for unsupported claims

Same pinned model (src/llm.py), different agent identity.

Both summaries (clinical + patient) are audited in a SINGLE Gemini call, not two
sequential ones. They used to run as separate calls with an artificial 5s sleep between
them purely to avoid tripping the rate limit — auditing both in one prompt removes that
call and the sleep, since there's nothing left to space out.
"""

import json
import re

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from src.llm import generate_text, is_rate_limit

from src.security.deidentification import redact


VERIFIER_SYSTEM_PROMPT = """You are a clinical verification agent. Your ONLY job is to audit
AI-generated medical summaries for factual accuracy against ground-truth extracted data.

You are NOT a generator. You are an auditor.

You will be given ONE ground-truth data block and TWO summaries to audit against it: a
CLINICAL summary and a PATIENT summary. Audit them independently — one may pass while the
other fails.

Rules (apply to each summary separately):
1. Every specific value and medication mentioned in a summary MUST appear in the ground truth.
2. A condition may be named only if it is in the ground truth, or if it is the plain meaning of a
   flagged value (for example "severe anaemia" for a CRITICAL low hemoglobin). Saying how urgent a
   flagged result is, or telling the patient to contact their doctor, is allowed. A cause or a
   diagnosis that the data does not support is NOT allowed.
3. If a summary mentions ANY value not in the ground truth, or breaks rule 2, its verdict is FAIL.
4. If a summary is consistent with the ground truth, its verdict is PASS.
5. Always list specific issues found for each summary, or "none" if clean.
6. When you rewrite a summary ("corrected"), keep every CRITICAL result with its value, keep the
   statement of how urgent it is, and in the patient summary keep the advice to contact their
   doctor. Remove only what was unsupported.

Output: ONLY a JSON object, no prose and no code fences, exactly this shape:
{"clinical": {"verdict": "PASS" or "FAIL", "issues": [<each unsupported claim as a string>],
              "corrected": <string, REQUIRED when verdict is FAIL: the summary rewritten using only
              ground-truth data; null when PASS>},
 "patient":  {"verdict": "PASS" or "FAIL", "issues": [...], "corrected": ...}}
"issues" is an empty list when a summary is clean.
"""


def _build_audit_prompt(ground_truth: str, clinical_summary: str, patient_summary: str) -> str:
    """The verifier's user prompt. With VERIFIER_SYSTEM_PROMPT it defines the verifier's version
    (src/prompts.py)."""
    return f"""GROUND TRUTH DATA (the only facts that exist in this patient's report):
{ground_truth}

CLINICAL SUMMARY TO AUDIT:
{clinical_summary}

PATIENT SUMMARY TO AUDIT:
{patient_summary}

Audit BOTH summaries against the ground truth, independently. Does either contain any values,
conditions, or medications NOT present in the ground truth?"""


def _build_ground_truth(entities: dict, flags: list, trend_flags: list,
                         redaction_map: dict | None = None,
                         drug_interactions: list | None = None) -> str:
    """Build a minimal, strict ground-truth string from verified extracted data.

    The patient name is redact()ed here too (same map the Orchestrator and
    Summarizer used — see deidentification.py) so this prompt matches the
    already-redacted clinical/patient summaries it's auditing: both sides
    reference the same placeholder, not a real name the audit would
    otherwise flag as an unexplained mismatch."""
    redaction_map = redaction_map or {}
    lines = []
    patient = entities.get("patient", {})
    if patient.get("name"):
        lines.append(f"Patient: {redact(patient['name'], redaction_map)}")
    age = patient.get("age_years")
    if age is not None:
        lines.append(f"Age: {age:g}" if age >= 1 else f"Age: {round(age * 12)} months")
    elif patient.get("age"):
        lines.append(f"Age: {patient['age']}")
    if patient.get("sex") or patient.get("gender"):
        lines.append(f"Sex: {patient.get('sex') or patient.get('gender')}")

    diagnoses = entities.get("clinical", {}).get("diagnoses", [])
    if diagnoses:
        lines.append(f"Diagnoses: {', '.join(diagnoses)}")

    meds = entities.get("medications", [])
    if meds:
        lines.append(f"Medications: {', '.join(m['name'] for m in meds)}")

    if flags:
        lines.append("Lab values (verified):")
        for f in flags:
            lines.append(f"  {f['test']}: {f['value']} {f.get('unit', '')} [{f['severity']}]")

    if trend_flags:
        lines.append("Trends detected:")
        for t in trend_flags:
            vals = " → ".join(str(v) for v in t.get("values_over_time", []))
            lines.append(f"  {t['test']}: {vals} ({t.get('total_change_pct', '')}% change)")

    # The Summarizer is shown these alerts, so they are part of the facts it may cite;
    # leaving them out would make the audit flag every correct mention as unsupported.
    if drug_interactions:
        lines.append("Medication interaction alerts (from the curated rule set):")
        for it in drug_interactions:
            lines.append(f"  [{it['severity']}] {' + '.join(it['drugs'])}: {it['effect']}")

    return "\n".join(lines) if lines else "No clinical data extracted."


class SideAudit(BaseModel):
    verdict: str
    # No defaults on purpose: the Gemini schema rejects/ignores them inconsistently, so every field
    # is required and the prompt shows all three.
    issues: list[str]
    corrected: str | None

    @field_validator("verdict")
    @classmethod
    def _verdict(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if v not in ("PASS", "FAIL"):
            raise ValueError("verdict must be PASS or FAIL")
        return v

    @model_validator(mode="after")
    def _fail_needs_a_correction(self):
        # A FAIL without a rewrite is the old silent bug: it used to be reported as "corrected"
        # while the text was unchanged. It is invalid, so it triggers the repair step instead.
        if self.verdict == "FAIL" and not (self.corrected or "").strip():
            raise ValueError("verdict FAIL requires a non-empty 'corrected' summary")
        return self


class AuditResult(BaseModel):
    clinical: SideAudit
    patient: SideAudit


class VerifierOutputInvalid(Exception):
    """The model's audit could not be turned into a valid AuditResult, even after one repair."""


def parse_audit(text: str) -> AuditResult:
    """Strict: JSON (optionally in a code fence) that validates against AuditResult, or raises."""
    t = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1)
    try:
        return AuditResult.model_validate(json.loads(t))
    except (json.JSONDecodeError, ValidationError) as e:
        raise VerifierOutputInvalid(str(e)[:600]) from e


def _repair_prompt(bad_output: str, error: str) -> str:
    return ("Your previous audit output was not valid.\n"
            f"Problem: {error}\n"
            f"Previous output:\n{bad_output[:4000]}\n\n"
            "Re-emit the SAME audit as ONLY a JSON object in the required shape. "
            "If a verdict is FAIL, \"corrected\" must contain the full corrected summary.")


async def _audit(prompt: str) -> AuditResult:
    """One audit call, plus at most ONE repair call if the output is invalid. Anything still
    invalid after that raises VerifierOutputInvalid, and the caller fails closed."""
    text = await generate_text(prompt, system_instruction=VERIFIER_SYSTEM_PROMPT,
                               label="VERIFIER", json_schema=AuditResult)
    try:
        return parse_audit(text)
    except VerifierOutputInvalid as first:
        text = await generate_text(_repair_prompt(text, str(first)), system_instruction=VERIFIER_SYSTEM_PROMPT,
                                   label="VERIFIER/repair", json_schema=AuditResult)
        return parse_audit(text)


def _side_result(side: SideAudit, original: str) -> dict:
    passed = side.verdict == "PASS"
    return {
        "passed": passed,
        "unverified": False,
        "issues": "none" if not side.issues else "; ".join(side.issues),
        "final_summary": original if passed else side.corrected.strip(),
        "was_corrected": not passed,
    }


# ── Completeness: a check the model does not get a vote on ─────────────────────────────────────
# The audit above looks for claims that are wrong. It says nothing about what a summary leaves
# out, and a summary that leaves out everything ("no significant findings") has no wrong claims
# at all - it passed, with "verified" beside it, on a report with two critical values. So the
# one omission that matters most is checked in code: every CRITICAL lab flag has to be mentioned
# in the clinical summary, by its value or by its name, or the summary is not verified.

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> set[float]:
    out = set()
    for token in _NUMBER.findall(text or ""):
        try:
            out.add(float(token.replace(",", "")))
        except ValueError:
            pass
    return out


def _names_for(test: str) -> set[str]:
    """Every way this codebase knows of writing a test's name, plus the name as flagged."""
    from src.flagging.risk_flagging import DIRECT_CANONICAL, LAB_ALIASES, _canonical_name
    raw = (test or "").strip().lower()
    names = {raw, re.sub(r"\s*\(.*?\)\s*", " ", raw).strip(), raw.replace("_", " ")}
    try:
        canonical = _canonical_name(test or "")
    except Exception:  # noqa: BLE001 - an odd name just means fewer aliases
        canonical = None
    if canonical:
        names.add(canonical.replace("_", " "))
        for table in (DIRECT_CANONICAL, LAB_ALIASES):
            names.update(alias for alias, target in table.items() if target == canonical)
    return {n for n in names if len(n) >= 2}


def missing_critical_results(summary: str, flags: list) -> list[str]:
    """Names of the CRITICAL lab flags that `summary` mentions neither by value nor by name."""
    from src.flagging.risk_flagging import Severity, normalize_severity
    text = (summary or "").lower()
    numbers = _numbers_in(text)
    missing = []
    for f in flags or []:
        if normalize_severity(f.get("severity")) != Severity.CRITICAL:
            continue
        try:
            value_cited = float(f.get("value")) in numbers
        except (TypeError, ValueError):
            value_cited = False
        name_cited = any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", text)
                         for n in _names_for(str(f.get("test") or "")))
        if not (value_cited or name_cited):
            missing.append(str(f.get("test") or "unnamed test"))
    return missing


NO_SUMMARY_REASON = ("No AI summary was produced for this report, so there was nothing to verify. "
                     "Review the flagged results directly.")
NO_PATIENT_SUMMARY = "No patient summary was produced for this report."
NO_URGENCY = ("This summary gives the critical result(s) but does not say they are critical or urgent. "
              "Read the flagged results; do not rely on this summary.")
NO_DOCTOR_ADVICE = ("This patient summary gives critical results but does not tell the patient to contact "
                    "their doctor.")

# Words that say a result is serious or time-sensitive, and words that point the patient to care.
# Deliberately broad: these checks catch a summary that says nothing of the kind, not bad style.
_URGENCY = re.compile(r"\b(critical(ly)?|urgent(ly)?|immediate(ly)?|same[-\u2010-\u2015 ]day|emergency|severe(ly)?|"
                      r"dangerous(ly)?|life[-\u2010-\u2015 ]threatening|prompt(ly)?|without delay|today|right away)\b",
                      re.IGNORECASE)
_CARE_CONTACT = re.compile(r"\b(doctor|physician|clinician|gp|healthcare provider|health care provider|"
                           r"care team|medical team|hospital|emergency|clinic)\b", re.IGNORECASE)


def _has_critical(flags: list) -> bool:
    from src.flagging.risk_flagging import Severity, normalize_severity
    return any(normalize_severity(f.get("severity")) == Severity.CRITICAL for f in flags or [])


def check_summaries(clinical: str, patient: str, flags: list) -> dict:
    """What a summary must contain, checked in code on the text that will be shown.

    Returns {"clinical": reason or None, "patient": reason or None, "missing_critical": [...]}.
    A reason means that summary is not verified, whatever the model audit said:
      * clinical: produced at all; names every CRITICAL result; says they are critical or urgent;
      * patient:  produced at all; with a CRITICAL result, tells the patient to contact their doctor.
    The last two exist because a summary that only lists the values ("Your hemoglobin is 6.2 g/dL")
    contains no false claim, so the audit passes it, yet it tells neither reader that anything is wrong.
    """
    from src.agents.summarizer import is_placeholder
    critical = _has_critical(flags)
    missing: list[str] = []
    clinical_reason = patient_reason = None
    if is_placeholder(clinical):
        clinical_reason = NO_SUMMARY_REASON
    else:
        missing = missing_critical_results(clinical, flags)
        if missing:
            clinical_reason = (f"This summary does not mention {len(missing)} critical result(s): "
                               f"{', '.join(missing)}. Read the flagged results; do not rely on this summary.")
        elif critical and not _URGENCY.search(clinical):
            clinical_reason = NO_URGENCY
    if is_placeholder(patient):
        patient_reason = NO_PATIENT_SUMMARY
    elif critical and not _CARE_CONTACT.search(patient):
        patient_reason = NO_DOCTOR_ADVICE
    return {"clinical": clinical_reason, "patient": patient_reason, "missing_critical": missing}


def content_issues(checked: dict) -> list[str]:
    """The reasons from check_summaries, in the order a clinician should read them."""
    if checked["clinical"] == NO_SUMMARY_REASON and checked["patient"] == NO_PATIENT_SUMMARY:
        return [NO_SUMMARY_REASON]                     # one message, not two, when nothing was written
    return [r for r in (checked["clinical"], checked["patient"]) if r]


def _not_verified(side: dict, reason: str) -> dict:
    return {**side, "passed": False, "unverified": True, "issues": reason}


async def verify_and_correct(
    clinical_summary: str,
    patient_summary: str,
    entities: dict,
    flags: list,
    trend_flags: list,
    redaction_map: dict | None = None,
    drug_interactions: list | None = None,
) -> dict:
    """
    Verification agent entry point.

    Audits both summaries against ground-truth extracted data in a SINGLE Gemini call.
    Returns corrected versions if hallucinations are found.

    `clinical_summary`/`patient_summary` are expected to still contain redaction
    placeholders (see summarizer.py / deidentification.py) — pass the SAME
    redaction_map used to produce them so the ground truth this function builds
    matches. This function does NOT rehydrate; the returned summaries still carry
    placeholders. main.py rehydrates once, after this call.

    Returns:
        {
            clinical_summary: str (corrected if needed),
            patient_summary: str (corrected if needed),
            clinical_corrected: bool,
            patient_corrected: bool,
            clinical_issues: str,
            patient_issues: str,
        }
    """
    from src.agents.summarizer import is_placeholder
    if is_placeholder(clinical_summary) and is_placeholder(patient_summary):
        # Nothing was written, so there is nothing to audit: do not spend a model call on it, and
        # above all do not let "no claims found" come back as "verified".
        return {
            "clinical_summary": clinical_summary, "patient_summary": patient_summary,
            "clinical_corrected": False, "patient_corrected": False,
            "clinical_unverified": True, "patient_unverified": True,
            "clinical_issues": NO_SUMMARY_REASON, "patient_issues": NO_SUMMARY_REASON,
            "clinical_missing_critical": missing_critical_results("", flags),
            "content_issues": [NO_SUMMARY_REASON],
            "clinical_audit_unverified": True,
        }

    ground_truth = _build_ground_truth(entities, flags, trend_flags, redaction_map, drug_interactions)

    prompt = _build_audit_prompt(ground_truth, clinical_summary, patient_summary)

    # Retry on rate limit. IMPORTANT — fail CLOSED: if the audit cannot run at
    # all (rate-limited past all retries, or any other error), we do NOT
    # report either summary as verified. A hallucination-check that silently
    # passes everything the moment it errors is worse than no
    # hallucination-check, because it looks like a safety net that isn't
    # there. Instead both are flagged `unverified` so the caller (API
    # response + UI) can surface "not reviewed" rather than implying the
    # audit ran and found nothing wrong.
    def _unverified(reason: str) -> dict:
        side = {
            "passed": False,
            "unverified": True,
            "issues": reason,
            "was_corrected": False,
        }
        return {
            "clinical": {**side, "final_summary": clinical_summary},
            "patient": {**side, "final_summary": patient_summary},
        }

    try:
        audit = await _audit(prompt)
        parsed = {
            "clinical": _side_result(audit.clinical, clinical_summary),
            "patient": _side_result(audit.patient, patient_summary),
        }
    except VerifierOutputInvalid:
        parsed = _unverified(
            "Verification output was malformed even after one repair attempt. "
            "This summary has NOT been audited — review manually."
        )
    except Exception as e:  # noqa: BLE001 - every failure fails CLOSED
        if is_rate_limit(e):
            parsed = _unverified(
                "Verification could not run (rate limit exhausted after retries). "
                "This summary has NOT been audited — review manually."
            )
        else:
            print(f"[VERIFIER] Error: {type(e).__name__}")
            parsed = _unverified(
                f"Verification could not run: {type(e).__name__}. "
                f"This summary has NOT been audited — review manually."
            )

    clinical_result = parsed["clinical"]
    patient_result = parsed["patient"]

    # Whether the model audit itself ran, before the content checks below add their own reasons. The
    # eval gate scores the audit on this, so a summary that is merely incomplete is not counted as an
    # audit failure.
    audit_unverified = clinical_result["unverified"]

    # Checks in code, applied to the text that will actually be shown (after any correction).
    checked = check_summaries(clinical_result["final_summary"], patient_result["final_summary"], flags)
    if checked["clinical"]:
        clinical_result = _not_verified(clinical_result, checked["clinical"])
    if checked["patient"]:
        reason = NO_SUMMARY_REASON if checked["patient"] == NO_PATIENT_SUMMARY else checked["patient"]
        patient_result = _not_verified(patient_result, reason)

    return {
        "clinical_missing_critical": checked["missing_critical"],
        "clinical_audit_unverified": audit_unverified,
        "content_issues": content_issues(checked),
        "clinical_summary": clinical_result["final_summary"],
        "patient_summary": patient_result["final_summary"],
        "clinical_corrected": clinical_result["was_corrected"],
        "patient_corrected": patient_result["was_corrected"],
        "clinical_unverified": clinical_result["unverified"],
        "patient_unverified": patient_result["unverified"],
        "clinical_issues": clinical_result["issues"],
        "patient_issues": patient_result["issues"],
    }
