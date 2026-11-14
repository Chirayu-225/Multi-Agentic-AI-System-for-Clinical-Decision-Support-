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
1. Every specific value, condition, or medication mentioned in a summary MUST appear in the
   ground truth
2. If a summary mentions ANY value not in the ground truth, its verdict is FAIL
3. If a summary is consistent with the ground truth, its verdict is PASS
4. Always list specific issues found for each summary, or "none" if clean

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

    return {
        "clinical_summary": clinical_result["final_summary"],
        "patient_summary": patient_result["final_summary"],
        "clinical_corrected": clinical_result["was_corrected"],
        "patient_corrected": patient_result["was_corrected"],
        "clinical_unverified": clinical_result["unverified"],
        "patient_unverified": patient_result["unverified"],
        "clinical_issues": clinical_result["issues"],
        "patient_issues": patient_result["issues"],
    }
