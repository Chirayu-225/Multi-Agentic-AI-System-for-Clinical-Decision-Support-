"""
src/agents/verifier.py

Verification Agent — the second agent in the Agentic Medical Assistant's multi-agent system.

Distinct from the Orchestrator in:
  - Role: critical auditor, not generator
  - Memory: only the current report's extracted data (no history)
  - Tools: none — structured evaluation only
  - Authority: can reject and trigger correction of the orchestrator's output
  - System prompt: adversarial perspective, looking for unsupported claims

Same model (gemini-2.5-flash), different agent identity.

Both summaries (clinical + patient) are audited in a SINGLE Gemini call, not two
sequential ones. They used to run as separate calls with an artificial 5s sleep between
them purely to avoid tripping the rate limit — auditing both in one prompt removes that
call and the sleep, since there's nothing left to space out.
"""

import os
import re
import json
import asyncio
import time

import google.generativeai as genai

from src.security.deidentification import redact

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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

Output format (strictly follow this — both blocks required, in this order):
VERDICT_CLINICAL: PASS or FAIL
ISSUES_CLINICAL: <specific unsupported claims, or "none">
CORRECTED_CLINICAL: <only if VERDICT_CLINICAL is FAIL — rewrite using only ground truth data>
VERDICT_PATIENT: PASS or FAIL
ISSUES_PATIENT: <specific unsupported claims, or "none">
CORRECTED_PATIENT: <only if VERDICT_PATIENT is FAIL — rewrite using only ground truth data>
"""


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
    if patient.get("age"):
        lines.append(f"Age: {patient['age']}")
    if patient.get("gender"):
        lines.append(f"Gender: {patient['gender']}")

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


def _parse_side(text: str, suffix: str, fallback_summary: str) -> dict:
    """Extract one summary's VERDICT/ISSUES/CORRECTED block for the given suffix
    (CLINICAL or PATIENT) out of the combined audit response."""
    verdict_match = re.search(rf"VERDICT_{suffix}:\s*(PASS|FAIL)", text, re.IGNORECASE)
    passed = bool(verdict_match) and verdict_match.group(1).upper() == "PASS"

    # Stop at the next labeled section (either side's own CORRECTED_, or the
    # other side's VERDICT_) so PATIENT's block doesn't swallow CLINICAL's,
    # and CORRECTED_CLINICAL doesn't run on into VERDICT_PATIENT.
    issues_match = re.search(
        rf"ISSUES_{suffix}:\s*(.+?)(?:\n(?:CORRECTED_{suffix}|VERDICT_CLINICAL|VERDICT_PATIENT):|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    issues = issues_match.group(1).strip() if issues_match else ""

    corrected_match = re.search(
        rf"CORRECTED_{suffix}:\s*(.+?)(?:\n(?:VERDICT_CLINICAL|VERDICT_PATIENT):|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    corrected_text = corrected_match.group(1).strip() if corrected_match else fallback_summary

    return {
        "passed": passed,
        "unverified": False,
        "issues": issues,
        "final_summary": fallback_summary if passed else corrected_text,
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
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=VERIFIER_SYSTEM_PROMPT,
    )

    ground_truth = _build_ground_truth(entities, flags, trend_flags, redaction_map, drug_interactions)

    prompt = f"""GROUND TRUTH DATA (the only facts that exist in this patient's report):
{ground_truth}

CLINICAL SUMMARY TO AUDIT:
{clinical_summary}

PATIENT SUMMARY TO AUDIT:
{patient_summary}

Audit BOTH summaries against the ground truth, independently. Does either contain any values,
conditions, or medications NOT present in the ground truth?"""

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

    parsed = None
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            text = response.text.strip()
            parsed = {
                "clinical": _parse_side(text, "CLINICAL", clinical_summary),
                "patient": _parse_side(text, "PATIENT", patient_summary),
            }
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 35 * (attempt + 1)
                print(f"[VERIFIER] Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"[VERIFIER] Error: {e}")
                parsed = _unverified(
                    f"Verification could not run: {type(e).__name__}. "
                    f"This summary has NOT been audited — review manually."
                )
                break
    if parsed is None:
        parsed = _unverified(
            "Verification could not run (rate limit exhausted after retries). "
            "This summary has NOT been audited — review manually."
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
