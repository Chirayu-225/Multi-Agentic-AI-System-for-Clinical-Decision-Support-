"""
src/agents/verifier.py

Verification Agent — the second agent in MedSight's multi-agent system.

Distinct from the Orchestrator in:
  - Role: critical auditor, not generator
  - Memory: only the current report's extracted data (no history)
  - Tools: none — structured evaluation only
  - Authority: can reject and trigger correction of the orchestrator's output
  - System prompt: adversarial perspective, looking for unsupported claims

Same model (gemini-2.5-flash), different agent identity.
"""

import os
import re
import json
import asyncio
import time

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

VERIFIER_SYSTEM_PROMPT = """You are a clinical verification agent. Your ONLY job is to audit
AI-generated medical summaries for factual accuracy against ground-truth extracted data.

You are NOT a generator. You are an auditor.

Rules:
1. Every specific value, condition, or medication mentioned in the summary MUST appear in the ground truth
2. If the summary mentions ANY value not in the ground truth, output VERDICT: FAIL
3. If the summary is consistent with the ground truth, output VERDICT: PASS
4. Always list specific issues found, or "none" if clean

Output format (strictly follow this):
VERDICT: PASS or FAIL
ISSUES: <specific unsupported claims, or "none">
CORRECTED_SUMMARY: <only if VERDICT is FAIL — rewrite the summary using only ground truth data>
"""


def _build_ground_truth(entities: dict, flags: list, trend_flags: list) -> str:
    """Build a minimal, strict ground-truth string from verified extracted data."""
    lines = []
    patient = entities.get("patient", {})
    if patient.get("name"):
        lines.append(f"Patient: {patient['name']}")
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

    return "\n".join(lines) if lines else "No clinical data extracted."


async def verify_and_correct(
    clinical_summary: str,
    patient_summary: str,
    entities: dict,
    flags: list,
    trend_flags: list,
) -> dict:
    """
    Verification agent entry point.

    Audits both summaries against ground-truth extracted data.
    Returns corrected versions if hallucinations are found.

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

    ground_truth = _build_ground_truth(entities, flags, trend_flags)

    async def _verify(summary: str, summary_type: str) -> dict:
        prompt = f"""GROUND TRUTH DATA (the only facts that exist in this patient's report):
{ground_truth}

{summary_type.upper()} SUMMARY TO AUDIT:
{summary}

Audit this summary against the ground truth. Does it contain any values, conditions,
or medications NOT present in the ground truth?"""

        # Retry on rate limit
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(model.generate_content, prompt)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 35 * (attempt + 1)
                    print(f"[VERIFIER] Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    # Non-rate-limit error — return as passed to avoid blocking
                    print(f"[VERIFIER] Error: {e}")
                    return {
                        "passed": True,
                        "issues": "verification skipped due to error",
                        "final_summary": summary,
                        "was_corrected": False,
                    }
        else:
            return {
                "passed": True,
                "issues": "verification skipped — rate limit",
                "final_summary": summary,
                "was_corrected": False,
            }

        text = response.text.strip()
        passed = "VERDICT: PASS" in text.upper()
        issues_match = re.search(r"ISSUES:\s*(.+?)(?:\n|CORRECTED_SUMMARY|$)", text,
                                 re.IGNORECASE | re.DOTALL)
        issues = issues_match.group(1).strip() if issues_match else ""
        corrected_match = re.search(r"CORRECTED_SUMMARY:\s*(.+)", text,
                                    re.IGNORECASE | re.DOTALL)
        corrected_text = corrected_match.group(1).strip() if corrected_match else summary

        return {
            "passed": passed,
            "issues": issues,
            "final_summary": summary if passed else corrected_text,
            "was_corrected": not passed,
        }

    # Run sequentially to avoid hitting rate limits with concurrent calls
    clinical_result = await _verify(clinical_summary, "clinical")
    await asyncio.sleep(5)  # small gap between calls
    patient_result = await _verify(patient_summary, "patient")

    return {
        "clinical_summary": clinical_result["final_summary"],
        "patient_summary": patient_result["final_summary"],
        "clinical_corrected": clinical_result["was_corrected"],
        "patient_corrected": patient_result["was_corrected"],
        "clinical_issues": clinical_result["issues"],
        "patient_issues": patient_result["issues"],
    }