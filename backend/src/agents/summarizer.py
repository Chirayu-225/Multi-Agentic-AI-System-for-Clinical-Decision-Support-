"""
src/agents/summarizer.py

Generates clinical and patient summaries using Gemini 2.5 Flash.
Replaces the local Phi-3 Mini inference from v1.
Called by the orchestrator after all tool calls are complete.
"""

import os
import re
import asyncio

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _build_context(entities: dict, flags: list, trend_flags: list,
                   pubmed_refs: list, drug_safety: list) -> str:
    lines = []
    patient = entities.get("patient", {})
    if any(patient.values()):
        parts = [v for v in patient.values() if v]
        lines.append(f"Patient: {', '.join(parts)}")

    diagnoses = entities.get("clinical", {}).get("diagnoses", [])
    if diagnoses:
        lines.append(f"Diagnoses: {', '.join(diagnoses)}")

    meds = entities.get("medications", [])
    if meds:
        lines.append(f"Medications: {', '.join(m['name'] for m in meds)}")

    abnormal = [f for f in flags if f.get("severity") not in ("NORMAL", "Severity.NORMAL")]
    if abnormal:
        lines.append("Abnormal lab values:")
        for f in abnormal:
            lines.append(f"  [{f['severity']}] {f['test']}: {f['value']} {f.get('unit', '')} — {f.get('message', '')}")

    if trend_flags:
        lines.append("Longitudinal trends:")
        for t in trend_flags:
            lines.append(f"  [{t['severity']}] {t['test']}: {t.get('message', '')}")

    if pubmed_refs:
        lines.append("Clinical evidence found:")
        for ref in pubmed_refs[:2]:
            if "title" in ref:
                lines.append(f"  - {ref['title']} ({ref.get('url', '')})")

    if drug_safety:
        for ds in drug_safety:
            if ds.get("has_warnings"):
                lines.append(f"Drug safety alert — {ds['medication']}: {ds.get('boxed_warning', '')[:200]}")

    return "\n".join(lines)


async def generate_summaries(
    entities: dict,
    flags: list,
    trend_flags: list,
    pubmed_refs: list = None,
    drug_safety: list = None,
) -> dict:
    """
    Generate all summaries in a SINGLE Gemini API call to preserve quota.
    Returns: { clinical_summary, patient_summary, explained_flags, followup_actions }
    """
    model = genai.GenerativeModel(model_name=MODEL)
    context = _build_context(
        entities, flags, trend_flags,
        pubmed_refs or [], drug_safety or []
    )
    abnormal = [f for f in flags if f.get("severity") not in ("NORMAL", "Severity.NORMAL")]

    # Build flag explanation requests inline
    flag_lines = ""
    for i, f in enumerate(abnormal):
        flag_lines += f"\nFLAG_{i}: {f.get('message', '')} (source: {f.get('source', 'ICMR')})"

    prompt = f"""You are a clinical decision support assistant for Indian healthcare.
Given the following verified patient data, produce ALL of the following outputs in ONE response.
Format your response EXACTLY as shown with these section headers.

PATIENT DATA:
{context}

ABNORMAL FLAGS TO EXPLAIN:
{flag_lines if flag_lines else "None"}

---
SECTION: CLINICAL_SUMMARY
Write a concise clinical summary (150 words max) for a GP. Only use values from the data above.

SECTION: PATIENT_SUMMARY
Explain results to the patient in plain language (100 words max). Simple, honest, not alarming.

SECTION: FLAG_EXPLANATIONS
For each FLAG_N above, write one sentence explanation referencing the ICMR guideline and recommended action.
Format: FLAG_0: <explanation>

SECTION: FOLLOWUP_ACTIONS
List 3 specific numbered follow-up actions for the GP based on the findings.
"""

    # Retry on rate limit
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 35 * (attempt + 1)
                print(f"[SUMMARIZER] Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"[SUMMARIZER] Error: {e}")
                return {
                    "clinical_summary": "Summary unavailable.",
                    "patient_summary": "Summary unavailable.",
                    "explained_flags": [{**f, "explanation": ""} for f in abnormal],
                    "followup_actions": ["Consult a healthcare provider."],
                }
    else:
        return {
            "clinical_summary": "Summary unavailable — quota exceeded.",
            "patient_summary": "Summary unavailable — quota exceeded.",
            "explained_flags": [{**f, "explanation": ""} for f in abnormal],
            "followup_actions": ["Consult a healthcare provider."],
        }

    text = response.text.strip()

    def _extract_section(section_name: str) -> str:
        pattern = rf"SECTION: {section_name}\n(.*?)(?=SECTION:|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    clinical  = _extract_section("CLINICAL_SUMMARY")
    patient   = _extract_section("PATIENT_SUMMARY")
    flag_exps = _extract_section("FLAG_EXPLANATIONS")
    followup  = _extract_section("FOLLOWUP_ACTIONS")

    # Parse flag explanations
    explained = []
    for i, f in enumerate(abnormal):
        m = re.search(rf"FLAG_{i}:\s*(.+)", flag_exps)
        explained.append({**f, "explanation": m.group(1).strip() if m else ""})

    # Parse follow-up actions
    actions = [
        line.strip()
        for line in followup.split("\n")
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))
    ]
    if not actions:
        actions = [followup] if followup else ["Routine monitoring recommended."]

    return {
        "clinical_summary": clinical or "No significant findings to summarize.",
        "patient_summary": patient or "Your results have been reviewed.",
        "explained_flags": explained,
        "followup_actions": actions,
    }