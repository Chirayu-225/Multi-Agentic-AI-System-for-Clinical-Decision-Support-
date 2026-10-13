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

from src.security.deidentification import redact

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _build_context(entities: dict, flags: list, trend_flags: list,
                   pubmed_refs: list, drug_safety: list,
                   baseline_flags: list | None = None,
                   redaction_map: dict | None = None,
                   drug_interactions: list | None = None) -> str:
    lines = []
    patient = entities.get("patient", {})
    if any(patient.values()):
        # Patient name is a direct identifier — redact it before it reaches
        # this function's caller's Gemini prompt (LLM02:2026, see
        # deidentification.py). Age/gender are kept as-is: they're clinical
        # context the summary genuinely needs, not identifiers on their own.
        redaction_map = redaction_map or {}
        parts = []
        for key, value in patient.items():
            if not value:
                continue
            parts.append(redact(value, redaction_map) if key == "name" else value)
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

    if baseline_flags:
        lines.append("Personal-baseline deviations (vs. this patient's own history):")
        for b in baseline_flags:
            lines.append(f"  [{b['severity']}] {b['test']}: {b.get('message', '')}")

    if pubmed_refs:
        lines.append("Clinical evidence found:")
        for ref in pubmed_refs[:2]:
            if "title" in ref:
                lines.append(f"  - {ref['title']} ({ref.get('url', '')})")

    if drug_safety:
        for ds in drug_safety:
            if ds.get("has_warnings"):
                # boxed_warning/warnings_snippet can each be explicitly None
                # (not just absent) when OpenFDA didn't return one — `dict.get`
                # only falls back to its default for a MISSING key, not a
                # present key whose value is None, so `or ""` is required
                # here (a bare `.get(..., "")[:200]` crashes on None).
                detail = ds.get("boxed_warning") or ds.get("warnings_snippet") or ""
                lines.append(f"Drug safety alert — {ds['medication']}: {detail[:200]}")

    if drug_interactions:
        lines.append("Medication interaction alerts (curated rules - state them as flagged, do not extend them):")
        for it in drug_interactions[:5]:
            lines.append(f"  [{it['severity']}] {' + '.join(it['drugs'])}: {it['effect']}")

    return "\n".join(lines)


async def generate_summaries(
    entities: dict,
    flags: list,
    trend_flags: list,
    pubmed_refs: list = None,
    drug_safety: list = None,
    baseline_flags: list = None,
    redaction_map: dict | None = None,
    drug_interactions: list | None = None,
) -> dict:
    """
    Generate all summaries in a SINGLE Gemini API call to preserve quota.
    Returns: { clinical_summary, patient_summary, explained_flags, followup_actions,
               longitudinal_narrative }

    IMPORTANT: the returned text still contains redaction placeholders (e.g.
    "[REDACTED_NAME_1]") wherever the patient's name would appear — see
    deidentification.py. The caller (main.py) is responsible for calling
    rehydrate() on every field here exactly once, after the Verifier has
    also run on this same (still-redacted) text. Do NOT rehydrate here.
    """
    baseline_flags = baseline_flags or []
    model = genai.GenerativeModel(model_name=MODEL)
    context = _build_context(
        entities, flags, trend_flags,
        pubmed_refs or [], drug_safety or [],
        baseline_flags, redaction_map, drug_interactions,
    )
    abnormal = [f for f in flags if f.get("severity") not in ("NORMAL", "Severity.NORMAL")]

    # Build flag explanation requests inline
    flag_lines = ""
    for i, f in enumerate(abnormal):
        flag_lines += f"\nFLAG_{i}: {f.get('message', '')} (source: {f.get('source', 'ICMR')})"

    has_trends = bool(trend_flags) or bool(baseline_flags)

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

SECTION: LONGITUDINAL_NARRATIVE
{"Write a short (80 words max) narrative tracing this patient's trajectory across visits, "
 "using ONLY the 'Longitudinal trends' and 'Personal-baseline deviations' values listed "
 "above (cite the actual values and the visit-over-visit direction; if a value is flagged "
 "only as a personal-baseline deviation, note that it may still be within the general "
 "population's normal range). Do not speculate about a diagnosis or cause — describe the "
 "pattern only, and end with one sentence on why a clinician should look at the trend, not "
 "just the latest value." if has_trends else "Write exactly: N/A"}
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
                    "longitudinal_narrative": "",
                }
    else:
        return {
            "clinical_summary": "Summary unavailable — quota exceeded.",
            "patient_summary": "Summary unavailable — quota exceeded.",
            "explained_flags": [{**f, "explanation": ""} for f in abnormal],
            "followup_actions": ["Consult a healthcare provider."],
            "longitudinal_narrative": "",
        }

    text = response.text.strip()

    def _extract_section(section_name: str) -> str:
        pattern = rf"SECTION: {section_name}\n(.*?)(?=SECTION:|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    clinical   = _extract_section("CLINICAL_SUMMARY")
    patient    = _extract_section("PATIENT_SUMMARY")
    flag_exps  = _extract_section("FLAG_EXPLANATIONS")
    followup   = _extract_section("FOLLOWUP_ACTIONS")
    narrative  = _extract_section("LONGITUDINAL_NARRATIVE")

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
        # Only surface a narrative when there's actually a trend to narrate —
        # showing "not enough history yet" on every single upload (including
        # a patient's very first report) would just be repetitive noise.
        "longitudinal_narrative": (narrative or "Trend detected across visits.") if has_trends else "",
    }
