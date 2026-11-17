"""
src/agents/summarizer.py

Generates clinical and patient summaries using the pinned Gemini model (src/llm.py).
Replaces the local Phi-3 Mini inference from v1.
Called by the orchestrator after all tool calls are complete.
"""

import json
import logging
import re

from src.llm import generate_text

logger = logging.getLogger("medsight.summarizer")

from src.security.deidentification import redact



# ── Reading the model's answer ──────────────────────────────────────────────────────────────────
# The prompt asks for "SECTION: NAME" headers. Models do not all write them the same way (markdown
# headings, bold, "Clinical Summary:", a JSON object), so the reader accepts those forms. What it
# must never do is turn an answer it could not read into reassuring text: an empty section becomes
# "Summary unavailable", which the verifier refuses to mark as verified.

SECTION_NAMES = ("CLINICAL_SUMMARY", "PATIENT_SUMMARY", "FLAG_EXPLANATIONS", "FOLLOWUP_ACTIONS",
                 "LONGITUDINAL_NARRATIVE")
UNAVAILABLE_PREFIX = "Summary unavailable"
UNREADABLE_CLINICAL = (f"{UNAVAILABLE_PREFIX} - the AI's answer could not be read. "
                       "Review the flagged results in this report directly.")
UNREADABLE_PATIENT = f"{UNAVAILABLE_PREFIX} - please discuss these results with your doctor."
NO_ACTIONS = "No follow-up actions were generated. Decide follow-up from the flagged results."

_HEADER = re.compile(
    r"^[ \t>#*_`=\-]*(?:\d+[.)][ \t]*)?[*_`]*"                 # markdown or numbering around a header
    r"(?P<prefix>SECTION\b[ \t]*[:\-\u2013\u2014]?[ \t]*)?[*_`]*"
    r"(?P<name>CLINICAL[ _]SUMMARY|PATIENT[ _]SUMMARY|FLAG[ _]EXPLANATIONS?|"
    r"FOLLOW[ _\-]?UP[ _]ACTIONS?|LONGITUDINAL[ _]NARRATIVE)"
    r"[*_`]*[ \t]*(?P<sep>[:\-\u2013\u2014])?[ \t]*[*_`]*[ \t]*(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE)
_KEY_BY_START = (("CLIN", "CLINICAL_SUMMARY"), ("PAT", "PATIENT_SUMMARY"), ("FLAG", "FLAG_EXPLANATIONS"),
                 ("FOLLOW", "FOLLOWUP_ACTIONS"), ("LONG", "LONGITUDINAL_NARRATIVE"))


def is_placeholder(text: str | None) -> bool:
    """True for text that stands in for a summary that was never produced."""
    t = (text or "").strip()
    return not t or t.startswith(UNAVAILABLE_PREFIX)


def _section_key(name: str) -> str:
    n = name.upper()
    return next(key for start, key in _KEY_BY_START if n.startswith(start))


def _sections_from_json(text: str) -> dict[str, str] | None:
    t = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    if not t.startswith("{"):
        return None
    try:
        data = json.loads(t)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for raw_key, value in data.items():
        m = _HEADER.match(str(raw_key).replace("_", " "))
        if not m:
            continue
        if isinstance(value, dict):
            value = "\n".join(f"{k}: {v}" for k, v in value.items())
        elif isinstance(value, list):
            value = "\n".join(f"{i}. {v}" for i, v in enumerate(value, 1))
        out.setdefault(_section_key(m.group("name")), str(value or "").strip())
    return out or None


def split_sections(text: str) -> dict[str, str]:
    """The answer's sections by name. Missing or unreadable sections are simply absent."""
    text = text or ""
    from_json = _sections_from_json(text)
    if from_json is not None:
        return from_json
    headers = []
    for m in _HEADER.finditer(text):
        rest = re.sub(r"[*_`#]", "", m.group("rest")).strip()
        # A line is a header if it says SECTION, ends in a colon, or is nothing but the name.
        # That keeps a sentence which merely starts with "Patient summary of ..." in the body.
        if m.group("prefix") or m.group("sep") == ":" or not rest:
            headers.append((m, rest))
    out: dict[str, str] = {}
    for i, (m, rest) in enumerate(headers):
        end = headers[i + 1][0].start() if i + 1 < len(headers) else len(text)
        body = (rest + "\n" + text[m.end():end]).strip()
        body = re.sub(r"\n[ \t]*[-=*_]{3,}[ \t]*$", "", body).strip()      # a trailing markdown rule
        key = _section_key(m.group("name"))
        if body and not out.get(key):
            out[key] = body
    return out


# ── Follow-up actions are suggestions, not orders ───────────────────────────────────────────────
# The actions are not audited by the verifier and go straight to the clinician. A model once wrote
# "Admit the patient ... initiate transfusion" and "Start diabetic emergency protocol: IV insulin
# infusion" for a report whose only facts were two lab values. They are not rewritten here (that
# would hide what the model said); they are found, and the report carries a warning the clinician
# must acknowledge before release (SAFETY.md H18).

_ORDER_VERBS = {"admit", "start", "initiate", "begin", "commence", "administer", "give", "prescribe",
                "transfuse", "infuse", "inject", "stop", "discontinue", "titrate", "hospitalise",
                "hospitalize", "intubate"}
_LEADING_ADVERBS = {"urgently", "immediately", "promptly", "please", "now"}


def directive_actions(actions: list[str]) -> list[str]:
    """The actions that read as treatment orders: their first word (after an adverb such as
    "urgently") is a verb like admit, start, give or prescribe. "Consider starting ..." is fine."""
    out = []
    for a in actions or []:
        words = re.findall(r"[a-z]+", (a or "").lower())
        while words and words[0] in _LEADING_ADVERBS:
            words = words[1:]
        if words and words[0] in _ORDER_VERBS:
            out.append(a)
    return out


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
        if patient.get("name"):
            parts.append(redact(str(patient["name"]), redaction_map))
        # Prefer the normalised values (orchestrator: src/flagging/demographics.py).
        age = patient.get("age_years")
        if age is not None:
            parts.append(f"age {age:g}" if age >= 1 else f"age {round(age * 12)} months")
        elif patient.get("age"):
            parts.append(f"age {patient['age']}")
        sex = patient.get("sex") or patient.get("gender")
        if sex:
            parts.append(str(sex))
        if parts:
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


def _build_prompt(context: str, flag_lines: str, has_trends: bool) -> str:
    """The summarizer's whole prompt. Its source text is the prompt's version (src/prompts.py):
    change a word here and the version recorded on every model call changes with it."""
    return f"""You are a clinical decision support assistant for Indian healthcare.
Given the following verified patient data, produce ALL of the following outputs in ONE response.
Format your response EXACTLY as shown with these section headers.

PATIENT DATA:
{context}

ABNORMAL FLAGS TO EXPLAIN:
{flag_lines if flag_lines else "None"}

---
SECTION: CLINICAL_SUMMARY
Write a clinical summary for a GP in 2 to 4 sentences (150 words max). For every CRITICAL or HIGH
result, give its value, say whether it is low or high, and say how urgent it is (for example "needs
same-day review"). Only use values from the data above. Do not name a cause or a diagnosis that is
not in the data.

SECTION: PATIENT_SUMMARY
Explain the results to the patient in plain language (100 words max). Be honest and calm. Never say
the results are fine or normal when any result is flagged. If any result is CRITICAL, say clearly
that it needs attention and that they should contact their doctor today.

SECTION: FLAG_EXPLANATIONS
For each FLAG_N above, write one sentence explaining it with reference to the ICMR guideline and
what a clinician should consider next.
Format: FLAG_0: <explanation>

SECTION: FOLLOWUP_ACTIONS
List 3 numbered follow-up actions for the GP to consider, based on the findings. Write each one as
a suggestion that starts with "Consider" (for example "Consider same-day review of ..."). Do not
write orders to admit, start, give, prescribe or stop a treatment, and do not give drug doses: the
clinician decides treatment.

SECTION: LONGITUDINAL_NARRATIVE
{"Write a short (80 words max) narrative tracing this patient's trajectory across visits, "
 "using ONLY the 'Longitudinal trends' and 'Personal-baseline deviations' values listed "
 "above (cite the actual values and the visit-over-visit direction; if a value is flagged "
 "only as a personal-baseline deviation, note that it may still be within the general "
 "population's normal range). Do not speculate about a diagnosis or cause — describe the "
 "pattern only, and end with one sentence on why a clinician should look at the trend, not "
 "just the latest value." if has_trends else "Write exactly: N/A"}
"""


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

    prompt = _build_prompt(context, flag_lines, has_trends)

    unavailable = lambda why: {  # noqa: E731
        "clinical_summary": f"Summary unavailable{why}.",
        "patient_summary": f"Summary unavailable{why}.",
        "explained_flags": [{**f, "explanation": ""} for f in abnormal],
        "followup_actions": ["Consult a healthcare provider."],
        "longitudinal_narrative": "",
    }
    try:
        text = await generate_text(prompt, label="SUMMARIZER")
    except Exception as e:  # noqa: BLE001
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return unavailable(" — quota exceeded")
        print(f"[SUMMARIZER] Error: {type(e).__name__}")
        return unavailable("")

    sections = split_sections(text)
    clinical   = sections.get("CLINICAL_SUMMARY", "")
    patient    = sections.get("PATIENT_SUMMARY", "")
    flag_exps  = sections.get("FLAG_EXPLANATIONS", "")
    followup   = sections.get("FOLLOWUP_ACTIONS", "")
    narrative  = sections.get("LONGITUDINAL_NARRATIVE", "")
    if re.fullmatch(r"[\W_]*N/?A[\W_]*", narrative, re.IGNORECASE):
        narrative = ""
    if not clinical or not patient:
        # Sizes and section names only: the text itself can contain lab values.
        logger.warning("Summarizer answer was not fully readable: %d chars, sections found: %s",
                       len(text or ""), sorted(sections) or "none")

    # Parse flag explanations
    explained = []
    for i, f in enumerate(abnormal):
        m = re.search(rf"FLAG[ _]?{i}\b[*_`]*\s*[:\-\u2013]\s*[*_`]*\s*(.+)", flag_exps, re.IGNORECASE)
        explained.append({**f, "explanation": re.sub(r"\*\*|__", "", m.group(1)).strip() if m else ""})

    # Parse follow-up actions
    def _plain(t: str) -> str:
        # The UI shows its own numbering and renders text literally, so strip markdown
        # emphasis and any "1." / "-" prefix the model added.
        t = re.sub(r"\*\*|__", "", t)
        return re.sub(r"^\s*(?:\d+[.)]|[-*\u2022])\s*", "", t).strip()

    actions = [
        _plain(line)
        for line in followup.split("\n")
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith(("-", "*", "\u2022")))
    ]
    actions = [a for a in actions if a]
    if not actions:
        # Never invent a recommendation. "Routine monitoring" used to appear here, including under
        # a report whose flags were critical.
        actions = [_plain(followup)] if followup else [NO_ACTIONS]

    return {
        # An empty section is a failed answer, not a clean bill of health.
        "clinical_summary": clinical or UNREADABLE_CLINICAL,
        "patient_summary": patient or UNREADABLE_PATIENT,
        "explained_flags": explained,
        "followup_actions": actions,
        # Only surface a narrative when there's actually a trend to narrate —
        # showing "not enough history yet" on every single upload (including
        # a patient's very first report) would just be repetitive noise.
        "longitudinal_narrative": (narrative or "Trend detected across visits.") if has_trends else "",
    }
