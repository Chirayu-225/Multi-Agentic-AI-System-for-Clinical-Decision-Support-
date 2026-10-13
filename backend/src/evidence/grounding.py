"""
src/evidence/grounding.py

Two deterministic (no-LLM) checks that sit alongside the Verifier agent:

1. build_evidence()  - for every extracted lab value, diagnosis, medication and
   symptom, find the exact line/phrase in the SOURCE report it came from, so a
   clinician can click a claim in the summary and see the original text instead of
   trusting the model. `cited_in` records which summaries actually mention each item.

2. find_ungrounded_numbers() - every number in a generated summary must be traceable
   to the structured ground truth (lab values, flags, trends, age, dates). A number
   that appears nowhere is the classic LLM hallucination ("HbA1c of 8.4" when the
   report says 7.9). The Verifier is itself an LLM and can miss this; a set-membership
   check on numbers cannot be talked out of it.

Both are advisory: they add warnings for a human, they never block or rewrite output.
"""

import json
import re

_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
# "in 2 weeks", "3 follow-up actions", "after 6 months" - advice intervals the
# model legitimately introduces; not clinical measurements.
_INTERVAL_AFTER_RE = re.compile(
    r"^\s*[-–]?\s*(?:day|week|month|year|hour|minute|time|step|point|action|item|visit)s?\b", re.I
)
_SNIPPET_CONTEXT = 70


_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{2,3}(?:\D|$))")


def _strip_thousands(text: str) -> str:
    """'1,50,000' / '150,000' -> '150000' so a grouped number is one number, not several
    ('150' and '000' were being reported as ungrounded)."""
    prev = None
    while prev != text:
        prev, text = text, _THOUSANDS_RE.sub("", text)
    return text


def _norm_number(s: str) -> str:
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def _fmt_variants(value) -> list[str]:
    """Textual forms a numeric value may take in the source ('9.2', '9.20', '142')."""
    out = []
    try:
        f = float(value)
    except (TypeError, ValueError):
        return [str(value)]
    out.append(f"{f:g}")
    out.append(f"{f:.1f}")
    out.append(f"{f:.2f}")
    if f == int(f):
        out.append(str(int(f)))
    return list(dict.fromkeys(out))


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, (len(text) if end == -1 else end)


def _snippet(text: str, start: int, end: int, whole_line: bool) -> tuple[str, int, int]:
    """Return (snippet, match_start_in_snippet, match_end_in_snippet)."""
    if whole_line:
        ls, le = _line_bounds(text, start)
        raw = text[ls:le]
        lead = len(raw) - len(raw.lstrip())
        snip = raw.strip()
        return snip[:300], max(0, start - ls - lead), max(0, end - ls - lead)
    a = max(0, start - _SNIPPET_CONTEXT)
    b = min(len(text), end + _SNIPPET_CONTEXT)
    snip = text[a:b].replace("\n", " ")
    return snip, start - a, end - a


def _find_lab(raw_text: str, test: str, value) -> tuple[int, int] | None:
    """Locate a lab value: prefer the line that names the test AND contains the number."""
    low = raw_text.lower()
    t = test.lower().strip()
    values = _fmt_variants(value)
    search_from = 0
    while t and (idx := low.find(t, search_from)) != -1:
        ls, le = _line_bounds(raw_text, idx)
        line = raw_text[ls:le]
        for v in values:
            m = re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", line)
            if m:
                return ls + m.start(), ls + m.end()
        search_from = idx + len(t)
    # Test name unmatched (abbreviation differences): fall back to the number alone.
    for v in values:
        m = re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", raw_text)
        if m:
            return m.start(), m.end()
    return None


def build_evidence(raw_text: str, entities: dict, flags: list[dict],
                   summaries: dict[str, str] | None = None) -> list[dict]:
    """
    Link extracted facts to their source text.

    summaries: {"clinical": "...", "patient": "..."} - used only to fill `cited_in`.
    Each item: {id, kind, label, value?, snippet, match_start, match_end, found, cited_in}
    """
    raw_text = raw_text or ""
    summaries = summaries or {}
    evidence: list[dict] = []

    def _add(kind, label, value, span, whole_line, severity=None):
        item = {"id": f"ev{len(evidence) + 1}", "kind": kind, "label": label, "found": span is not None}
        if value is not None:
            item["value"] = value
        if severity:
            item["severity"] = severity
        if span:
            snip, ms, me = _snippet(raw_text, span[0], span[1], whole_line)
            item.update(snippet=snip, match_start=ms, match_end=me)
        else:
            item.update(snippet="", match_start=0, match_end=0)
        item["cited_in"] = [
            name for name, text in summaries.items()
            if text and (label.lower() in text.lower() or (value is not None and _mentions_number(text, value)))
        ]
        evidence.append(item)

    seen_labs = set()
    for f in flags:
        key = (str(f.get("test")), f.get("value"))
        if key in seen_labs or f.get("value") is None:
            continue
        seen_labs.add(key)
        span = _find_lab(raw_text, str(f.get("test", "")), f.get("value"))
        _add("lab", str(f.get("test", "")), f.get("value"), span, True, str(f.get("severity", "")).replace("Severity.", ""))

    def _phrase(kind, phrases):
        for p in phrases:
            if not p or not isinstance(p, str):
                continue
            m = re.search(re.escape(p), raw_text, re.I)
            _add(kind, p, None, (m.start(), m.end()) if m else None, False)

    _phrase("diagnosis", entities.get("clinical", {}).get("diagnoses", []))
    _phrase("medication", [m.get("name") for m in entities.get("medications", [])])
    _phrase("symptom", entities.get("clinical", {}).get("symptoms", []))
    return evidence


def _mentions_number(text: str, value) -> bool:
    nums = {_norm_number(n) for n in _NUMBER_RE.findall(text)}
    return _norm_number(str(value)) in nums


def collect_allowed_numbers(*structures, extra_texts: list[str] | None = None) -> set[str]:
    """Every number that appears anywhere in the structured ground truth."""
    blob_parts = []
    for s in structures:
        try:
            blob_parts.append(json.dumps(s, default=str))
        except Exception:
            blob_parts.append(str(s))
    blob_parts += [t for t in (extra_texts or []) if t]
    blob = _strip_thousands(" ".join(blob_parts))
    return {_norm_number(n) for n in re.findall(r"\d+(?:\.\d+)?", blob)}


def find_ungrounded_numbers(summary: str, allowed: set[str]) -> list[str]:
    """Numbers in `summary` that are not in `allowed`. Ignores list numbering
    ('1.' at line start) and advice intervals ('in 2 weeks')."""
    if not summary:
        return []
    cleaned = _strip_thousands(re.sub(r"(?m)^\s*\d+[.)]\s+", "", summary))
    bad: list[str] = []
    for m in _NUMBER_RE.finditer(cleaned):
        tail = cleaned[m.end(): m.end() + 14]
        if _INTERVAL_AFTER_RE.match(tail):
            continue
        n = _norm_number(m.group())
        if n not in allowed and m.group() not in bad:
            bad.append(m.group())
    return bad
