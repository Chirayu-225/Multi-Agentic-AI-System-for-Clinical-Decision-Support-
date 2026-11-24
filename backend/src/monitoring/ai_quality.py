"""
src/monitoring/ai_quality.py

The clinician-edit feedback loop. A clinician approving, editing or rejecting an AI summary is the
only ground truth this system will ever get for free, so it is measured instead of discarded:

  * how often summaries are approved untouched, edited, or rejected;
  * how much they are edited (word-level change, so a typo fix and a rewrite are not the same);
  * whether the Verifier's verdict predicts the clinician's (edits on summaries the Verifier passed
    are the Verifier's misses);
  * all of the above per prompt version, so a prompt change is judged by what clinicians then do.

Pure functions over plain dicts: no database, easy to test. Aggregates only - the summaries
themselves never leave the report row.
"""
import difflib
from statistics import median


def word_diff(before: str, after: str) -> tuple[float, int]:
    """(change ratio 0..1, words inserted+deleted). 0 = identical, 1 = nothing in common."""
    a, b = (before or "").split(), (after or "").split()
    if not a and not b:
        return 0.0, 0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changed = sum((i2 - i1) + (j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return round(1.0 - sm.ratio(), 4), changed


def edit_stats(pairs: list[tuple[str, str]]) -> tuple[float, int]:
    """Combined change over several (ai_text, final_text) pairs: texts are joined first, so one
    long rewrite outweighs one short tweak."""
    return word_diff(" ".join(p[0] or "" for p in pairs), " ".join(p[1] or "" for p in pairs))


def _rate(n: int, d: int):
    return round(n / d, 4) if d else None


def _block(records: list[dict], min_n: int) -> dict:
    n = len(records)
    approved = [r for r in records if r["status"] == "approved"]
    rejected = [r for r in records if r["status"] == "rejected"]
    edited = [r for r in approved if r.get("edited")]
    ratios = [r["edit_ratio"] for r in edited if r.get("edit_ratio") is not None]
    return {
        "reviewed": n, "approved": len(approved), "rejected": len(rejected),
        "approved_unedited": len(approved) - len(edited), "approved_edited": len(edited),
        "unedited_rate": _rate(len(approved) - len(edited), n),
        "edit_rate": _rate(len(edited), len(approved)),
        "rejection_rate": _rate(len(rejected), n),
        "median_edit_ratio": round(median(ratios), 4) if ratios else None,
        "enough_data": n >= min_n,
    }


def quality_summary(records: list[dict], min_n: int = 20) -> dict:
    """records: {status: approved|rejected, edited, edit_ratio, verifier_corrected, verifier_unverified,
    summarizer_version, verifier_version, model}."""
    out = _block(records, min_n)
    passed = [r for r in records if r["status"] == "approved" and not r.get("verifier_corrected")
              and not r.get("verifier_unverified")]
    out["verifier"] = {
        "corrected_rate": _rate(sum(1 for r in records if r.get("verifier_corrected")), len(records)),
        "unverified_rate": _rate(sum(1 for r in records if r.get("verifier_unverified")), len(records)),
        # approved reports the Verifier passed that a clinician then edited: the Verifier missed something
        "missed_by_verifier_rate": _rate(sum(1 for r in passed if r.get("edited")), len(passed)),
    }
    for key, label in (("summarizer_version", "by_summarizer_version"), ("verifier_version", "by_verifier_version"),
                       ("model", "by_model")):
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(r.get(key) or "unknown", []).append(r)
        out[label] = {k: _block(v, min_n) for k, v in sorted(groups.items())}
    out["note"] = (f"Rates from fewer than {min_n} reviews are marked enough_data=false: read them as "
                   "anecdotes, not as evidence that one prompt is better than another.")
    return out
