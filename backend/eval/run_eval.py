"""
eval/run_eval.py - offline evaluation harness.

    python -m eval.run_eval            # run, print, write eval/results/latest.json + EVAL_RESULTS.md
    python -m eval.run_eval --check    # also fail (exit 1) if a metric regresses past eval/thresholds.json
    python -m eval.run_eval --live     # additionally measure the real Gemini Verifier (needs GEMINI_API_KEY)

What is measured, and against what:
  extraction     lab-value parser (regex path) recall/precision on synthetic reports in 4 layouts
  flagging       ICMR flagging accuracy given the CORRECT lab values (isolates flag logic + name mapping)
  deidentification  identifiers surviving redaction, with oracle entities vs regex-only
  injection      detection rate on an attack corpus, false-positive rate on benign clinical text
  grounding      does the numeric-grounding check catch corrupted summaries (and which kinds it cannot)
  interactions   brand->generic resolution and rule self-consistency

What is NOT measured offline: the GLiNER model's NER quality and the Gemini Verifier. Those need the
model weights / an API key; `--live` covers the Verifier. Numbers here are on SYNTHETIC data, so they
show the pipeline's logic works and where it breaks - they are not clinical validation.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.stubs import install_stubs  # noqa: E402  (stubs the heavy ML deps; see tests/conftest.py)

install_stubs()

from eval.synthetic import TESTS, generate_dataset  # noqa: E402
from src.evidence.grounding import collect_allowed_numbers, find_ungrounded_numbers  # noqa: E402
from src.flagging.risk_flagging import _canonical_name, flag_lab_values  # noqa: E402
from src.ner.medical_ner import _extract_lab_values_regex  # noqa: E402
from src.security.deidentification import build_redaction_map, redact, rehydrate  # noqa: E402
from src.security.llm_guardrails import scan_for_injection  # noqa: E402
from src.tools.drug_interactions import (DRUG_DRUG_RULES, CLASSES, check_drug_interactions,  # noqa: E402
                                         resolve_generics)
from tests.adversarial_corpus import ATTACKS, BENIGN, KNOWN_MISSES  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
N_CASES = int(os.getenv("EVAL_CASES", "200"))
SEED = int(os.getenv("EVAL_SEED", "42"))


def _pct(num, den):
    return round(num / den, 4) if den else None


# ── 1. Extraction ─────────────────────────────────────────────────────────────

def eval_extraction(cases):
    by_layout = {}
    tp = fp = fn = 0
    for c in cases:
        extracted = _extract_lab_values_regex(c.text)
        got = {}
        for e in extracted:
            got.setdefault(_canonical_name(e["test"]), []).append(float(e["value"]))
        want = {l.canonical: l.value for l in c.labs}
        stats = by_layout.setdefault(c.layout, {"tp": 0, "fn": 0, "fp": 0})
        for canon, val in want.items():
            if any(abs(v - val) < 1e-6 for v in got.get(canon, [])):
                stats["tp"] += 1; tp += 1
            else:
                stats["fn"] += 1; fn += 1
        for canon, vals in got.items():
            if canon not in want:
                stats["fp"] += len(vals); fp += len(vals)
    return {
        "recall": _pct(tp, tp + fn), "precision": _pct(tp, tp + fp),
        "by_layout_recall": {k: _pct(v["tp"], v["tp"] + v["fn"]) for k, v in sorted(by_layout.items())},
        "values_total": tp + fn,
    }


# ── 2. Flagging (oracle lab values) ───────────────────────────────────────────

def eval_flagging(cases):
    abn = abn_hit = crit = crit_hit = norm = norm_false = 0
    unmapped = []
    dir_ok = dir_total = 0
    canon_ok = canon_total = 0
    wrong_map = {}
    for c in cases:
        labs = [{"test": l.display, "value": l.value, "unit": l.unit} for l in c.labs]
        flags = flag_lab_values(labs, gender=c.gender)
        for l, f in zip(c.labs, flags):
            sev = str(f["severity"]).replace("Severity.", "")
            if f.get("normal_range") is None:
                unmapped.append(l.display)
            canon_total += 1
            if f.get("canonical") == l.canonical:
                canon_ok += 1
            else:
                wrong_map[f"{l.display} -> {f.get('canonical')} (expected {l.canonical})"] = True
            if l.band == "normal":
                norm += 1; norm_false += sev != "NORMAL"
            else:
                abn += 1; abn_hit += sev != "NORMAL"
                dir_total += 1; dir_ok += f.get("direction") == l.direction
            if l.band == "critical":
                crit += 1; crit_hit += sev == "CRITICAL"
    return {
        "abnormal_recall": _pct(abn_hit, abn), "critical_recall": _pct(crit_hit, crit),
        "false_alarm_rate_on_normal": _pct(norm_false, norm), "direction_accuracy": _pct(dir_ok, dir_total),
        "canonical_mapping_accuracy": _pct(canon_ok, canon_total),
        "misrouted_test_names": sorted(wrong_map),
        "unmapped_test_names": sorted(set(unmapped)),
        "values_total": abn + norm,
    }


# ── 3. De-identification ──────────────────────────────────────────────────────

def _leaks(case, redacted):
    low = redacted.lower()
    parts = case.name.split()
    return {
        "full_name": case.name.lower() in low, "name_uppercase": case.name.upper() in redacted,
        "last_first": f"{parts[1]}, {parts[0]}".lower() in low, "first_name_alone": re.search(rf"\b{parts[0]}\b", redacted, re.I) is not None,
        "surname_alone": re.search(rf"\b{parts[1]}\b", redacted, re.I) is not None,
        "phone": case.phone in redacted, "email": case.email in redacted,
        "mrn": case.mrn in redacted, "aadhaar": case.aadhaar in redacted,
        "doctor": case.doctor.lower() in low, "hospital": case.hospital.lower() in low,
    }


def eval_deid(cases):
    modes = {}
    for mode in ("oracle_entities", "regex_only"):
        counts, roundtrip_ok = {}, 0
        for c in cases:
            ents = ({"patient": {"name": c.name}, "providers": {"doctor": c.doctor, "hospital": c.hospital}}
                    if mode == "oracle_entities" else {})
            m = build_redaction_map(ents, c.text)
            red = redact(c.text, m)
            for k, v in _leaks(c, red).items():
                counts[k] = counts.get(k, 0) + int(v)
            roundtrip_ok += int(rehydrate(red, m).count("[REDACTED") == 0)
        modes[mode] = {"leak_rate_by_identifier": {k: _pct(v, len(cases)) for k, v in counts.items()},
                       "cases_with_any_leak": None, "rehydration_clean": _pct(roundtrip_ok, len(cases))}
        any_leak = 0
        for c in cases:
            ents = ({"patient": {"name": c.name}, "providers": {"doctor": c.doctor, "hospital": c.hospital}}
                    if mode == "oracle_entities" else {})
            red = redact(c.text, build_redaction_map(ents, c.text))
            any_leak += any(_leaks(c, red).values())
        modes[mode]["cases_with_any_leak"] = _pct(any_leak, len(cases))
    return modes


# ── 4. Injection ──────────────────────────────────────────────────────────────

def eval_injection(cases):
    hit = sum(bool(scan_for_injection(t)) for t in ATTACKS.values())
    fp_benign = [k for k, t in BENIGN.items() if scan_for_injection(t)]
    fp_reports = sum(bool(scan_for_injection(c.text)) for c in cases)
    return {
        "attack_detection_rate": _pct(hit, len(ATTACKS)), "attacks_total": len(ATTACKS),
        "missed_attacks": [k for k, t in ATTACKS.items() if not scan_for_injection(t)],
        "benign_false_positive_rate": _pct(len(fp_benign), len(BENIGN)), "benign_false_positives": fp_benign,
        "synthetic_report_false_positive_rate": _pct(fp_reports, len(cases)),
        "known_evasions_still_undetected": sum(not scan_for_injection(t) for t in KNOWN_MISSES.values()),
        "known_evasions_total": len(KNOWN_MISSES),
    }


# ── 5. Numeric grounding vs corrupted summaries ───────────────────────────────

def _num(v):
    """Plain decimal text (never scientific notation, which no clinician-facing summary uses)."""
    return str(int(v)) if float(v) == int(v) else f"{v:.4f}".rstrip("0").rstrip(".")


def _truth_summary(c):
    lines = [f"{l.display} is {_num(l.value)} {l.unit}." for l in c.labs]
    return "Clinical summary: " + " ".join(lines)


def eval_grounding(cases):
    rng = random.Random(SEED + 1)
    kinds = {"novel_value": [0, 0], "fabricated_lab": [0, 0], "value_swap": [0, 0], "wrong_direction": [0, 0]}
    clean_fp = 0
    for c in cases:
        flags = flag_lab_values([{"test": l.display, "value": l.value, "unit": l.unit} for l in c.labs], c.gender)
        allowed = collect_allowed_numbers({"labs": [{"v": l.value} for l in c.labs]}, flags,
                                          {"age": "x"}, extra_texts=[c.text])
        # NOTE: `allowed` is built from the SAME structured data main.py uses (flags + entities),
        # not from the summary under test.
        allowed = collect_allowed_numbers({"labs": [{"test": l.display, "value": l.value, "unit": l.unit} for l in c.labs]}, flags)
        clean = _truth_summary(c)
        clean_fp += bool(find_ungrounded_numbers(clean, allowed))

        l = rng.choice(c.labs)
        new = round(l.value * 1.37 + 0.37, 2)
        bad = clean.replace(f"{l.display} is {_num(l.value)}", f"{l.display} is {_num(new)}")
        kinds["novel_value"][1] += 1; kinds["novel_value"][0] += bool(find_ungrounded_numbers(bad, allowed))

        bad = clean + " LDL cholesterol is 188 mg/dL."
        kinds["fabricated_lab"][1] += 1; kinds["fabricated_lab"][0] += bool(find_ungrounded_numbers(bad, allowed))

        if len(c.labs) >= 2 and c.labs[0].value != c.labs[1].value:
            a, b = c.labs[0], c.labs[1]
            bad = clean.replace(f"{a.display} is {_num(a.value)}", "@@A").replace(f"{b.display} is {_num(b.value)}", f"{b.display} is {_num(a.value)}") \
                       .replace("@@A", f"{a.display} is {_num(b.value)}")
            kinds["value_swap"][1] += 1; kinds["value_swap"][0] += bool(find_ungrounded_numbers(bad, allowed))

        sev = [x for x in c.labs if x.direction in ("high", "low")]
        if sev:
            x = sev[0]
            word = "high" if x.direction == "low" else "low"
            bad = clean + f" {x.display} is {word}."
            kinds["wrong_direction"][1] += 1; kinds["wrong_direction"][0] += bool(find_ungrounded_numbers(bad, allowed))
    return {
        "catch_rate_by_corruption": {k: _pct(v[0], v[1]) for k, v in kinds.items()},
        "clean_summary_false_positive_rate": _pct(clean_fp, len(cases)),
        "note": "value_swap and wrong_direction use only numbers/words already in the ground truth, so a "
                "numeric check cannot see them - that is the job of the LLM Verifier (measure with --live).",
    }


# ── 6. Interactions ───────────────────────────────────────────────────────────

BRAND_CASES = [
    ("Tab. Glycomet 500mg", "metformin"), ("Glyciphage SR", "metformin"), ("AMARYL 2", "glimepiride"),
    ("Norvasc 5 mg OD", "amlodipine"), ("Ecosprin 75", "aspirin"), ("Lipitor 10", "atorvastatin"),
    ("Plavix", "clopidogrel"), ("Coumadin 5mg", "warfarin"), ("Voveran SR", "diclofenac"),
    ("Thyronorm 50 mcg", "levothyroxine"), ("Telma 40", "telmisartan"), ("Aldactone 25", "spironolactone"),
    ("Pantocid", "pantoprazole"), ("Lasix 40", "furosemide"), ("Lanoxin", "digoxin"),
]


def eval_interactions():
    ok = sum(expected in resolve_generics(name) for name, expected in BRAND_CASES)
    fired = 0
    for a, b, *_ in DRUG_DRUG_RULES:
        pick = lambda side: sorted(CLASSES[side[6:]])[0] if side.startswith("class:") else side
        meds = [{"name": pick(a)}, {"name": pick(b)}]
        fired += any(i["type"] == "drug-drug" for i in check_drug_interactions(meds)["interactions"])
    return {"brand_resolution_accuracy": _pct(ok, len(BRAND_CASES)),
            "rule_self_consistency": _pct(fired, len(DRUG_DRUG_RULES)), "rules_total": len(DRUG_DRUG_RULES)}


# ── 7. Live verifier (optional) ───────────────────────────────────────────────

def eval_live_verifier(cases, limit=20):
    """Run the REAL Verifier agent on corrupted summaries. Needs GEMINI_API_KEY and network;
    spends `limit` x 2 Gemini calls. Not part of the default/CI run."""
    from src.agents.verifier import verify_and_correct
    rng = random.Random(SEED + 2)
    caught = {"novel_value": [0, 0], "value_swap": [0, 0], "wrong_direction": [0, 0], "clean": [0, 0]}

    async def one(c, kind):
        flags = flag_lab_values([{"test": l.display, "value": l.value, "unit": l.unit} for l in c.labs], c.gender)
        clean = _truth_summary(c)
        summary = clean
        if kind == "novel_value":
            l = rng.choice(c.labs); summary = clean.replace(f"{l.display} is {_num(l.value)}", f"{l.display} is {l.value * 1.4 + 0.4:.2f}")
        elif kind == "value_swap" and len(c.labs) > 1 and c.labs[0].value != c.labs[1].value:
            a, b = c.labs[0], c.labs[1]
            summary = clean.replace(f"{a.display} is {_num(a.value)}", "@@").replace(f"{b.display} is {_num(b.value)}", f"{b.display} is {_num(a.value)}").replace("@@", f"{a.display} is {_num(b.value)}")
        elif kind == "wrong_direction":
            x = next((x for x in c.labs if x.direction in ("high", "low")), None)
            if not x: return
            summary = clean + f" {x.display} is {'high' if x.direction == 'low' else 'low'}."
        ents = {"patient": {}, "clinical": {"diagnoses": [c.diagnosis]}, "medications": [{"name": m} for m in c.meds]}
        out = await verify_and_correct(summary, "Your results were reviewed.", ents, flags, [])
        flagged = out["clinical_corrected"] or out["clinical_unverified"]
        caught[kind][1] += 1
        caught[kind][0] += int(flagged)

    async def main():
        for c in cases[:limit]:
            for kind in caught:
                await one(c, kind)
    asyncio.run(main())
    return {k: _pct(v[0], v[1]) for k, v in caught.items()} | {"n_per_kind": limit,
            "note": "'clean' is the false-alarm rate (lower is better); the others are catch rates."}


# ── Runner ────────────────────────────────────────────────────────────────────

def run(live: bool = False) -> dict:
    t0 = time.time()
    cases = generate_dataset(N_CASES, SEED)
    res = {
        "meta": {"cases": N_CASES, "seed": SEED, "data": "synthetic (seeded), no real patient data",
                 "live_verifier": live},
        "extraction": eval_extraction(cases),
        "flagging": eval_flagging(cases),
        "deidentification": eval_deid(cases),
        "injection": eval_injection(cases),
        "grounding": eval_grounding(cases),
        "interactions": eval_interactions(),
    }
    if live:
        if not os.getenv("GEMINI_API_KEY"):
            raise SystemExit("--live needs GEMINI_API_KEY")
        res["live_verifier"] = eval_live_verifier(cases)
    res["meta"]["seconds"] = round(time.time() - t0, 2)
    return res


def to_markdown(r: dict) -> str:
    pc = lambda x: "n/a" if x is None else f"{x * 100:.1f}%"
    e, f, d, i, g, x = r["extraction"], r["flagging"], r["deidentification"], r["injection"], r["grounding"], r["interactions"]
    L = [f"# Evaluation results", "",
         f"Synthetic data, {r['meta']['cases']} reports, seed {r['meta']['seed']}. Generated by `python -m eval.run_eval`. "
         "These numbers show where the pipeline's logic holds and where it breaks on controlled inputs; "
         "they are not clinical validation.", "",
         "## Lab-value extraction (regex path)", "",
         f"Recall {pc(e['recall'])}, precision {pc(e['precision'])} over {e['values_total']} values.", "",
         "| Layout | Recall |", "|---|---|"]
    L += [f"| {k} | {pc(v)} |" for k, v in e["by_layout_recall"].items()]
    L += ["", "## ICMR flagging (given correct values)", "",
          "| Metric | Result |", "|---|---|",
          f"| Abnormal values flagged | {pc(f['abnormal_recall'])} |",
          f"| Critical values graded CRITICAL | {pc(f['critical_recall'])} |",
          f"| False alarms on normal values | {pc(f['false_alarm_rate_on_normal'])} |",
          f"| Direction (high/low) correct | {pc(f['direction_accuracy'])} |",
          f"| Test name mapped to the right reference range | {pc(f['canonical_mapping_accuracy'])} |",
          f"| Mis-routed names | {'; '.join(f['misrouted_test_names']) or 'none'} |",
          f"| Test names with no reference range | {', '.join(f['unmapped_test_names']) or 'none'} |", "",
          "## De-identification: identifiers surviving redaction", "",
          "| Identifier | With NER entities (oracle) | Regex only (NER missed) |", "|---|---|---|"]
    keys = list(d["oracle_entities"]["leak_rate_by_identifier"])
    for k in keys:
        L.append(f"| {k} | {pc(d['oracle_entities']['leak_rate_by_identifier'][k])} | {pc(d['regex_only']['leak_rate_by_identifier'][k])} |")
    L += [f"| **reports with any leak** | **{pc(d['oracle_entities']['cases_with_any_leak'])}** | **{pc(d['regex_only']['cases_with_any_leak'])}** |", "",
          "Lower is better. The right column is the honest worst case: names and facilities depend on the NER model finding them.", "",
          "## Prompt-injection detection", "",
          f"- Attack corpus: {pc(i['attack_detection_rate'])} detected ({i['attacks_total']} payloads). Missed: {', '.join(i['missed_attacks']) or 'none'}",
          f"- False positives on benign clinical text: {pc(i['benign_false_positive_rate'])} ({', '.join(i['benign_false_positives']) or 'none'})",
          f"- False positives on synthetic reports: {pc(i['synthetic_report_false_positive_rate'])}",
          f"- Documented evasions still undetected: {i['known_evasions_still_undetected']}/{i['known_evasions_total']} (base64, leetspeak, paraphrase, split fields)", "",
          "## Numeric grounding check vs corrupted summaries", "",
          "| Corruption | Caught |", "|---|---|"]
    L += [f"| {k} | {pc(v)} |" for k, v in g["catch_rate_by_corruption"].items()]
    L += [f"| _clean summary flagged (false alarm)_ | {pc(g['clean_summary_false_positive_rate'])} |", "", g["note"], "",
          "## Medication rules", "",
          f"- Brand to generic resolution: {pc(x['brand_resolution_accuracy'])}",
          f"- Rule self-consistency: {pc(x['rule_self_consistency'])} of {x['rules_total']} rules fire on a representative pair"]
    if "live_verifier" in r:
        lv = r["live_verifier"]
        L += ["", "## Live LLM Verifier (Gemini)", "", "| Corruption | Flagged |", "|---|---|"]
        L += [f"| {k} | {pc(v)} |" for k, v in lv.items() if k not in ("n_per_kind", "note")]
        L += ["", lv["note"]]
    else:
        L += ["", "_Live Gemini Verifier not measured in this run (`--live` needs an API key)._"]
    return "\n".join(L) + "\n"


def check(r: dict) -> list[str]:
    thr = json.loads((ROOT / "eval" / "thresholds.json").read_text())
    def get(path):
        cur = r
        for p in path.split("."):
            cur = cur[p]
        return cur
    failures = []
    for path, rule in thr["min"].items():
        v = get(path)
        if v is None or v < rule:
            failures.append(f"{path} = {v} < required {rule}")
    for path, rule in thr["max"].items():
        v = get(path)
        if v is None or v > rule:
            failures.append(f"{path} = {v} > allowed {rule}")
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    result = run(live=a.live)
    md = to_markdown(result)
    print(md)
    if not a.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "latest.json").write_text(json.dumps(result, indent=2))
        (ROOT.parent / "EVAL_RESULTS.md").write_text(md)
    if a.check:
        bad = check(result)
        if bad:
            print("\nEVAL REGRESSION:\n  " + "\n  ".join(bad))
            sys.exit(1)
        print("\nAll evaluation thresholds met.")
