"""
eval/run_gold.py - evaluation on REAL lab reports (a "gold set"), not synthetic ones.

The synthetic harness (run_eval.py) scores the parser on reports produced by our own
generator, so 100% there only proves the parser reads the templates it was written for.
This harness runs the real ingestion path (PDF text extraction / local OCR for photos) and
the real parser, range selection and flagging over reports you collect, against labels you
enter by hand from the paper report.

    python -m eval.run_gold                     # reads eval/gold/labels.json
    python -m eval.run_gold --labels path.json  # another label file
    python -m eval.run_gold --check             # exit 1 if below eval/gold_thresholds.json

Building the set (aim for 30-50 reports):
  - Different lab chains and hospital formats, digital PDFs AND phone photos of paper reports.
  - De-identify BEFORE the file leaves the clinic: black out name, phone, address, IDs.
  - eval/gold/ is git-ignored. Never commit the reports or the label file.
  - Copy eval/gold/labels.example.json to eval/gold/labels.json and fill one entry per report:
    every lab value printed on it, the printed reference range, and the patient's sex/age.

Metrics:
  extraction.recall / precision   lab values found vs labelled (value within 1%)
  printed_range.capture_rate      labelled printed ranges the parser also read
  demographics.sex_accuracy / age_accuracy
  agreement_with_lab              of values the LAB marks abnormal by its printed range, how many
                                  we also flag (a proxy for "would a clinician agree")
  breakdown by layout (digital/photo) and lab chain
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLD_DIR = ROOT / "eval" / "gold"
RESULTS = ROOT / "eval" / "results" / "gold.json"
THRESHOLDS = ROOT / "eval" / "gold_thresholds.json"


def _pct(n, d):
    return round(n / d, 4) if d else None


def _close(a, b, tol=0.01):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(abs(b) * tol, 1e-9)


def _text_for(path: Path) -> tuple[str | None, str | None]:
    """Run the production loader. Returns (text, error)."""
    try:
        from src.ingestion.document_loader import load_document
        return load_document(str(path))["raw_text"], None
    except ModuleNotFoundError as e:  # e.g. easyocr not installed for photo reports
        return None, f"missing dependency: {e.name}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def evaluate(labels: dict, base: Path) -> dict:
    from src.flagging.demographics import parse_demographics, normalize_sex
    from src.flagging.risk_flagging import _canonical_name, flag_lab_values
    from src.ner.medical_ner import _extract_lab_values_regex

    tot = defaultdict(int)
    by_group = defaultdict(lambda: defaultdict(int))
    per_report, skipped = [], []

    for i, rep in enumerate(labels.get("reports", [])):
        path = base / rep["file"]
        text, err = _text_for(path) if path.exists() else (None, "file not found")
        if text is None:
            skipped.append({"index": i, "reason": err})
            continue

        found = _extract_lab_values_regex(text)
        found_by = {_canonical_name(f["test"]): f for f in found}
        gold_labs = rep.get("labs", [])
        hits = 0
        ranges_gold = ranges_hit = 0
        for g in gold_labs:
            f = found_by.get(_canonical_name(g["test"]))
            ok = f is not None and _close(f["value"], g["value"])
            hits += ok
            if g.get("ref_low") is not None or g.get("ref_high") is not None:
                ranges_gold += 1
                if ok and all(_close(f.get(k), g.get(k)) for k in ("ref_low", "ref_high") if g.get(k) is not None):
                    ranges_hit += 1
        correct_found = sum(
            1 for c, f in found_by.items()
            if any(_canonical_name(g["test"]) == c and _close(f["value"], g["value"]) for g in gold_labs)
        )

        demo = parse_demographics(text)
        sex_ok = rep.get("sex") is None or demo["sex"] == normalize_sex(rep["sex"])
        age_ok = rep.get("age") is None or (demo["age_years"] is not None
                                            and abs(demo["age_years"] - float(rep["age"])) < 1)

        # Agreement: judge the LABELLED values (isolates flagging from extraction) and compare
        # our direction with the lab's own printed range.
        flags = flag_lab_values(
            [{"test": g["test"], "value": g["value"], "unit": g.get("unit", ""),
              "ref_low": g.get("ref_low"), "ref_high": g.get("ref_high")} for g in gold_labs],
            gender=rep.get("sex"), age=rep.get("age"))
        lab_abn = [f for f in flags if f.get("lab_direction") in ("high", "low")]
        agree = sum(1 for f in lab_abn if f["direction"] == f["lab_direction"])
        discordant = sum(1 for f in flags if f.get("range_discordant"))

        row = {"gold": len(gold_labs), "hits": hits, "found": len(found), "correct_found": correct_found,
               "ranges_gold": ranges_gold, "ranges_hit": ranges_hit, "sex_ok": int(sex_ok),
               "age_ok": int(age_ok), "lab_abnormal": len(lab_abn), "agree": agree,
               "discordant": discordant, "reports": 1}
        for k, v in row.items():
            tot[k] += v
            by_group[f"layout:{rep.get('layout', 'unknown')}"][k] += v
            by_group[f"lab:{rep.get('lab_chain', 'unknown')}"][k] += v
        per_report.append({"index": i, "layout": rep.get("layout"), "lab_chain": rep.get("lab_chain"),
                           "recall": _pct(hits, len(gold_labs)), "missed": [
                               g["test"] for g in gold_labs
                               if not (_canonical_name(g["test"]) in found_by
                                       and _close(found_by[_canonical_name(g["test"])]["value"], g["value"]))]})

    def summarise(t):
        return {"reports": t["reports"],
                "extraction": {"recall": _pct(t["hits"], t["gold"]),
                               "precision": _pct(t["correct_found"], t["found"]), "values": t["gold"]},
                "printed_range": {"capture_rate": _pct(t["ranges_hit"], t["ranges_gold"])},
                "demographics": {"sex_accuracy": _pct(t["sex_ok"], t["reports"]),
                                 "age_accuracy": _pct(t["age_ok"], t["reports"])},
                "agreement_with_lab": _pct(t["agree"], t["lab_abnormal"]),
                "discordant_values": t["discordant"]}

    return {"overall": summarise(tot),
            "by_group": {k: summarise(v) for k, v in sorted(by_group.items())},
            "per_report": per_report, "skipped": skipped,
            "note": "Real reports, hand-labelled. Per-report rows carry only an index, never file names."}


def check(res: dict) -> list[str]:
    if not THRESHOLDS.exists():
        return []
    thr = json.loads(THRESHOLDS.read_text())
    out = []
    for path, floor in thr.get("min", {}).items():
        v = res["overall"]
        for p in path.split("."):
            v = v.get(p) if isinstance(v, dict) else None
        if v is None or v < floor:
            out.append(f"{path} = {v} < required {floor}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(GOLD_DIR / "labels.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    lp = Path(a.labels)
    if not lp.exists():
        raise SystemExit(f"No label file at {lp}. Copy eval/gold/labels.example.json to "
                         f"eval/gold/labels.json and add your de-identified reports to eval/gold/.")
    res = evaluate(json.loads(lp.read_text()), lp.parent)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["overall"], indent=2))
    if res["skipped"]:
        print(f"\nSkipped {len(res['skipped'])} report(s): " +
              "; ".join(f"#{s['index']}: {s['reason']}" for s in res["skipped"]))
    if a.check:
        bad = check(res)
        if bad:
            print("\nGOLD-SET REGRESSION:\n  " + "\n  ".join(bad))
            sys.exit(1)
        print("\nGold-set thresholds met.")
