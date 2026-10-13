"""
src/tools/trend_series.py

Turn a patient's stored reports into chart-ready series: one line per lab test, with the
ICMR reference band so the UI can shade it. Pure function over already-loaded rows.
"""

from src.flagging.risk_flagging import _canonical_name, _get_range, REFERENCE_RANGES


def build_trend_series(reports: list[dict]) -> list[dict]:
    """
    reports: [{report_id, report_date, lab_values:[{test,value,unit}], gender}] in any order.
    Returns [{test, canonical, unit, ref_low, ref_high, critical_low, critical_high,
              points:[{date, value, report_id}]}] - only tests with >= 1 numeric point.
    """
    by_test: dict[str, dict] = {}
    gender = None
    for r in sorted(reports, key=lambda x: x.get("report_date") or ""):
        gender = r.get("gender") or gender
    for r in sorted(reports, key=lambda x: x.get("report_date") or ""):
        for lab in r.get("lab_values") or []:
            try:
                value = float(lab.get("value"))
            except (TypeError, ValueError):
                continue
            test = str(lab.get("test") or "").strip()
            if not test:
                continue
            canonical = _canonical_name(test)
            series = by_test.setdefault(canonical, {
                "test": test, "canonical": canonical, "unit": lab.get("unit") or "", "points": [],
            })
            if lab.get("unit") and not series["unit"]:
                series["unit"] = lab["unit"]
            series["points"].append({"date": r.get("report_date") or "", "value": value,
                                     "report_id": r.get("report_id")})

    out = []
    for canonical, s in by_test.items():
        ref = _get_range(canonical, gender)
        meta = REFERENCE_RANGES.get(canonical, {})
        s["ref_low"] = ref.get("min") if ref else None
        s["ref_high"] = ref.get("max") if ref else None
        s["critical_low"] = meta.get("critical_low")
        s["critical_high"] = meta.get("critical_high")
        out.append(s)
    out.sort(key=lambda s: (-len(s["points"]), s["test"].lower()))
    return out
