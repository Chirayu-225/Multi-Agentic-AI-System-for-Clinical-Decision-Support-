"""
src/flagging/risk_flagging.py

Dual-layer risk flagging:
  Layer 1 — Rule-based: checks lab values against ICMR Indian reference ranges
  Layer 2 — Trend-based: detects worsening patterns across longitudinal history

Severity levels:
  CRITICAL  — immediate clinical attention required
  HIGH      — significantly outside normal range
  MODERATE  — borderline, monitor closely
  NORMAL    — within ICMR reference range
"""

import json
from pathlib import Path
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    NORMAL   = "NORMAL"


# Load ICMR reference ranges
_REF_PATH = Path(__file__).parent / "icmr_reference_ranges.json"
with open(_REF_PATH) as f:
    REFERENCE_RANGES = json.load(f)

# Direct canonical name priority map — bypasses alias lookup entirely
# for the most common lab tests
DIRECT_CANONICAL = {
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "glucose": "glucose_fasting",
    "fbs": "glucose_fasting",
    "fasting glucose": "glucose_fasting",
    "glucose fasting": "glucose_fasting",
    "creatinine": "creatinine",
    "serum creatinine": "creatinine",
    "wbc": "wbc",
    "tlc": "wbc",
    "platelets": "platelets",
    "plt": "platelets",
    "sodium": "sodium",
    "potassium": "potassium",
    "tsh": "tsh",
    "hba1c": "hba1c",
    "alt": "alt",
    "sgpt": "alt",
    "ast": "ast",
    "sgot": "ast",
    "bilirubin": "total_bilirubin",
}

# Normalise lab test name variations to canonical keys
LAB_ALIASES = {
    "hb": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "haemoglobin level": "hemoglobin",
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "blood glucose fasting": "glucose_fasting",
    "fasting glucose": "glucose_fasting",
    "fasting blood glucose": "glucose_fasting",
    "fbs": "glucose_fasting",
    "glucose fasting": "glucose_fasting",
    "glucose": "glucose_fasting",
    "pp glucose": "glucose_postprandial",
    "ppbs": "glucose_postprandial",
    "postprandial glucose": "glucose_postprandial",
    "serum creatinine": "creatinine",
    "s creatinine": "creatinine",
    "creatinine": "creatinine",
    "white blood cells": "wbc",
    "total wbc": "wbc",
    "total leucocyte count": "wbc",
    "tlc": "wbc",
    "wbc": "wbc",
    "leucocytes": "wbc",
    "platelet count": "platelets",
    "plt": "platelets",
    "platelets": "platelets",
    "thrombocytes": "platelets",
    "serum sodium": "sodium",
    "na+": "sodium",
    "sodium": "sodium",
    "serum potassium": "potassium",
    "k+": "potassium",
    "potassium": "potassium",
    "total bilirubin": "total_bilirubin",
    "t bili": "total_bilirubin",
    "bilirubin": "total_bilirubin",
    "sgpt": "alt",
    "alt": "alt",
    "alanine aminotransferase": "alt",
    "sgot": "ast",
    "ast": "ast",
    "aspartate aminotransferase": "ast",
    "thyroid stimulating hormone": "tsh",
    "tsh": "tsh",
    "glycated hemoglobin": "hba1c",
    "hba1c": "hba1c",
    "glycosylated hemoglobin": "hba1c",
}


def _canonical_name(test_name: str) -> str:
    """Map a raw lab test name to its canonical key."""
    key = test_name.lower().strip()

    # Direct lookup first — most reliable
    if key in DIRECT_CANONICAL:
        return DIRECT_CANONICAL[key]

    # Alias lookup
    if key in LAB_ALIASES:
        return LAB_ALIASES[key]

    # Word-level match
    for alias, canonical in LAB_ALIASES.items():
        alias_words = alias.split()
        key_words   = key.split()
        if key == alias:
            return canonical
        if len(key_words) == 1 and key_words[0] in alias_words:
            return canonical
        if len(alias_words) == 1 and alias_words[0] in key_words:
            return canonical

    return key


def _get_range(canonical: str, gender: str = None) -> dict | None:
    """Look up the appropriate ICMR range for a lab test."""
    if canonical not in REFERENCE_RANGES:
        return None
    ref    = REFERENCE_RANGES[canonical]
    ranges = ref["ranges"]

    # Try gender-specific range first
    if gender:
        g = gender.lower().strip()
        if ("male" in g and "female" not in g) and "adult_male" in ranges:
            return ranges["adult_male"]
        if "female" in g and "adult_female" in ranges:
            return ranges["adult_female"]

    # For tests with severity-named ranges (glucose), return the normal range
    if "normal" in ranges:
        return ranges["normal"]

    # Fall back to default
    if "default" in ranges:
        return ranges["default"]

    # Return first available range
    return list(ranges.values())[0]


def flag_lab_values(lab_values: list[dict], gender: str = None) -> list[dict]:
    """
    Layer 1: Rule-based flagging against ICMR reference ranges.

    Args:
        lab_values: list of { test, value, unit }
        gender: optional patient gender for range selection

    Returns:
        list of flags: {
            test, value, unit,
            canonical, normal_range,
            severity, direction, message, source
        }
    """
    flags = []

    for lab in lab_values:
        test = lab.get("test", "")
        value = lab.get("value")
        unit = lab.get("unit", "")

        if value is None:
            continue

        try:
            numeric_val = float(value)
        except (TypeError, ValueError):
            continue

        canonical = _canonical_name(test)
        ref = _get_range(canonical, gender)

        if ref is None:
            # Unknown test — still report it but as informational
            flags.append({
                "test": test,
                "value": numeric_val,
                "unit": unit,
                "canonical": canonical,
                "normal_range": None,
                "severity": Severity.NORMAL,
                "direction": "unknown",
                "message": f"{test}: {numeric_val} {unit} — no ICMR reference range available",
                "source": "N/A",
            })
            continue

        ref_data = REFERENCE_RANGES.get(canonical, {})
        low = ref.get("min")
        high = ref.get("max")
        critical_low = ref_data.get("critical_low")
        critical_high = ref_data.get("critical_high")
        source = ref_data.get("source", "ICMR")

        # Determine severity
        severity = Severity.NORMAL
        direction = "normal"
        message = f"{test}: {numeric_val} {unit} is within normal range ({low}–{high} {unit})"

        if critical_low is not None and numeric_val <= critical_low:
            severity = Severity.CRITICAL
            direction = "low"
            message = (
                f"{test}: {numeric_val} {unit} is CRITICALLY LOW "
                f"(critical threshold: {critical_low} {unit}, normal: {low}–{high} {unit}). "
                f"Immediate clinical attention required."
            )
        elif critical_high is not None and numeric_val >= critical_high:
            severity = Severity.CRITICAL
            direction = "high"
            message = (
                f"{test}: {numeric_val} {unit} is CRITICALLY HIGH "
                f"(critical threshold: {critical_high} {unit}, normal: {low}–{high} {unit}). "
                f"Immediate clinical attention required."
            )
        elif low is not None and numeric_val < low:
            diff_pct = ((low - numeric_val) / low) * 100
            severity = Severity.HIGH if diff_pct > 20 else Severity.MODERATE
            direction = "low"
            message = (
                f"{test}: {numeric_val} {unit} is below normal range "
                f"({low}–{high} {unit}) per {source}. "
                f"Deviation: {diff_pct:.1f}% below minimum."
            )
        elif high is not None and numeric_val > high:
            diff_pct = ((numeric_val - high) / high) * 100
            severity = Severity.HIGH if diff_pct > 20 else Severity.MODERATE
            direction = "high"
            message = (
                f"{test}: {numeric_val} {unit} is above normal range "
                f"({low}–{high} {unit}) per {source}. "
                f"Deviation: {diff_pct:.1f}% above maximum."
            )

        flags.append({
            "test": test,
            "value": numeric_val,
            "unit": unit,
            "canonical": canonical,
            "normal_range": {"min": low, "max": high, "unit": unit},
            "severity": severity,
            "direction": direction,
            "message": message,
            "source": source,
        })

    return flags


def flag_trends(
    current_labs: list[dict],
    historical_labs: list[list[dict]],
    min_reports: int = 2,
) -> list[dict]:
    """
    Layer 2: Longitudinal trend flagging.
    Detects consistently worsening or declining values across visits.

    Args:
        current_labs:   lab values from the current report
        historical_labs: list of lab value lists from past reports (oldest first)
        min_reports:    minimum number of historical reports to compute trend

    Returns:
        list of trend flags: {
            test, trend_direction, values_over_time,
            severity, message
        }
    """
    trend_flags = []

    if len(historical_labs) < min_reports:
        return trend_flags

    # Build value series per test
    current_map = {_canonical_name(l["test"]): l for l in current_labs if l.get("value") is not None}

    # Collect historical values per test — skip nulls
    historical_map: dict[str, list[float]] = {}
    for report_labs in historical_labs:
        for lab in report_labs:
            if lab.get("value") is None:
                continue
            try:
                val = float(lab["value"])
            except (TypeError, ValueError):
                continue
            canon = _canonical_name(lab["test"])
            historical_map.setdefault(canon, []).append(val)

    for canon, hist_values in historical_map.items():
        if canon not in current_map:
            continue

        try:
            current_val = float(current_map[canon]["value"])
        except (TypeError, ValueError):
            continue

        unit = current_map[canon].get("unit", "")
        all_values = hist_values + [current_val]

        # Need at least 3 data points for trend
        if len(all_values) < 3:
            continue

        # Compute trend
        diffs = [all_values[i + 1] - all_values[i] for i in range(len(all_values) - 1)]
        consistently_decreasing = all(d < 0 for d in diffs)
        consistently_increasing = all(d > 0 for d in diffs)
        total_change_pct = (
            abs((current_val - all_values[0]) / all_values[0]) * 100
            if all_values[0] != 0 else 0
        )

        # Flag if change is meaningful (>10%)
        if total_change_pct < 10:
            continue

        ref = _get_range(canon)
        if ref is None:
            continue

        ref_min = ref.get("min", float("inf"))
        ref_max = ref.get("max", float("-inf"))

        direction = None
        if consistently_decreasing and current_val < ref_min:
            direction = "declining"
        elif consistently_increasing and current_val > ref_max:
            direction = "rising"
        elif consistently_decreasing and total_change_pct > 20:
            # Flag significant decline even if not yet below threshold
            direction = "declining"
        elif consistently_increasing and total_change_pct > 20:
            direction = "rising"

        if direction:
            severity = Severity.HIGH if total_change_pct > 25 else Severity.MODERATE
            trend_flags.append({
                "test": canon,
                "trend_direction": direction,
                "values_over_time": all_values,
                "total_change_pct": round(total_change_pct, 1),
                "unit": unit,
                "severity": severity,
                "message": (
                    f"{canon.replace('_', ' ').title()} shows a consistently {direction} trend "
                    f"across {len(all_values)} visits: {' → '.join(str(v) for v in all_values)} {unit}. "
                    f"Total change: {total_change_pct:.1f}%. "
                    f"Trend-based flagging may indicate worsening condition even when individual values appear borderline."
                ),
            })

    return trend_flags


def compute_risk_summary(rule_flags: list[dict], trend_flags: list[dict]) -> dict:
    """
    Aggregate all flags into a top-level risk summary.
    """
    all_severities = [f["severity"] for f in rule_flags] + [f["severity"] for f in trend_flags]

    if Severity.CRITICAL in all_severities:
        overall = Severity.CRITICAL
        color = "red"
    elif Severity.HIGH in all_severities:
        overall = Severity.HIGH
        color = "orange"
    elif Severity.MODERATE in all_severities:
        overall = Severity.MODERATE
        color = "yellow"
    else:
        overall = Severity.NORMAL
        color = "green"

    critical_flags = [f for f in rule_flags if f["severity"] == Severity.CRITICAL]
    high_flags = [f for f in rule_flags if f["severity"] == Severity.HIGH]

    return {
        "overall_severity": overall,
        "color": color,
        "total_flags": len(rule_flags) + len(trend_flags),
        "critical_count": len(critical_flags),
        "high_count": len(high_flags),
        "trend_count": len(trend_flags),
        "critical_tests": [f["test"] for f in critical_flags],
        "requires_immediate_attention": overall == Severity.CRITICAL,
    }
