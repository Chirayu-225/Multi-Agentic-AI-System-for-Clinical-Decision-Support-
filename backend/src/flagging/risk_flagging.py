"""
src/flagging/risk_flagging.py

Multi-layer risk flagging:
  Layer 1 — Rule-based:        lab values vs. ICMR Indian population reference ranges
  Layer 2 — Trend-based:       consistently worsening direction across visits
  Layer 3 — Personal baseline: deviation from THIS patient's own historical
                                mean, independent of the population range —
                                catches drift while still numerically "normal"

Severity levels:
  CRITICAL  — immediate clinical attention required (Layer 1 only — reserved
              for genuine clinical thresholds, never assigned by Layer 3)
  HIGH      — significantly outside normal range / trend / personal baseline
  MODERATE  — borderline, monitor closely
  NORMAL    — within ICMR reference range
"""

import json
import math
import re
import statistics
from datetime import date
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
    # Post-prandial names MUST be listed here: without them the word-level fallback in
    # _canonical_name() matched the single-word alias "glucose" and silently graded a
    # post-prandial value against the FASTING range (found by eval/run_eval.py).
    "postprandial glucose": "glucose_postprandial",
    "post prandial glucose": "glucose_postprandial",
    "post-prandial glucose": "glucose_postprandial",
    "pp glucose": "glucose_postprandial",
    "ppbs": "glucose_postprandial",
    "post prandial blood sugar": "glucose_postprandial",
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
    key = re.sub(r"\s+", " ", key.replace("_", " "))

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
                "flag_type": "population_range",
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
            "flag_type": "population_range",
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
                "flag_type": "longitudinal_trend",
            })

    return trend_flags


# ── Layer 3: Personal-baseline deviation ──────────────────────────────────────

MIN_BASELINE_REPORTS = 3   # need this many PRIOR reports before trusting a mean/std
Z_MODERATE = 1.5
Z_HIGH = 2.5
MIN_PCT_CHANGE = 5.0       # ignore deviations smaller than this even if z-score is high
                           # (guards against flagging rounding-level noise on tests
                           # with very low natural per-patient variance, e.g. sodium)


def flag_baseline_deviation(
    current_labs: list[dict],
    historical_labs: list[list[dict]],
    min_reports: int = MIN_BASELINE_REPORTS,
) -> list[dict]:
    """
    Layer 3: Personal-baseline flagging.

    Unlike Layer 1 (population ICMR range) or Layer 2 (consistent directional
    trend), this flags a value that has drifted meaningfully from THIS
    patient's own historical mean — even if the value is still numerically
    inside the population-normal range and even if it isn't part of a smooth
    multi-visit trend. That's the point of it: it can catch a patient whose
    "normal for them" is a narrow band, before a swing crosses the (wider)
    population threshold that Layer 1 checks.

    A value is flagged only when BOTH:
      - |z-score| against the patient's own historical mean/stdev exceeds
        Z_MODERATE (needs >= min_reports prior values for the stdev to be
        statistically meaningful at all — with fewer, we skip rather than
        guess), AND
      - the percentage change from that mean exceeds MIN_PCT_CHANGE

    Severity is capped at HIGH — CRITICAL is reserved for genuine clinical
    thresholds (Layer 1), not a purely statistical personal-deviation score.

    Returns list of: {
        test, canonical, current_value, patient_mean, patient_stdev,
        z_score, pct_change_from_mean, unit, severity, message, flag_type
    }
    """
    baseline_flags = []

    current_map = {
        _canonical_name(l["test"]): l for l in current_labs if l.get("value") is not None
    }

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
        if canon not in current_map or len(hist_values) < min_reports:
            continue

        try:
            current_val = float(current_map[canon]["value"])
        except (TypeError, ValueError):
            continue

        unit = current_map[canon].get("unit", "")
        mean = statistics.mean(hist_values)
        stdev = statistics.pstdev(hist_values)  # population stdev — this IS the full known history

        if stdev == 0 or mean == 0:
            continue  # no variance (or degenerate mean) to compare against

        z = (current_val - mean) / stdev
        pct_change = abs((current_val - mean) / mean) * 100

        if abs(z) < Z_MODERATE or pct_change < MIN_PCT_CHANGE:
            continue

        severity = Severity.HIGH if abs(z) >= Z_HIGH else Severity.MODERATE
        direction = "above" if z > 0 else "below"

        baseline_flags.append({
            "test": canon,
            "canonical": canon,
            "current_value": current_val,
            "patient_mean": round(mean, 2),
            "patient_stdev": round(stdev, 2),
            "z_score": round(z, 2),
            "pct_change_from_mean": round(pct_change, 1),
            "unit": unit,
            "direction": direction,
            "severity": severity,
            "message": (
                f"{canon.replace('_', ' ').title()}: {current_val} {unit} is {direction} "
                f"this patient's own historical average ({mean:.1f} {unit} across "
                f"{len(hist_values)} prior visits, ±{stdev:.1f}) by {pct_change:.1f}% "
                f"(z={z:.1f}) — unusual for THIS patient specifically, even if within "
                f"the general population range."
            ),
            "flag_type": "personal_baseline",
        })

    return baseline_flags


# ── Layer 4: Slope-based threshold forecasting ────────────────────────────────

MIN_FORECAST_POINTS = 3   # historical + current, need this many to trust a linear fit


def _avg_days_between(date_strs: list[str]) -> float | None:
    """Best-effort average days between consecutive dated visits, oldest-first.
    Returns None if fewer than 2 parseable dates."""
    parsed = []
    for d in date_strs:
        try:
            parsed.append(date.fromisoformat(d))
        except (TypeError, ValueError):
            continue
    if len(parsed) < 2:
        return None
    parsed.sort()
    span_days = (parsed[-1] - parsed[0]).days
    if span_days <= 0:
        return None
    return span_days / (len(parsed) - 1)


def _linear_fit(y: list[float]) -> tuple[float, float]:
    """Least-squares fit of y against x = 0..n-1. Returns (slope, intercept)."""
    n = len(y)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0, y_mean
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def flag_forecast(
    current_labs: list[dict],
    historical_labs: list[list[dict]],
    historical_dates: list[str] | None = None,
    min_points: int = MIN_FORECAST_POINTS,
) -> list[dict]:
    """
    Layer 4: Slope-based threshold-crossing projection.

    IMPORTANT — this is an illustrative estimate from a simple linear fit
    over a handful of points, NOT a validated clinical predictive model. It
    deliberately never drives the top-level risk_summary severity (see
    compute_risk_summary, which does not take forecast_flags) — a shaky
    extrapolation should never outrank an actual critical lab value. Severity
    here is always capped at MODERATE for that reason.

    Only fires when the value is moving monotonically toward the next
    clinically meaningful threshold ahead of it (population-normal boundary
    if still normal, else the critical boundary if already abnormal) — never
    when the series is flat or reversing direction.

    Returns list of: {
        test, canonical, current_value, slope_per_visit, unit,
        target_threshold, target_label, visits_to_cross,
        estimated_days_to_cross (nullable), severity, message, flag_type
    }
    """
    forecasts = []

    current_map = {
        _canonical_name(l["test"]): l for l in current_labs if l.get("value") is not None
    }

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

    avg_days = _avg_days_between(historical_dates) if historical_dates else None

    for canon, hist_values in historical_map.items():
        if canon not in current_map:
            continue
        try:
            current_val = float(current_map[canon]["value"])
        except (TypeError, ValueError):
            continue

        series = hist_values + [current_val]
        if len(series) < min_points:
            continue

        # Require a strictly monotonic series — same bar flag_trends() uses —
        # so we never extrapolate from noisy, non-monotonic readings.
        diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
            continue

        slope, _ = _linear_fit(series)
        if slope == 0:
            continue

        unit = current_map[canon].get("unit", "")
        ref = _get_range(canon)
        ref_data = REFERENCE_RANGES.get(canon, {})
        if ref is None:
            continue
        low, high = ref.get("min"), ref.get("max")
        crit_low, crit_high = ref_data.get("critical_low"), ref_data.get("critical_high")

        target, label = None, None
        if slope < 0:  # declining
            if low is not None and current_val > low:
                target, label = low, "population-normal lower bound"
            elif crit_low is not None and current_val > crit_low:
                target, label = crit_low, "critical-low threshold"
        else:  # rising
            if high is not None and current_val < high:
                target, label = high, "population-normal upper bound"
            elif crit_high is not None and current_val < crit_high:
                target, label = crit_high, "critical-high threshold"

        if target is None:
            continue  # already past every threshold ahead of it, or test unknown

        visits_to_cross = (target - current_val) / slope
        if visits_to_cross <= 0:
            continue  # moving away from the target, not toward it
        visits_to_cross = math.ceil(visits_to_cross)

        estimated_days = round(avg_days * visits_to_cross) if avg_days else None
        eta = f"~{estimated_days} days" if estimated_days else f"~{visits_to_cross} visit(s)"

        forecasts.append({
            "test": canon,
            "canonical": canon,
            "current_value": current_val,
            "slope_per_visit": round(slope, 3),
            "unit": unit,
            "target_threshold": target,
            "target_label": label,
            "visits_to_cross": visits_to_cross,
            "estimated_days_to_cross": estimated_days,
            "severity": Severity.MODERATE,  # always capped — see docstring
            "message": (
                f"{canon.replace('_', ' ').title()} is moving "
                f"{'down' if slope < 0 else 'up'} by ~{abs(slope):.2f} {unit}/visit. "
                f"At this rate it would cross the {label} ({target} {unit}) in {eta}. "
                f"Illustrative linear projection only — not a validated predictive "
                f"model. Clinical correlation required."
            ),
            "flag_type": "forecast",
        })

    return forecasts


def compute_risk_summary(
    rule_flags: list[dict],
    trend_flags: list[dict],
    baseline_flags: list[dict] | None = None,
) -> dict:
    """
    Aggregate all flags into a top-level risk summary.
    `baseline_flags` (Layer 3, personal-deviation) is optional/backward-compatible.
    """
    baseline_flags = baseline_flags or []
    all_severities = (
        [f["severity"] for f in rule_flags]
        + [f["severity"] for f in trend_flags]
        + [f["severity"] for f in baseline_flags]
    )

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
        "total_flags": len(rule_flags) + len(trend_flags) + len(baseline_flags),
        "critical_count": len(critical_flags),
        "high_count": len(high_flags),
        "trend_count": len(trend_flags),
        "baseline_count": len(baseline_flags),
        "critical_tests": [f["test"] for f in critical_flags],
        "requires_immediate_attention": overall == Severity.CRITICAL,
    }
