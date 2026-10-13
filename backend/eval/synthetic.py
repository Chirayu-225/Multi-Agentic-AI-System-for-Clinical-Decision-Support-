"""
eval/synthetic.py

Deterministic synthetic lab reports with KNOWN ground truth, so every metric in
run_eval.py is measured against something other than the system's own output.

No real patient data is used or needed: names, phone numbers, IDs and values are
generated from a seeded RNG. Lab values are constructed to fall in a chosen band
(normal / abnormal / critical) relative to the ICMR reference ranges, then written
into the report text using one of several real-world layouts and several spellings
of each test name - the variations are where extraction and name-mapping break.
"""

import random
from dataclasses import dataclass, field

from src.flagging.risk_flagging import REFERENCE_RANGES, _get_range

# canonical key -> (display-name variants, unit, decimals)
TESTS: dict[str, tuple[list[str], str, int]] = {
    "hemoglobin":           (["Hemoglobin", "Haemoglobin", "Hb"], "g/dL", 1),
    "creatinine":           (["Creatinine", "Serum Creatinine"], "mg/dL", 2),
    "glucose_fasting":      (["Fasting Glucose", "FBS", "Glucose Fasting"], "mg/dL", 0),
    "glucose_postprandial": (["Post Prandial Glucose", "PPBS"], "mg/dL", 0),
    "wbc":                  (["WBC", "TLC", "Total WBC"], "cells/mcL", 0),
    "platelets":            (["Platelets", "Platelet Count"], "cells/mcL", 0),
    "sodium":               (["Sodium", "Serum Sodium"], "mEq/L", 0),
    "potassium":            (["Potassium", "Serum Potassium"], "mEq/L", 1),
    "alt":                  (["ALT", "SGPT"], "U/L", 0),
    "ast":                  (["AST", "SGOT"], "U/L", 0),
    "tsh":                  (["TSH"], "mIU/L", 2),
    "hba1c":                (["HbA1c"], "%", 1),
}

FIRST = ["Priya", "Mohammed", "Anita", "Rahul", "Kavya", "Suresh", "Fatima", "Arjun", "Meena", "Imran",
         "Deepa", "Karthik", "Sneha", "Vikram", "Lakshmi", "Faraaz"]
LAST = ["Sharma", "Shaik", "Rao", "Nair", "Reddy", "Khan", "Iyer", "Gupta", "Patel", "Menon",
        "Naidu", "Hegde", "Joshi", "Kulkarni"]
HOSPITALS = ["Sunrise General Hospital", "Apollo Clinic Whitefield", "Lakeview Medical Centre",
             "Sri Sai Diagnostics", "Narayana Health City"]
DRUGS = ["Metformin", "Glimepiride", "Amlodipine", "Atorvastatin", "Pantoprazole", "Telmisartan"]
DIAGNOSES = ["Type 2 diabetes mellitus", "Hypertension", "Iron deficiency anaemia", "Hypothyroidism"]
LAYOUTS = ["colon", "table", "dash", "sentence", "grouped", "decimal_comma"]


@dataclass
class Lab:
    canonical: str
    display: str
    value: float
    unit: str
    band: str            # normal | abnormal | critical
    direction: str       # normal | high | low


@dataclass
class Case:
    case_id: int
    layout: str
    gender: str
    name: str
    phone: str
    email: str
    mrn: str
    aadhaar: str
    doctor: str
    hospital: str
    labs: list[Lab]
    meds: list[str]
    diagnosis: str
    text: str = ""
    name_mentions: list[str] = field(default_factory=list)


def _fmt(v: float, dp: int) -> str:
    return f"{v:.{dp}f}" if dp else str(int(round(v)))


def make_value(canonical: str, band: str, gender: str, rng: random.Random):
    """Return (value, direction) in the requested band, or None if the test has no such band."""
    ref = _get_range(canonical, gender)
    meta = REFERENCE_RANGES[canonical]
    lo, hi = ref["min"], ref["max"]
    clo, chi = meta.get("critical_low"), meta.get("critical_high")
    dp = TESTS[canonical][2]
    if band == "normal":
        if lo < 0.2 * hi:          # e.g. HbA1c range starts at 0 - keep "normal" physiologically plausible
            lo = 0.6 * hi
        span = hi - lo
        return round(rng.uniform(lo + 0.2 * span, hi - 0.2 * span), dp), "normal"
    options = []
    if band == "abnormal":
        # strictly outside the range, strictly inside the critical thresholds
        up = hi * rng.uniform(1.06, 1.18)
        if chi is None or up < chi * 0.95:
            options.append((up, "high"))
        dn = lo * rng.uniform(0.82, 0.94)
        if lo > 0 and (clo is None or dn > clo * 1.05):
            options.append((dn, "low"))
    elif band == "critical":
        if chi is not None:
            options.append((chi * rng.uniform(1.05, 1.25), "high"))
        if clo is not None:
            options.append((clo * rng.uniform(0.6, 0.93), "low"))
    if not options:
        return None
    v, d = rng.choice(options)
    return round(v, dp), d


def indian_group(n: int) -> str:
    """12345678 -> 1,23,45,678 (the lakh/crore grouping common on Indian lab reports)."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def _lab_line(lab: Lab, layout: str, ref: dict) -> str:
    v = _fmt(lab.value, TESTS[lab.canonical][2])
    if layout == "grouped":
        if TESTS[lab.canonical][2] == 0 and lab.value >= 1000:
            v = indian_group(int(round(lab.value)))
        return f"{lab.display}: {v} {lab.unit}"
    if layout == "decimal_comma":
        v = v.replace(".", ",")
        return f"{lab.display:<28}{v:<10}{lab.unit:<12}{ref['min']} - {ref['max']}".replace(".", ".")
    if layout == "colon":
        return f"{lab.display}: {v} {lab.unit}"
    if layout == "dash":
        return f"{lab.display} - {v} {lab.unit}"
    if layout == "table":
        rng = f"{ref['min']} - {ref['max']}"
        return f"{lab.display:<28}{v:<10}{lab.unit:<12}{rng}"
    return f"{lab.display} was found to be {v} {lab.unit} on this sample."


def generate_case(case_id: int, rng: random.Random) -> Case:
    gender = rng.choice(["Male", "Female"])
    first, last = rng.choice(FIRST), rng.choice(LAST)
    name = f"{first} {last}"
    doctor = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"
    layout = rng.choice(LAYOUTS)

    labs: list[Lab] = []
    for canonical in rng.sample(list(TESTS), k=rng.randint(5, 8)):
        band = rng.choices(["normal", "abnormal", "critical"], weights=[0.45, 0.4, 0.15])[0]
        made = make_value(canonical, band, gender, rng)
        if made is None:
            band, made = "normal", make_value(canonical, "normal", gender, rng)
        value, direction = made
        names, unit, _ = TESTS[canonical]
        labs.append(Lab(canonical, rng.choice(names), value, unit, band, direction))

    case = Case(
        case_id=case_id, layout=layout, gender=gender, name=name,
        phone=f"{rng.choice('6789')}{rng.randint(10**8, 10**9 - 1)}",
        email=f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@example.com",
        mrn=str(rng.randint(10**9, 10**10 - 1)),
        aadhaar=f"{rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}",
        doctor=doctor, hospital=rng.choice(HOSPITALS), labs=labs,
        meds=rng.sample(DRUGS, k=rng.randint(1, 3)), diagnosis=rng.choice(DIAGNOSES),
    )

    # Real reports repeat the patient's name in several shapes; each is a leak if it survives.
    case.name_mentions = [name, name.upper(), f"{last}, {first}", f"Ms. {last}", first]
    header = [
        case.hospital, "Department of Clinical Pathology", "",
        f"{rng.choice(['Patient Name:', 'Name:', 'PATIENT NAME :', 'Patient Name -', 'Pt. Name:', 'Mr/Ms:', 'Client Name:'])} {name}", f"Age/Sex: {rng.randint(22, 78)} / {gender}",
        f"MRN: {case.mrn}   Aadhaar: {case.aadhaar}", f"Phone: {case.phone}   Email: {case.email}",
        f"Referred by: {doctor}", f"Report Date: 2026-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}", "",
    ]
    if layout in ("table", "decimal_comma"):
        header.append(f"{'Test':<28}{'Result':<10}{'Unit':<12}Reference Range")
    body = [_lab_line(l, layout, _get_range(l.canonical, gender)) for l in labs]
    footer = [
        "", f"Diagnosis: {case.diagnosis}", f"Medication: {', '.join(case.meds)}",
        f"Printed copy for {name.upper()}.", f"Dear Ms. {last}, kindly collect the report.",
        f"Clinical notes: {first} reports no fever.", f"Signature: {doctor}",
    ]
    case.text = "\n".join(header + body + footer)
    return case


def generate_dataset(n: int = 200, seed: int = 42) -> list[Case]:
    rng = random.Random(seed)
    return [generate_case(i, rng) for i in range(n)]
