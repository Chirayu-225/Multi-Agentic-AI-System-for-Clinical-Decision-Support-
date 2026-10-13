"""
src/tools/drug_interactions.py

Cross-medication safety check: drug-drug interactions, drug-lab contraindications
and single-drug advisories, evaluated across the WHOLE medication list on a report
(the OpenFDA lookup in tool_registry.check_drug_safety only looks at one drug at a time).

IMPORTANT - what this is and isn't. This is a small, hand-curated, deterministic rule
set covering well-documented interactions commonly relevant to Indian primary care
(brand names such as Glycomet, Ecosprin, Amaryl are resolved to generics). It is
illustrative decision support, NOT a licensed drug-interaction database: it will miss
interactions, and a clean result is never evidence of safety. It runs offline, needs
no API key, sends nothing to a third party, and every finding names its rule so a
clinician can verify it. A production system should integrate a maintained source
(e.g. a licensed formulary database) instead.
"""

import re

SOURCE = "Curated rule set (illustrative, not exhaustive)"
DISCLAIMER = (
    "Curated, non-exhaustive rule set - absence of a finding does NOT mean a combination "
    "is safe. Verify against a maintained drug-interaction reference before prescribing."
)

# brand / alternate name -> list of generic components
BRAND_TO_GENERIC: dict[str, list[str]] = {
    "glycomet": ["metformin"], "glyciphage": ["metformin"], "obimet": ["metformin"],
    "amaryl": ["glimepiride"], "glimy": ["glimepiride"], "glimestar": ["glimepiride"],
    "diamicron": ["gliclazide"], "daonil": ["glibenclamide"],
    "norvasc": ["amlodipine"], "amlip": ["amlodipine"], "amlokind": ["amlodipine"],
    "zantac": ["ranitidine"], "rantac": ["ranitidine"], "aciloc": ["ranitidine"],
    "ecosprin": ["aspirin"], "disprin": ["aspirin"], "loprin": ["aspirin"],
    "lipitor": ["atorvastatin"], "atorva": ["atorvastatin"], "storvas": ["atorvastatin"],
    "crestor": ["rosuvastatin"], "rosuvas": ["rosuvastatin"], "zocor": ["simvastatin"],
    "cozaar": ["losartan"], "losar": ["losartan"], "telma": ["telmisartan"],
    "cardace": ["ramipril"], "envas": ["enalapril"],
    "lasix": ["furosemide"], "aldactone": ["spironolactone"],
    "coumadin": ["warfarin"], "warf": ["warfarin"],
    "plavix": ["clopidogrel"], "clopivas": ["clopidogrel"],
    "omez": ["omeprazole"], "prilosec": ["omeprazole"], "pantocid": ["pantoprazole"], "pan": ["pantoprazole"],
    "brufen": ["ibuprofen"], "voveran": ["diclofenac"], "voltaren": ["diclofenac"],
    "combiflam": ["ibuprofen", "paracetamol"], "dolo": ["paracetamol"], "crocin": ["paracetamol"],
    "calpol": ["paracetamol"], "naprosyn": ["naproxen"],
    "thyronorm": ["levothyroxine"], "eltroxin": ["levothyroxine"],
    "klacid": ["clarithromycin"], "ciplox": ["ciprofloxacin"], "levoflox": ["levofloxacin"],
    "lanoxin": ["digoxin"], "sorbitrate": ["isosorbide"], "nitroglycerin": ["isosorbide"],
    "viagra": ["sildenafil"], "ultracet": ["tramadol", "paracetamol"],
    "zoloft": ["sertraline"], "shelcal": ["calcium"], "calcium carbonate": ["calcium"],
    "ferrous sulphate": ["iron"], "ferrous sulfate": ["iron"], "fefol": ["iron"],
}

CLASSES: dict[str, set[str]] = {
    "NSAID": {"ibuprofen", "diclofenac", "naproxen", "ketorolac", "aceclofenac", "nimesulide"},
    "ACE_INHIBITOR": {"ramipril", "enalapril", "lisinopril", "perindopril"},
    "ARB": {"losartan", "telmisartan", "olmesartan", "valsartan"},
    "STATIN": {"atorvastatin", "rosuvastatin", "simvastatin", "pravastatin"},
    "NITRATE": {"isosorbide"},
    "K_SPARING": {"spironolactone", "eplerenone", "amiloride", "triamterene"},
    "SULFONYLUREA": {"glimepiride", "gliclazide", "glibenclamide", "glipizide"},
    "FLUOROQUINOLONE": {"ciprofloxacin", "levofloxacin", "ofloxacin", "moxifloxacin"},
    "STRONG_MACROLIDE": {"clarithromycin", "erythromycin"},
}
CLASSES["RAAS"] = CLASSES["ACE_INHIBITOR"] | CLASSES["ARB"]

KNOWN_GENERICS = set().union(*CLASSES.values()) | {
    "metformin", "amlodipine", "ranitidine", "aspirin", "furosemide", "warfarin", "clopidogrel",
    "omeprazole", "pantoprazole", "paracetamol", "levothyroxine", "digoxin", "sildenafil",
    "tramadol", "sertraline", "lithium", "calcium", "iron", "famotidine",
}

# (side A, side B, severity, effect, advice)  - side = generic name or "class:NAME"
DRUG_DRUG_RULES = [
    ("warfarin", "aspirin", "major", "Additive bleeding risk.", "Avoid unless specifically indicated; monitor INR and watch for bleeding."),
    ("warfarin", "class:NSAID", "major", "NSAIDs raise bleeding risk with warfarin (GI bleeding, platelet effect).", "Prefer paracetamol for analgesia; if unavoidable, use the shortest course with INR monitoring."),
    ("aspirin", "class:NSAID", "moderate", "Some NSAIDs (notably ibuprofen) can blunt aspirin's antiplatelet effect; combined GI bleeding risk.", "Separate dosing times or choose an alternative analgesic; consider gastroprotection."),
    ("clopidogrel", "omeprazole", "moderate", "Omeprazole inhibits CYP2C19 and can reduce clopidogrel activation.", "Prefer pantoprazole if a PPI is needed."),
    ("class:RAAS", "class:K_SPARING", "major", "ACE inhibitor/ARB with a potassium-sparing diuretic - risk of hyperkalaemia.", "Monitor serum potassium and renal function closely."),
    ("class:RAAS", "class:NSAID", "moderate", "NSAIDs can blunt the antihypertensive effect and add to acute kidney injury risk.", "Monitor blood pressure and creatinine; avoid prolonged combined use."),
    ("class:STATIN", "class:STRONG_MACROLIDE", "major", "CYP3A4 inhibition raises statin exposure - risk of myopathy/rhabdomyolysis.", "Withhold the statin during the macrolide course or choose a non-interacting antibiotic."),
    ("simvastatin", "amlodipine", "moderate", "Amlodipine raises simvastatin exposure; labelling limits simvastatin to 20 mg/day.", "Cap the simvastatin dose or switch statin."),
    ("class:SULFONYLUREA", "class:FLUOROQUINOLONE", "moderate", "Fluoroquinolones can cause dysglycaemia, including hypoglycaemia, with sulfonylureas.", "Counsel on hypoglycaemia symptoms; monitor glucose during the course."),
    ("digoxin", "furosemide", "moderate", "Loop-diuretic-induced hypokalaemia/hypomagnesaemia increases digoxin toxicity risk.", "Monitor potassium, magnesium and digoxin level."),
    ("digoxin", "class:STRONG_MACROLIDE", "major", "Macrolides can raise digoxin levels (P-gp inhibition / gut flora).", "Monitor digoxin level or choose another antibiotic."),
    ("levothyroxine", "calcium", "moderate", "Calcium reduces levothyroxine absorption.", "Separate doses by at least 4 hours."),
    ("levothyroxine", "iron", "moderate", "Iron reduces levothyroxine absorption.", "Separate doses by at least 4 hours."),
    ("sildenafil", "class:NITRATE", "major", "Severe, potentially fatal hypotension.", "Contraindicated - do not co-prescribe."),
    ("tramadol", "sertraline", "major", "Serotonin syndrome and seizure risk.", "Avoid or use with close monitoring."),
    ("lithium", "class:NSAID", "major", "NSAIDs can increase lithium levels to toxicity.", "Avoid; if needed, monitor lithium level."),
]

# (drug side, lab keyword in the canonical test name, required flag direction, severity, effect, advice)
DRUG_LAB_RULES = [
    ("metformin", "creatinine", "high", "major", "Raised creatinine suggests reduced renal clearance; metformin accumulation risks lactic acidosis.", "Check eGFR; review dose or stop if eGFR is below 30 mL/min."),
    ("class:NSAID", "creatinine", "high", "major", "NSAIDs are nephrotoxic and can worsen existing renal impairment.", "Avoid NSAIDs; consider paracetamol."),
    ("class:RAAS", "potassium", "high", "major", "ACE inhibitor/ARB reduces potassium excretion; potassium is already above range.", "Review the need/dose and repeat potassium urgently."),
    ("class:K_SPARING", "potassium", "high", "major", "Potassium-sparing diuretic with potassium already above range.", "Withhold and repeat potassium urgently."),
    ("class:STATIN", "alt", "high", "moderate", "Transaminases are elevated; statins can add hepatic stress.", "Repeat liver enzymes; review statin if more than 3x upper limit."),
    ("class:SULFONYLUREA", "glucose", "low", "major", "Sulfonylurea with a low glucose reading - hypoglycaemia risk.", "Review dose and counsel on hypoglycaemia."),
]

# single-drug advisories
DRUG_ADVISORIES = {
    "ranitidine": ("moderate", "Ranitidine was withdrawn from the US market in 2020 after NDMA contamination was found.", "Consider famotidine or a PPI; check local regulatory status."),
    "tramadol": ("moderate", "Lowers seizure threshold and carries dependence risk.", "Use the lowest effective dose for the shortest time."),
}


def _tokens(name: str) -> list[str]:
    cleaned = re.sub(r"\b(tab|tablet|cap|capsule|inj|syrup|syp|mg|mcg|g|ml|od|bd|tds|qid|hs)\b\.?", " ", name.lower())
    cleaned = re.sub(r"[\d.,/()+\-]+", " ", cleaned)
    return [t for t in re.split(r"\s+", cleaned.strip()) if t]


def resolve_generics(name: str) -> list[str]:
    """Brand/generic medication string -> list of generic components ([] if unknown)."""
    if not name:
        return []
    low = name.lower()
    for brand, generics in BRAND_TO_GENERIC.items():
        if " " in brand and brand in low:
            return list(generics)
    found: list[str] = []
    for tok in _tokens(name):
        if tok in BRAND_TO_GENERIC:
            found += BRAND_TO_GENERIC[tok]
        elif tok in KNOWN_GENERICS:
            found.append(tok)
    return list(dict.fromkeys(found))


def _matches(side: str, generics: set[str]) -> set[str]:
    if side.startswith("class:"):
        return generics & CLASSES[side[6:]]
    return generics & {side}


def check_drug_interactions(medications: list[dict], flags: list[dict] | None = None) -> dict:
    """
    medications: entities["medications"] - [{name, dosage}]
    flags: rule_flags - used for drug-lab rules (needs 'canonical'/'test' and 'direction').
    """
    flags = flags or []
    resolved: dict[str, list[str]] = {}
    unrecognized: list[str] = []
    for med in medications or []:
        name = med.get("name") if isinstance(med, dict) else str(med)
        generics = resolve_generics(name or "")
        if generics:
            resolved[name] = generics
        elif name:
            unrecognized.append(name)

    all_generics = {g for gs in resolved.values() for g in gs}
    findings: list[dict] = []

    for a, b, severity, effect, advice in DRUG_DRUG_RULES:
        ma, mb = _matches(a, all_generics), _matches(b, all_generics)
        if ma and mb and not (ma == mb and len(ma) < 2 and a == b):
            drugs = sorted(ma | mb)
            if len(drugs) < 2:
                continue
            findings.append({"type": "drug-drug", "severity": severity, "drugs": drugs,
                             "effect": effect, "advice": advice, "source": SOURCE})

    for drug_side, lab_kw, direction, severity, effect, advice in DRUG_LAB_RULES:
        hit = _matches(drug_side, all_generics)
        if not hit:
            continue
        for f in flags:
            name = f"{f.get('canonical', '')} {f.get('test', '')}".lower()
            if lab_kw in name and f.get("direction") == direction and \
                    str(f.get("severity", "")).replace("Severity.", "") != "NORMAL":
                findings.append({"type": "drug-lab", "severity": severity, "drugs": sorted(hit),
                                 "lab": f.get("test"), "lab_value": f.get("value"),
                                 "effect": effect, "advice": advice, "source": SOURCE})
                break

    for generic, (severity, effect, advice) in DRUG_ADVISORIES.items():
        if generic in all_generics:
            findings.append({"type": "drug-advisory", "severity": severity, "drugs": [generic],
                             "effect": effect, "advice": advice, "source": SOURCE})

    order = {"major": 0, "moderate": 1, "minor": 2}
    findings.sort(key=lambda x: (order.get(x["severity"], 3), x["type"]))
    return {
        "interactions": findings,
        "checked_medications": sorted(all_generics),
        "unrecognized_medications": unrecognized,
        "disclaimer": DISCLAIMER,
    }
