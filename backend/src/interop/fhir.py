"""
src/interop/fhir.py

Export a processed report as an HL7 FHIR R4 Bundle (type "collection"):
Patient, DiagnosticReport, Observation (one per lab value), Condition (diagnoses),
MedicationStatement (medications), and a Provenance-style extension noting AI generation.

Notes on scope and honesty:
  * Structure follows the FHIR R4 resource definitions; it is checked structurally in the
    test-suite (resource types, required fields, internal references resolve) but has NOT
    been run through the official HL7 validator.
  * LOINC codes are attached only for tests in a small, well-known mapping; everything else
    carries text only rather than a guessed code.
  * The Patient resource carries the pseudonymous patient_id and gender only. The patient's
    NAME is deliberately omitted unless include_name=True - an export that leaves the system
    should not carry direct identifiers by default.
  * DiagnosticReport.status is "preliminary" until a clinician approves the report, then
    "final". An AI-generated, unreviewed report is never exported as final.
  * India's ABDM profiles are not applied; this is base R4.
"""

import uuid
from datetime import datetime, timezone

LOINC = {
    "hemoglobin": ("718-7", "Hemoglobin [Mass/volume] in Blood"),
    "glucose_fasting": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma"),
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
    "total_cholesterol": ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma"),
    "hdl": ("2085-9", "HDL Cholesterol [Mass/volume] in Serum or Plasma"),
    "ldl": ("2089-1", "LDL Cholesterol [Mass/volume] in Serum or Plasma"),
    "triglycerides": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma"),
    "alt": ("1742-6", "ALT [Enzymatic activity/volume] in Serum or Plasma"),
    "ast": ("1920-8", "AST [Enzymatic activity/volume] in Serum or Plasma"),
    "tsh": ("3016-3", "TSH [Units/volume] in Serum or Plasma"),
    "wbc": ("6690-2", "Leukocytes [#/volume] in Blood"),
    "platelets": ("777-3", "Platelets [#/volume] in Blood"),
    "sodium": ("2951-2", "Sodium [Moles/volume] in Serum or Plasma"),
    "potassium": ("2823-3", "Potassium [Moles/volume] in Serum or Plasma"),
    "urea": ("3094-0", "Urea nitrogen [Mass/volume] in Serum or Plasma"),
    "uric_acid": ("3084-1", "Urate [Mass/volume] in Serum or Plasma"),
}

# UCUM codes for units seen in this app; unknown units are exported as free text.
UCUM = {"g/dL": "g/dL", "mg/dL": "mg/dL", "mEq/L": "meq/L", "U/L": "U/L", "IU/L": "[IU]/L",
        "mIU/L": "m[IU]/L", "%": "%", "cells/mcL": "/uL", "mmol/L": "mmol/L",
        "ng/mL": "ng/mL", "pg/mL": "pg/mL"}

INTERPRETATION = {  # (severity, direction) -> v3-ObservationInterpretation
    ("CRITICAL", "high"): ("HH", "Critical high"), ("CRITICAL", "low"): ("LL", "Critical low"),
    ("HIGH", "high"): ("H", "High"), ("MODERATE", "high"): ("H", "High"),
    ("HIGH", "low"): ("L", "Low"), ("MODERATE", "low"): ("L", "Low"),
}


def _sev(s) -> str:
    return str(s or "").replace("Severity.", "").upper()


def _urn() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def build_bundle(report: dict, *, include_name: bool = False) -> dict:
    """
    report keys used: report_id, patient_id, report_date, review_status, reviewed_at,
    entities, rule_flags, clinical_summary, filename, risk_summary.
    """
    entities = report.get("entities") or {}
    flags = report.get("rule_flags") or []
    patient_id = report["patient_id"]
    effective = report.get("report_date") or datetime.now(timezone.utc).date().isoformat()
    approved = report.get("review_status") == "approved"

    entries: list[dict] = []

    def add(resource: dict) -> str:
        full_url = _urn()
        entries.append({"fullUrl": full_url, "resource": resource})
        return full_url

    patient = {"resourceType": "Patient", "identifier": [{"system": "urn:agentic-medical-assistant:patient-id", "value": patient_id}]}
    gender = (entities.get("patient", {}).get("gender") or "").lower()
    if gender.startswith("f"):
        patient["gender"] = "female"
    elif gender.startswith("m"):
        patient["gender"] = "male"
    if include_name and entities.get("patient", {}).get("name"):
        patient["name"] = [{"text": entities["patient"]["name"]}]
    patient_ref = add(patient)

    obs_refs = []
    for f in flags:
        if f.get("value") is None:
            continue
        canonical = f.get("canonical")
        code: dict = {"text": f.get("test", "")}
        if canonical in LOINC:
            loinc, display = LOINC[canonical]
            code["coding"] = [{"system": "http://loinc.org", "code": loinc, "display": display}]
        value_q: dict = {"value": f["value"]}
        unit = f.get("unit") or ""
        if unit:
            value_q["unit"] = unit
            if unit in UCUM:
                value_q.update(system="http://unitsofmeasure.org", code=UCUM[unit])
        obs = {
            "resourceType": "Observation", "status": "final" if approved else "preliminary",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                      "code": "laboratory"}]}],
            "code": code, "subject": {"reference": patient_ref},
            "effectiveDateTime": effective, "valueQuantity": value_q,
        }
        rng = f.get("normal_range") or {}
        ref_range = {}
        if rng.get("min") is not None:
            ref_range["low"] = {"value": rng["min"], "unit": unit}
        if rng.get("max") is not None:
            ref_range["high"] = {"value": rng["max"], "unit": unit}
        if ref_range:
            ref_range["text"] = f"{f.get('source', 'ICMR')} reference range"
            obs["referenceRange"] = [ref_range]
        interp = INTERPRETATION.get((_sev(f.get("severity")), f.get("direction")))
        if interp:
            obs["interpretation"] = [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": interp[0], "display": interp[1]}]}]
        elif _sev(f.get("severity")) == "NORMAL" and f.get("direction") == "normal":
            obs["interpretation"] = [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "N", "display": "Normal"}]}]
        obs_refs.append({"reference": add(obs)})

    for dx in entities.get("clinical", {}).get("diagnoses", []):
        add({"resourceType": "Condition", "subject": {"reference": patient_ref},
             "code": {"text": dx},
             "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
             "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "unconfirmed"}]}})

    for med in entities.get("medications", []):
        stmt = {"resourceType": "MedicationStatement", "status": "unknown",
                "medicationCodeableConcept": {"text": med.get("name", "")},
                "subject": {"reference": patient_ref}}
        if med.get("dosage"):
            stmt["dosage"] = [{"text": med["dosage"]}]
        add(stmt)

    dr = {
        "resourceType": "DiagnosticReport",
        "status": "final" if approved else "preliminary",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB"}]}],
        "code": {"text": report.get("filename") or "Laboratory report"},
        "subject": {"reference": patient_ref},
        "effectiveDateTime": effective,
        "result": obs_refs,
        "conclusion": report.get("clinical_summary") or "",
        "extension": [{
            "url": "urn:agentic-medical-assistant:ai-generated",
            "valueBoolean": True,
        }, {
            "url": "urn:agentic-medical-assistant:review-status",
            "valueString": report.get("review_status") or "pending_review",
        }],
    }
    add(dr)
    return {"resourceType": "Bundle", "type": "collection",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entry": entries}
