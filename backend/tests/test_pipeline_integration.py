"""
Integration regression: PDF -> load_document -> tool_registry.extract_entities (chunking +
NER + lab parser) -> ICMR flagging -> drug interactions.

Why this file exists: the lab parser was unit-tested on raw text, but in production the text
goes through chunk_text first, which used to join every line into one long line. The parser
is line-based, so it extracted ZERO values from real uploads and a report with high
creatinine and potassium was shown as "ALL CLEAR / NORMAL". Found by uploading a test PDF.
"""
import asyncio
import io

import pytest

LABS = [
    ("Hemoglobin", "10.2", "g/dL", "13.0 - 17.0"), ("Platelet Count", "1,50,000", "cells/cumm", "150000 - 450000"),
    ("Fasting Blood Glucose", "142", "mg/dL", "70 - 100"), ("HbA1c", "7.8", "%", "4.0 - 5.6"),
    ("Serum Creatinine", "1.9", "mg/dL", "0.7 - 1.3"), ("Potassium", "5.6", "mmol/L", "3.5 - 5.1"),
    ("ALT (SGPT)", "78", "U/L", "7 - 56"), ("TSH", "2.4", "uIU/mL", "0.4 - 4.0"),
]
MEDS = ["Metformin 500 mg", "Lisinopril 10 mg", "Aspirin 75 mg", "Warfarin 5 mg", "Atorvastatin 20 mg", "Ibuprofen 400 mg"]


def _pdf(tmp_path, layout: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Preformatted, Table
    from reportlab.lib.styles import getSampleStyleSheet
    ss = getSampleStyleSheet()
    el = [Paragraph("Patient Name: Arjun Mehta", ss["Normal"]), Paragraph("Age/Gender: 54 Years / Male", ss["Normal"]),
          Paragraph("Current Medications: " + "; ".join("Tab " + m for m in MEDS), ss["Normal"])]
    if layout == "rows":
        for t, v, u, r in LABS:
            el.append(Preformatted(f"{t:<24}{v:<10}{u:<13}{r}", ss["Code"]))
    else:  # "cells": a real table - each cell comes out on its own line
        el.append(Table([["Test", "Result", "Unit", "Reference Range"], *[list(r) for r in LABS]]))
    path = str(tmp_path / f"{layout}.pdf")
    SimpleDocTemplate(path, pagesize=A4).build(el)
    return path


@pytest.fixture
def fake_ner(monkeypatch):
    """A NER model that makes the mistakes seen with the real one."""
    class M:
        def predict_entities(self, text, labels, threshold=0.4):
            return [
                {"label": "lab test", "text": "Serum Creatinine"},      # name with no value
                {"label": "medication", "text": "Tab"},                 # dosage-form word only
                {"label": "diagnosis", "text": "Laboratory Results"},   # junk
                {"label": "patient name", "text": "Arjun Mehta"},
                {"label": "patient gender", "text": "Male"},
            ]
    import src.ner.medical_ner as ner
    monkeypatch.setattr(ner, "_get_model", lambda: M())


@pytest.mark.parametrize("layout", ["rows", "cells"])
def test_real_upload_path_extracts_and_flags_labs(tmp_path, fake_ner, layout):
    from src.ingestion.document_loader import load_document
    from src.tools.tool_registry import extract_entities, check_icmr_flags
    text = load_document(_pdf(tmp_path, layout))["raw_text"]

    ent = asyncio.run(extract_entities(text))
    valued = {l["test"].lower(): l["value"] for l in ent["lab_values"] if l["value"] is not None}
    assert len(valued) == len(LABS), valued
    assert any(v == 150000 for v in valued.values()), "lakh-grouped platelet count misread"
    assert not any(l["value"] is None for l in ent["lab_values"]), "value-less NER stub left behind"
    assert "Tab" not in [m["name"] for m in ent["medications"]]

    flags = asyncio.run(check_icmr_flags(ent["lab_values"], gender="Male"))["flags"]
    abnormal = {f["canonical"] for f in flags if not f["severity"].endswith("NORMAL")}
    assert {"creatinine", "potassium", "hemoglobin"} <= abnormal, abnormal


def test_chunk_text_keeps_line_structure():
    from src.ingestion.document_loader import chunk_text
    text = "\n".join(f"Test{i}  {i}.5  mg/dL" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all("\n" in c for c in chunks)
    # every original line survives intact in at least one chunk
    joined = "\n".join(chunks)
    assert all(f"Test{i}  {i}.5  mg/dL" in joined for i in range(300))   # spacing preserved too


def test_empty_extraction_is_never_reported_as_all_clear():
    """No values read must produce a visible warning, not a silent NORMAL."""
    import inspect
    from src.agents import orchestrator
    src = inspect.getsource(orchestrator)
    assert "No numeric lab values could be extracted" in src
