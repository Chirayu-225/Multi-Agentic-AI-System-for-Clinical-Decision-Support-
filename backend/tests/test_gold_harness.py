"""Plumbing test for eval/run_gold.py on generated PDFs (the real gold set never enters CI)."""
from eval.run_gold import evaluate


def _pdf(path, header, rows):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate
    ss = getSampleStyleSheet()
    el = [Paragraph("City Diagnostics - Department of Biochemistry - Final Report", ss["Normal"]),
          Paragraph(header, ss["Normal"]),
          Paragraph("Sample collected and processed as per laboratory standard operating procedure.", ss["Normal"])]
    el += [Preformatted(r, ss["Code"]) for r in rows]
    SimpleDocTemplate(str(path), pagesize=A4).build(el)


def test_gold_harness_scores_real_ingestion(tmp_path):
    _pdf(tmp_path / "a.pdf", "Age/Gender: 34 Years / F",
         ["Hemoglobin              12.5      g/dL         12.8 - 16.0",
          "TSH                     2.4       uIU/mL       0.4 - 4.0"])
    _pdf(tmp_path / "b.pdf", "Age / Sex : 61Y/M",
         ["Serum Creatinine        1.9       mg/dL        0.7 - 1.3"])
    labels = {"reports": [
        {"file": "a.pdf", "layout": "digital", "lab_chain": "x", "sex": "F", "age": 34, "labs": [
            {"test": "Hemoglobin", "value": 12.5, "ref_low": 12.8, "ref_high": 16.0},
            {"test": "TSH", "value": 2.4, "ref_low": 0.4, "ref_high": 4.0}]},
        {"file": "b.pdf", "layout": "digital", "lab_chain": "y", "sex": "M", "age": 61, "labs": [
            {"test": "Creatinine", "value": 1.9, "ref_low": 0.7, "ref_high": 1.3}]},
        {"file": "missing.pdf", "labs": []},
    ]}
    res = evaluate(labels, tmp_path)
    o = res["overall"]
    assert o["reports"] == 2 and len(res["skipped"]) == 1
    assert o["extraction"]["recall"] == 1.0 and o["extraction"]["precision"] == 1.0
    assert o["printed_range"]["capture_rate"] == 1.0
    assert o["demographics"]["sex_accuracy"] == 1.0 and o["demographics"]["age_accuracy"] == 1.0
    # The lab calls Hb 12.5 low (its range starts at 12.8); ICMR female 12-15 calls it normal.
    assert o["discordant_values"] == 1
    assert o["agreement_with_lab"] == 0.5
    assert "lab:x" in res["by_group"] and all("file" not in r for r in res["per_report"])
