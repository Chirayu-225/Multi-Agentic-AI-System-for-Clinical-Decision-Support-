"""Review-queue PDF export: access control, de-identification, audit, markup safety."""
import fitz  # PyMuPDF

from src.db.models import Report
from src.db.session import SessionLocal
from tests.test_api_flow import (API_KEY, APPROVE, CLIN, PAT, client, fresh_state,  # noqa: F401
                                 login, upload)

URL = "/reviews/pending/export.pdf"


def _text(pdf: bytes) -> str:
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        return "\n".join(p.get_text() for p in doc)


def test_export_is_clinician_only(client):
    assert client.get(URL, headers={"X-API-Key": API_KEY}).status_code == 403
    assert client.get(URL, headers=login(client, PAT)).status_code == 403
    assert client.get(URL).status_code == 401


def test_export_is_deidentified_by_default(client):
    ch = login(client, CLIN)
    upload(client, ch, name="Report_Priya_Sharma.pdf")
    r = client.get(URL, headers=ch)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert "attachment" in r.headers["content-disposition"]
    assert r.headers["cache-control"] == "no-store"
    text = _text(r.content)
    assert "1 report awaiting review" in text and "Patient P001" in text
    assert "Priya" not in text and "Sharma" not in text          # name, and the filename
    assert "Hemoglobin" in text and "Must be acknowledged" in text and "8.4" in text
    assert "De-identified export" in text
    assert "REDACTED" not in text and "[name removed]" in text


def test_named_export_is_explicit_and_audited(client):
    ch = login(client, CLIN)
    upload(client, ch, name="Report_Priya_Sharma.pdf")
    text = _text(client.get(URL + "?include_names=true", headers=ch).content)
    assert "Priya Sharma" in text and "CONFIDENTIAL" in text
    events = client.get("/admin/audit", headers=ch).json()["events"]
    ex = [e for e in events if e["action"] == "reviews.export_pdf"]
    assert ex and ex[0]["detail"]["include_names"] is True


def test_reviewed_reports_are_not_exported(client):
    ch = login(client, CLIN)
    rid = upload(client, ch).json()["report_id"]
    client.post(f"/report/{rid}/review", headers=ch, json=APPROVE)
    text = _text(client.get(URL, headers=ch).content)
    assert "0 reports awaiting review" in text and "Nothing is waiting" in text


def test_model_text_is_escaped_not_rendered_as_markup(client):
    ch = login(client, CLIN)
    rid = upload(client, ch).json()["report_id"]
    with SessionLocal() as db:
        r = db.get(Report, rid)
        r.clinical_summary = "Ratio <b>bold</b> & K < 5 <font size=40>x</font>"
        db.commit()
    r = client.get(URL, headers=ch)
    assert r.status_code == 200
    assert "<b>bold</b>" in _text(r.content)


def test_range_disagreement_is_exported_even_when_icmr_says_normal(client):
    ch = login(client, CLIN)
    rid = upload(client, ch).json()["report_id"]
    with SessionLocal() as db:
        r = db.get(Report, rid)
        r.rule_flags = [{"test": "Hemoglobin", "value": 12.5, "unit": "g/dL", "severity": "NORMAL",
                         "direction": "normal", "normal_range": {"min": 12.0, "max": 15.0},
                         "range_basis": "adult female range", "lab_range": {"min": 12.8, "max": 16.0},
                         "lab_direction": "low", "range_discordant": True}]
        db.commit()
    text = _text(client.get(URL, headers=ch).content)
    assert "disagrees with ICMR" in text and "adult female range" in text
