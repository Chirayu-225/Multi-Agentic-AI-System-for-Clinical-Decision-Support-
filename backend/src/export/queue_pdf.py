"""
src/export/queue_pdf.py

Renders the clinician review queue as a PDF, for working offline or taking to a ward round.

De-identified by default: patient names (and anything else the redaction map knows about)
are replaced in every text field, and uploaded filenames — which often carry the patient's
name — are left out. include_names=True keeps them, for a clinician who needs the named
copy; the endpoint audits which kind was exported.

All text is escaped before it reaches ReportLab's Paragraph markup, because summaries and
warnings are model or document derived and must never be interpreted as markup.
"""

from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

DISCLAIMER = ("AI-generated analysis awaiting clinician review. Not a diagnosis and not a "
              "substitute for professional medical judgment. Verify every finding against the "
              "source report before acting on it.")

_SEV_COLOUR = {"CRITICAL": colors.HexColor("#B91C1C"), "HIGH": colors.HexColor("#C2410C"),
               "MODERATE": colors.HexColor("#A16207")}


def _sev(s) -> str:
    return str(s or "").replace("Severity.", "").upper()


def _num(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:g}"


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=17, leading=21, alignment=TA_LEFT,
                                spaceAfter=2),
        "meta": ParagraphStyle("m", parent=base["Normal"], fontSize=8.5, leading=11,
                               textColor=colors.HexColor("#4B5563")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12.5, leading=15, spaceBefore=4,
                             spaceAfter=2),
        "h3": ParagraphStyle("h3", parent=base["Heading4"], fontSize=9.5, leading=12, spaceBefore=6,
                             spaceAfter=2, textColor=colors.HexColor("#1F2937")),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9, leading=12),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=8, leading=10),
        "warn": ParagraphStyle("w", parent=base["Normal"], fontSize=8.5, leading=11,
                               textColor=colors.HexColor("#92400E"), leftIndent=8, bulletIndent=0),
        "note": ParagraphStyle("n", parent=base["Normal"], fontSize=8, leading=10,
                               textColor=colors.HexColor("#6B7280")),
    }


def _p(text, style) -> Paragraph:
    return Paragraph(escape(str(text or "")).replace("\n", "<br/>"), style)


def build_queue_pdf(items: list[dict], *, generated_by: str, include_names: bool,
                    generated_at: datetime | None = None) -> bytes:
    """
    items: one dict per pending report, already de-identified by the caller when
    include_names is False. Keys used: patient_id, report_date, created_at, filename,
    overall_severity, flags, interactions, warnings, clinical_summary, patient_summary.
    """
    st = _styles()
    generated_at = generated_at or datetime.now()
    buf = io.BytesIO()
    footer_text = ("CONFIDENTIAL — contains patient names" if include_names
                   else "De-identified export — patient names removed")

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(15 * mm, 10 * mm, f"Review queue · {footer_text}")
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=18 * mm,
                            title="Review queue", author="Agentic Medical Assistant")
    story: list = [
        _p(f"Review queue — {len(items)} report{'s' if len(items) != 1 else ''} awaiting review", st["title"]),
        _p(f"Generated {generated_at:%d %b %Y, %H:%M} by {generated_by} · {footer_text}", st["meta"]),
        Spacer(1, 3),
        _p(DISCLAIMER, st["note"]),
        Spacer(1, 8),
    ]

    if not items:
        story.append(_p("Nothing is waiting for review.", st["body"]))

    for i, it in enumerate(items):
        if i:
            story.append(PageBreak())
        sev = _sev(it.get("overall_severity")) or "UNKNOWN"
        head = f"Patient {it.get('patient_id', '?')} · Risk level {sev}"
        story.append(_p(head, st["h2"]))
        meta = [f"Report date {it.get('report_date') or '—'}",
                f"uploaded {it.get('created_at') or '—'}"]
        if include_names and it.get("filename"):
            meta.append(f"file {it['filename']}")
        story.append(_p(" · ".join(meta), st["meta"]))

        warnings = it.get("warnings") or []
        if warnings:
            story.append(_p(f"Must be acknowledged before release ({len(warnings)})", st["h3"]))
            for w in warnings:
                story.append(Paragraph(escape(str(w)), st["warn"], bulletText="•"))

        flags = it.get("flags") or []
        story.append(_p("Flagged results", st["h3"]))
        if flags:
            rows = [[_p(h, st["cell"]) for h in
                     ("Test", "Value", "Severity", "ICMR range (basis)", "Lab's printed range")]]
            styles = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                      ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                      ("VALIGN", (0, 0), (-1, -1), "TOP")]
            for r_i, f in enumerate(flags, start=1):
                nr = f.get("normal_range") or {}
                icmr = (f"{_num(nr.get('min'))}–{_num(nr.get('max'))}"
                        + (f" ({f['range_basis']})" if f.get("range_basis") else "")) if nr else "—"
                lr = f.get("lab_range") or {}
                lab = (f"{_num(lr.get('min'))}–{_num(lr.get('max'))}" if lr else "—")
                if f.get("range_discordant"):
                    lab += " · disagrees with ICMR"
                s = _sev(f.get("severity"))
                rows.append([_p(f.get("test"), st["cell"]),
                             _p(f"{_num(f.get('value'))} {f.get('unit') or ''} ({f.get('direction', '')})", st["cell"]),
                             _p(s, st["cell"]), _p(icmr, st["cell"]), _p(lab, st["cell"])])
                if s in _SEV_COLOUR:
                    styles.append(("TEXTCOLOR", (2, r_i), (2, r_i), _SEV_COLOUR[s]))
            t = Table(rows, colWidths=[30 * mm, 30 * mm, 20 * mm, 52 * mm, 48 * mm], repeatRows=1)
            t.setStyle(TableStyle(styles))
            story.append(t)
        else:
            story.append(_p("No values were flagged against ICMR or the lab's own range.", st["body"]))

        inter = it.get("interactions") or []
        if inter:
            story.append(_p("Medication alerts", st["h3"]))
            for x in inter:
                drugs = ", ".join(x.get("drugs") or [])
                lab = f" + {x['lab']} {_num(x.get('lab_value'))}" if x.get("lab") else ""
                story.append(Paragraph(
                    f"<b>{escape(str(x.get('severity', '')).capitalize())}</b> · {escape(drugs + lab)} — "
                    f"{escape(str(x.get('effect', '')))} <i>{escape(str(x.get('advice', '')))}</i>",
                    st["body"], bulletText="•"))

        for label, key in (("Clinical summary (AI-generated, not yet reviewed)", "clinical_summary"),
                           ("Patient summary (AI-generated, not yet reviewed)", "patient_summary")):
            if it.get(key):
                story.append(KeepTogether([_p(label, st["h3"]), _p(it[key], st["body"])]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
