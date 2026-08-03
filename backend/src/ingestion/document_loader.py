"""
src/ingestion/document_loader.py

Handles PDF ingestion for MedSight.
Supports:
  - Digital PDFs (text-based) via PyMuPDF
  - Scanned/handwritten PDFs via EasyOCR
  - Mixed Hindi/English reports via langdetect + indic-transliteration
"""

import fitz  # PyMuPDF
import easyocr
import numpy as np
from pathlib import Path
from langdetect import detect, LangDetectException
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# Lazy-load OCR reader (expensive to initialise)
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        # Support English + Hindi
        _ocr_reader = easyocr.Reader(["en", "hi"], gpu=True)
    return _ocr_reader


def _is_digital_pdf(pdf_path: str) -> bool:
    """Returns True if the PDF has extractable text (not a scan)."""
    doc = fitz.open(pdf_path)
    total_chars = sum(len(page.get_text("text")) for page in doc)
    doc.close()
    # Heuristic: if fewer than 100 chars total, treat as scanned
    return total_chars > 100


def _extract_digital(pdf_path: str) -> str:
    """Extract text from a digital (text-based) PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _extract_scanned(pdf_path: str) -> str:
    """Extract text from a scanned PDF using EasyOCR."""
    reader = _get_ocr_reader()
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        # Render page as image at 200 DPI
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        results = reader.readtext(img_array, detail=0, paragraph=True)
        pages.append(" ".join(results))
    doc.close()
    return "\n\n".join(pages)


def _transliterate_hindi(text: str) -> str:
    """
    Detect Hindi segments and transliterate Devanagari to Latin.
    Preserves English segments as-is.
    """
    lines = text.split("\n")
    processed = []
    for line in lines:
        if not line.strip():
            processed.append(line)
            continue
        try:
            lang = detect(line)
        except LangDetectException:
            lang = "en"
        if lang == "hi":
            latin = transliterate(line, sanscript.DEVANAGARI, sanscript.IAST)
            processed.append(f"[HI→IAST] {latin}")
        else:
            processed.append(line)
    return "\n".join(processed)


def load_document(pdf_path: str) -> dict:
    """
    Main entry point. Load a medical PDF and return structured text.

    Returns:
        {
            "raw_text": str,           # Full extracted text
            "source_type": str,        # "digital" or "scanned"
            "has_hindi": bool,         # Whether Hindi was detected
            "pages": int,              # Number of pages
            "filename": str
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    is_digital = _is_digital_pdf(pdf_path)
    source_type = "digital" if is_digital else "scanned"

    if is_digital:
        raw_text = _extract_digital(pdf_path)
    else:
        raw_text = _extract_scanned(pdf_path)

    # Check for Hindi content and transliterate
    has_hindi = False
    try:
        lang = detect(raw_text[:500])
        has_hindi = lang == "hi" or "।" in raw_text or any(
            "\u0900" <= ch <= "\u097F" for ch in raw_text
        )
    except LangDetectException:
        pass

    if has_hindi:
        raw_text = _transliterate_hindi(raw_text)

    # Count pages
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    return {
        "raw_text": raw_text,
        "source_type": source_type,
        "has_hindi": has_hindi,
        "pages": num_pages,
        "filename": path.name,
    }


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split extracted text into overlapping chunks for NER and embedding.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
