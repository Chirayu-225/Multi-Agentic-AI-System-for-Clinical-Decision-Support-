"""
src/ingestion/document_loader.py

Handles document ingestion for the Agentic Medical Assistant.
Supports:
  - Digital PDFs (text-based) via PyMuPDF
  - Scanned/handwritten PDFs via EasyOCR
  - Phone photos / scans as JPEG, PNG or WebP (preprocessed, then EasyOCR)
  - Mixed Hindi/English reports via langdetect + indic-transliteration

OCR runs LOCALLY on purpose. Sending the page image to a hosted vision model would put
the patient's name, ID numbers and everything else on the page in a third-party request,
which would bypass the de-identification layer (deidentification.py can only redact TEXT
it controls). OCR output goes through the same redact-before-send path as any other text.
"""

import os
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
        # Support English + Hindi. OCR_GPU=true|false|auto (default auto: use a GPU only
        # if torch can see one - the old hard-coded gpu=True warned/failed on CPU boxes).
        _ocr_reader = easyocr.Reader(["en", "hi"], gpu=_use_gpu())
    return _ocr_reader


def _use_gpu() -> bool:
    setting = os.getenv("OCR_GPU", "auto").lower()
    if setting in ("true", "1", "yes"):
        return True
    if setting in ("false", "0", "no"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ── File-type validation ──────────────────────────────────────────────────────

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_EXTS = PDF_EXTS | IMAGE_EXTS


def sniff_file_type(path: str) -> str | None:
    """Identify a file by its leading bytes ("pdf", "jpeg", "png", "webp") or None.
    The upload endpoint checks this against the extension so a renamed executable or HTML
    file cannot be pushed into the PDF/OCR parsers just by calling it report.pdf."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def extension_matches_content(filename: str, sniffed: str | None) -> bool:
    ext = Path(filename).suffix.lower()
    if sniffed is None:
        return False
    if ext in PDF_EXTS:
        return sniffed == "pdf"
    if ext in (".jpg", ".jpeg"):
        return sniffed == "jpeg"
    if ext == ".png":
        return sniffed == "png"
    if ext == ".webp":
        return sniffed == "webp"
    return False


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


def _ocr_array(reader, img_array) -> tuple[str, list[float]]:
    """Run EasyOCR on one image; return (text in reading order, per-box confidences).
    Boxes are grouped into lines by vertical position so 'Hemoglobin' and '9.2 g/dL' that
    sit on one visual row end up on one text line - the lab-value parser is line-based."""
    results = reader.readtext(img_array, detail=1, paragraph=False)
    boxes = []
    for box, text, conf in results:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        boxes.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), text, float(conf)))
    boxes.sort(key=lambda b: (b[0], b[1]))
    lines: list[list[tuple]] = []
    for b in boxes:
        if lines and abs(b[0] - lines[-1][0][0]) <= max(8.0, 0.6 * b[2]):
            lines[-1].append(b)
        else:
            lines.append([b])
    text = "\n".join(" ".join(t[3] for t in sorted(line, key=lambda t: t[1])) for line in lines)
    return text, [b[4] for b in boxes]


def _extract_scanned(pdf_path: str) -> tuple[str, float | None]:
    """Extract text from a scanned PDF using EasyOCR. Returns (text, mean confidence)."""
    reader = _get_ocr_reader()
    doc = fitz.open(pdf_path)
    pages, confs = [], []
    for page in doc:
        # Render page as image at 200 DPI
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        text, c = _ocr_array(reader, img_array)
        pages.append(text)
        confs += c
    doc.close()
    return "\n\n".join(pages), (sum(confs) / len(confs) if confs else None)


def preprocess_image(img):
    """Make a phone photo / flatbed scan friendlier to OCR: honour EXIF rotation, convert
    to greyscale, stretch contrast, and upscale small images (OCR accuracy drops sharply
    below ~1000px on the long edge). Takes and returns a PIL image."""
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    long_edge = max(img.size)
    if long_edge < 1400:
        scale = 1400 / long_edge
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return img


MAX_IMAGE_PIXELS = 40_000_000  # reject decompression-bomb style images


def _extract_image(image_path: str) -> tuple[str, float | None]:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    with Image.open(image_path) as im:
        if im.width * im.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image is too large to process safely.")
        prepared = preprocess_image(im.convert("RGB"))
    arr = np.array(prepared.convert("RGB"))
    text, confs = _ocr_array(_get_ocr_reader(), arr)
    return text, (sum(confs) / len(confs) if confs else None)


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
    Main entry point. Load a medical PDF (or a JPEG/PNG/WebP photo of a report) and
    return structured text.

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

    ocr_confidence = None
    if path.suffix.lower() in IMAGE_EXTS:
        source_type = "scanned_image"
        raw_text, ocr_confidence = _extract_image(pdf_path)
        num_pages_override = 1
    else:
        num_pages_override = None
        is_digital = _is_digital_pdf(pdf_path)
        source_type = "digital" if is_digital else "scanned"
        if is_digital:
            raw_text = _extract_digital(pdf_path)
        else:
            raw_text, ocr_confidence = _extract_scanned(pdf_path)

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
    if num_pages_override is not None:
        num_pages = num_pages_override
    else:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()

    return {
        "raw_text": raw_text,
        "source_type": source_type,
        "has_hindi": has_hindi,
        "pages": num_pages,
        "filename": path.name,
        # Mean OCR confidence in [0,1]; None for digital PDFs. main.py turns a low value
        # into an ingestion warning so a clinician knows the numbers may be misread.
        "ocr_confidence": ocr_confidence,
    }


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split extracted text into overlapping chunks for NER and embedding.

    Chunks are measured in words but built from whole, UNMODIFIED lines joined with
    newlines. The lab-value parser is line- and spacing-sensitive ("Hemoglobin   10.2   g/dL"):
    this function used to join every word with single spaces, which made it extract nothing
    (or the wrong number) from real uploads - a report with high creatinine was shown as
    "all clear". An over-long single line is split by words.
    """
    lines: list[tuple[str, int]] = []   # (raw line text, word count)
    for raw in text.split("\n"):
        raw = raw.rstrip()
        n = len(raw.split())
        if n == 0:
            continue
        if n <= chunk_size:
            lines.append((raw, n))
            continue
        words = raw.split()
        for i in range(0, n, chunk_size):
            part = words[i:i + chunk_size]
            lines.append((" ".join(part), len(part)))

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    count = 0
    for ln in lines:
        if current and count + ln[1] > chunk_size:
            chunks.append("\n".join(t for t, _ in current))
            keep: list[tuple[str, int]] = []
            kept = 0
            for prev in reversed(current):
                if kept + prev[1] > overlap:
                    break
                keep.insert(0, prev)
                kept += prev[1]
            current, count = keep, kept
        current.append(ln)
        count += ln[1]
    if current:
        chunks.append("\n".join(t for t, _ in current))
    return chunks
