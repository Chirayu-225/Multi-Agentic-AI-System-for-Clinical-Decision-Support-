import io

import pytest

from src.ingestion import document_loader as dl


def test_sniff_pdf_jpeg_png_webp(tmp_path):
    cases = {
        "a.pdf": (b"%PDF-1.7\n....", "pdf"),
        "a.jpg": (b"\xff\xd8\xff\xe0" + b"0" * 20, "jpeg"),
        "a.png": (b"\x89PNG\r\n\x1a\n" + b"0" * 20, "png"),
        "a.webp": (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "webp"),
    }
    for name, (data, kind) in cases.items():
        p = tmp_path / name
        p.write_bytes(data)
        assert dl.sniff_file_type(str(p)) == kind
        assert dl.extension_matches_content(name, kind)


def test_disguised_files_are_rejected(tmp_path):
    exe = tmp_path / "report.pdf"
    exe.write_bytes(b"MZ\x90\x00 this is not a pdf")
    assert dl.sniff_file_type(str(exe)) is None
    assert not dl.extension_matches_content("report.pdf", dl.sniff_file_type(str(exe)))

    html = tmp_path / "x.png"
    html.write_bytes(b"<html><script>alert(1)</script>")
    assert not dl.extension_matches_content("x.png", dl.sniff_file_type(str(html)))


def test_pdf_named_as_png_is_rejected(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"%PDF-1.4 ....")
    assert not dl.extension_matches_content("x.png", dl.sniff_file_type(str(p)))


def test_missing_file_sniffs_to_none(tmp_path):
    assert dl.sniff_file_type(str(tmp_path / "nope")) is None


def test_preprocess_upscales_and_greyscales():
    from PIL import Image
    im = Image.new("RGB", (400, 300), (200, 180, 160))
    out = dl.preprocess_image(im)
    assert out.mode == "L"
    assert max(out.size) >= 1400


def test_ocr_lines_group_boxes_on_same_row():
    class FakeReader:
        def readtext(self, arr, detail=1, paragraph=False):
            # (box, text, conf): 'Hemoglobin' and '9.2 g/dL' on one row, 'WBC' + '7200' on next
            def box(x, y): return [[x, y], [x + 50, y], [x + 50, y + 20], [x, y + 20]]
            return [(box(200, 100), "9.2 g/dL", 0.9), (box(10, 102), "Hemoglobin:", 0.8),
                    (box(10, 160), "WBC:", 0.7), (box(200, 161), "7200 cells/mcL", 0.6)]
    text, confs = dl._ocr_array(FakeReader(), None)
    assert text.split("\n") == ["Hemoglobin: 9.2 g/dL", "WBC: 7200 cells/mcL"]
    assert abs(sum(confs) / len(confs) - 0.75) < 1e-9
