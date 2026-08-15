"""Tests for scripts/run_ocr.py -- the OCR orchestration script. Mocks lma.ocr.pdf_ocr.ocr_pdf
(no real tesseract/PDF needed, consistent with this dev machine not having tesseract installed --
see lma/ocr/pdf_ocr.py's module docstring) to test the orchestration logic itself: joining pages
into one document per PDF, post-correction, source_type tagging, the too-short-after-correction
skip, and per-PDF failure isolation.
"""
import importlib.util
import json
import gzip
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_ocr.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_ocr", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _page(text, conf=85.0, n=1):
    from lma.ocr.pdf_ocr import OCRPageResult
    return OCRPageResult(page_number=n, text=text, mean_confidence=conf)


LONG_HINDI_TEXT = "यह एक वास्तविक पुस्तक का पृष्ठ है जिसमें पर्याप्त सामग्री है। " * 6


def test_ocr_one_pdf_returns_manual_tagged_record(tmp_path):
    module = _load_module()
    fake_pdf = tmp_path / "book.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")

    with patch.object(module, "ocr_pdf", return_value=[_page(LONG_HINDI_TEXT, n=1)]):
        record = module.ocr_one_pdf(fake_pdf, "hindi")

    assert record is not None
    assert record["source_type"] == "manual"
    assert record["url"] == "ocr://book.pdf"
    assert record["title"] == "book"
    assert LONG_HINDI_TEXT.split()[0] in record["text"]
    assert "content_sha256" in record


def test_ocr_one_pdf_joins_multiple_pages(tmp_path):
    module = _load_module()
    fake_pdf = tmp_path / "book.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")
    page1 = "पहला पृष्ठ " * 10
    page2 = "दूसरा पृष्ठ " * 10

    with patch.object(module, "ocr_pdf", return_value=[_page(page1, n=1), _page(page2, n=2)]):
        record = module.ocr_one_pdf(fake_pdf, "hindi")

    assert record is not None
    assert "पहला" in record["text"]
    assert "दूसरा" in record["text"]
    assert record["ocr_pages_kept"] == 2


def test_ocr_one_pdf_returns_none_when_no_pages_survive_confidence_floor(tmp_path):
    module = _load_module()
    fake_pdf = tmp_path / "blank.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")

    with patch.object(module, "ocr_pdf", return_value=[]):  # all pages below confidence floor
        record = module.ocr_one_pdf(fake_pdf, "hindi")

    assert record is None


def test_ocr_one_pdf_returns_none_when_text_too_short_after_postcorrect(tmp_path):
    module = _load_module()
    fake_pdf = tmp_path / "tiny.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")

    with patch.object(module, "ocr_pdf", return_value=[_page("क ख ग", n=1)]):  # far under MIN_DOC_CHARS
        record = module.ocr_one_pdf(fake_pdf, "hindi")

    assert record is None


def test_run_ocr_directory_writes_shard_with_manual_tag(tmp_path):
    module = _load_module()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "book1.pdf").write_bytes(b"%PDF-fake")
    (pdf_dir / "book2.pdf").write_bytes(b"%PDF-fake")
    out_dir = tmp_path / "out"

    with patch.object(module, "ocr_pdf", return_value=[_page(LONG_HINDI_TEXT, n=1)]):
        summary = module.run_ocr_directory("hindi", pdf_dir, out_dir)

    assert summary["pdfs_found"] == 2
    assert summary["documents_written"] == 2
    assert summary["documents_skipped"] == 0

    shard_path = next(out_dir.glob("shard_*.jsonl.gz"))
    with gzip.open(shard_path, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 2
    assert all(r["source_type"] == "manual" for r in records)


def test_run_ocr_directory_isolates_a_single_pdf_failure(tmp_path):
    # One corrupt/unreadable PDF must not abort the whole batch.
    module = _load_module()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "good.pdf").write_bytes(b"%PDF-fake")
    (pdf_dir / "bad.pdf").write_bytes(b"%PDF-fake")
    out_dir = tmp_path / "out"

    def fake_ocr_pdf(pdf_path, **kwargs):
        if pdf_path.name == "bad.pdf":
            raise RuntimeError("simulated corrupt PDF")
        return [_page(LONG_HINDI_TEXT, n=1)]

    with patch.object(module, "ocr_pdf", side_effect=fake_ocr_pdf):
        summary = module.run_ocr_directory("hindi", pdf_dir, out_dir)

    assert summary["pdfs_found"] == 2
    assert summary["documents_written"] == 1
    assert summary["documents_skipped"] == 1


def test_run_ocr_directory_with_no_pdfs_writes_nothing(tmp_path):
    module = _load_module()
    pdf_dir = tmp_path / "empty"
    pdf_dir.mkdir()
    out_dir = tmp_path / "out"

    summary = module.run_ocr_directory("hindi", pdf_dir, out_dir)

    assert summary == {"pdfs_found": 0, "documents_written": 0, "documents_skipped": 0}
