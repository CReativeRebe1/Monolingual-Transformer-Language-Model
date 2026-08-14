"""Tests for lma.ocr.pdf_ocr.

The OCR-dependent tests are skipped when the `tesseract` binary is not on PATH -- true on this
development machine, false on the Kaggle/Colab environment where corpus-scale OCR actually runs
(see the module docstring in lma/ocr/pdf_ocr.py for the apt-get install command). The
availability check itself and the PDF-rendering step (PyMuPDF only, no tesseract dependency) are
always tested, since those need no external binary.
"""
import fitz
import pytest

from lma.ocr.pdf_ocr import (
    available_ocr_languages,
    is_tesseract_available,
    ocr_pdf,
    render_pdf_pages_to_images,
)

requires_tesseract = pytest.mark.skipif(
    not is_tesseract_available(), reason="tesseract binary not installed on this machine"
)


def _make_test_pdf(tmp_path, n_pages: int = 2):
    """A minimal synthetic PDF with a text string drawn on each page, for render/OCR tests."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Test Page {i + 1}", fontsize=24)
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_is_tesseract_available_returns_bool():
    assert isinstance(is_tesseract_available(), bool)


def test_available_ocr_languages_returns_empty_list_when_tesseract_missing():
    if not is_tesseract_available():
        assert available_ocr_languages() == []


def test_render_pdf_pages_to_images_produces_one_image_per_page(tmp_path):
    pdf_path = _make_test_pdf(tmp_path, n_pages=3)
    images = render_pdf_pages_to_images(pdf_path, dpi=150)
    assert len(images) == 3
    for img in images:
        assert img.width > 0 and img.height > 0


def test_render_pdf_higher_dpi_produces_larger_image(tmp_path):
    pdf_path = _make_test_pdf(tmp_path, n_pages=1)
    low_dpi_images = render_pdf_pages_to_images(pdf_path, dpi=72)
    high_dpi_images = render_pdf_pages_to_images(pdf_path, dpi=300)
    assert high_dpi_images[0].width > low_dpi_images[0].width


def test_ocr_pdf_raises_clear_error_when_tesseract_missing(tmp_path):
    if is_tesseract_available():
        pytest.skip("tesseract IS available; this tests the missing-binary error path")
    pdf_path = _make_test_pdf(tmp_path, n_pages=1)
    with pytest.raises(RuntimeError, match="tesseract binary not found"):
        ocr_pdf(pdf_path)


@requires_tesseract
def test_ocr_pdf_extracts_text_from_synthetic_pdf(tmp_path):
    pdf_path = _make_test_pdf(tmp_path, n_pages=1)
    results = ocr_pdf(pdf_path, lang="eng", min_confidence=0.0)
    assert len(results) >= 1
    assert "Test" in results[0].text or "Page" in results[0].text
