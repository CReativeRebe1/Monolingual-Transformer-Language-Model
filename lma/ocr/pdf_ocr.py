"""
OCR pipeline for the manual-collection requirement: public-domain Hindi/Nepali books and PDFs.

The assignment explicitly counts "OCR from books/PDFs ... you gather yourself" as manual
collection (distinct from downloaded corpora), so this module -- not just the web crawler -- is
part of clearing the >=20% manual-token bar for both languages.

Requires the system `tesseract` binary plus the Devanagari-capable language packs (`hin` for
Hindi, `nep` for Nepali; `script/Devanagari` also works as a script-level fallback). On Kaggle or
Colab (this project's designated pretraining compute, see README) these install with:
    !apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-nep
Locally, `is_tesseract_available()` lets callers (and tests) detect and skip gracefully rather
than fail when the binary is not present -- OCR at corpus scale runs in the same Kaggle
environment as pretraining, not on a laptop.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def is_tesseract_available() -> bool:
    """True if the system `tesseract` binary is on PATH (language packs are checked separately)."""
    return shutil.which("tesseract") is not None


def available_ocr_languages() -> list[str]:
    if not is_tesseract_available():
        return []
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return []


@dataclass
class OCRPageResult:
    page_number: int  # 1-indexed
    text: str
    mean_confidence: float  # 0-100, tesseract's own per-word confidence, averaged


def render_pdf_pages_to_images(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """Rasterize every page of a PDF to a PIL Image at the given DPI.

    300 DPI is the standard OCR-quality target: high enough for tesseract to resolve Devanagari's
    combining vowel signs and conjunct ligatures reliably, without the multi-second-per-page cost
    of going substantially higher.
    """
    images = []
    doc = fitz.open(str(pdf_path))
    try:
        zoom = dpi / 72.0  # PDF's native unit is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def ocr_image(image: Image.Image, lang: str = "hin") -> OCRPageResult | None:
    """OCR a single page image. lang: tesseract language code ('hin' or 'nep').

    Returns None (rather than raising) if tesseract's own per-word confidences indicate the
    result is essentially noise (mean confidence below 0 filtered by tesseract as -1 sentinel
    for non-text regions) -- callers should treat this as "page contributed no usable text",
    not an error, since book scans routinely include blank/illustration-only pages.
    """
    data = pytesseract.image_to_data(
        image, lang=lang, output_type=pytesseract.Output.DICT
    )
    words = []
    confidences = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf = float(conf)
        if text and conf >= 0:  # tesseract uses -1 for non-text detections
            words.append(text)
            confidences.append(conf)
    if not words:
        return None
    return OCRPageResult(
        page_number=-1,  # filled in by the caller, which knows the page index
        text=" ".join(words),
        mean_confidence=sum(confidences) / len(confidences),
    )


def ocr_pdf(
    pdf_path: Path, lang: str = "hin", dpi: int = 300, min_confidence: float = 40.0
) -> list[OCRPageResult]:
    """OCR every page of a PDF, dropping pages whose mean word confidence is below min_confidence.

    The confidence floor exists because a book scan's damaged, blank, or illustration-only pages
    should not inject low-quality "text" into the training corpus -- exactly the kind of noise
    the heuristic filter cascade (lma/text/filters.py) is meant to catch downstream, but catching
    it here too means the OCR quality-vs-yield tradeoff is visible in the Phase-1 report as its
    own statistic (pages OCR'd vs pages kept), not folded into the general filter rejection count.
    """
    if not is_tesseract_available():
        raise RuntimeError(
            "tesseract binary not found on PATH. Install it (Kaggle/Colab: "
            "`apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-nep`) before "
            "calling ocr_pdf(). Use is_tesseract_available() to check first."
        )
    results = []
    images = render_pdf_pages_to_images(pdf_path, dpi=dpi)
    for i, image in enumerate(images, start=1):
        result = ocr_image(image, lang=lang)
        if result is None:
            continue
        result.page_number = i
        if result.mean_confidence >= min_confidence:
            results.append(result)
    return results
