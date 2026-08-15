"""
OCR orchestration: walks a directory of public-domain PDFs, OCRs each one page-by-page, joins the
pages into one document per PDF, post-corrects the common Devanagari OCR failure modes, and writes
the result to a JSONL.gz shard in the exact schema the rest of the pipeline expects (reusing
lma/crawl/async_crawler.py's ShardWriter, not a separate implementation).

This is the second half of the manual-collection requirement alongside the web crawler --
"OCR from books/PDFs ... you gather yourself" is explicitly counted as manual by the assignment
(see lma/ocr/pdf_ocr.py's module docstring). The primitives (rendering, per-page OCR, confidence
filtering, post-correction) are already built and unit-tested; this script is the piece that
actually runs them over a real folder of scanned books, something that requires the system
`tesseract` binary this dev machine does not have (see is_tesseract_available() below) -- intended
to run on Kaggle/Colab, this project's designated compute, where installing it is one line.

Usage:
    python scripts/run_ocr.py --lang hindi --pdf-dir /path/to/public_domain_hindi_books
    python scripts/run_ocr.py --lang nepali --pdf-dir /path/to/public_domain_nepali_books
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.crawl.async_crawler import ShardWriter  # noqa: E402
from lma.ocr.pdf_ocr import is_tesseract_available, ocr_pdf  # noqa: E402
from lma.ocr.postcorrect import postcorrect_ocr_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.run_ocr")

TESSERACT_LANG = {"hindi": "hin", "nepali": "nep"}
MIN_DOC_CHARS = 200  # matches lma/text/filters.py's FilterConfig.min_chars floor; below this a
                      # whole-PDF OCR result is not worth writing as a document at all


def ocr_one_pdf(pdf_path: Path, lang: str, dpi: int = 300, min_confidence: float = 40.0) -> dict | None:
    """OCRs one PDF into a single document record, or None if too little usable text resulted.

    Pages are joined into ONE document per PDF (not one document per page) so that a book's
    OCR'd text is treated the same unit-of-a-document as a crawled article -- consistent with
    every other stage of the pipeline (heuristic filters, dedup, splitting) operating at
    document granularity.
    """
    page_results = ocr_pdf(pdf_path, lang=TESSERACT_LANG[lang], dpi=dpi, min_confidence=min_confidence)
    if not page_results:
        logger.warning("no usable OCR text from %s (all pages below confidence floor)", pdf_path.name)
        return None

    joined_text = "\n\n".join(r.text for r in page_results)
    corrected_text = postcorrect_ocr_text(joined_text)
    if len(corrected_text) < MIN_DOC_CHARS:
        logger.warning(
            "%s: OCR text too short after post-correction (%d chars < %d floor); skipping",
            pdf_path.name, len(corrected_text), MIN_DOC_CHARS,
        )
        return None

    mean_conf = sum(r.mean_confidence for r in page_results) / len(page_results)
    return {
        "url": f"ocr://{pdf_path.name}",  # not a real URL; keeps the "url" field's schema uniform
        "title": pdf_path.stem,
        "text": corrected_text,
        "content_sha256": hashlib.sha256(corrected_text.encode("utf-8")).hexdigest(),
        "host": "ocr_local",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_type": "manual",  # explicit, matching lma/crawl/async_crawler.py's own tagging
        "ocr_pages_kept": len(page_results),
        "ocr_mean_confidence": round(mean_conf, 2),
    }


def run_ocr_directory(lang: str, pdf_dir: Path, out_dir: Path, dpi: int = 300, min_confidence: float = 40.0) -> dict:
    """OCRs every *.pdf in pdf_dir, writes results to JSONL.gz shards under out_dir. Returns a
    summary dict (pdfs found, documents written, documents skipped) for logging/reporting."""
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    writer = ShardWriter(out_dir)
    written = 0
    skipped = 0
    try:
        for pdf_path in pdf_paths:
            logger.info("OCRing %s...", pdf_path.name)
            try:
                record = ocr_one_pdf(pdf_path, lang, dpi=dpi, min_confidence=min_confidence)
            except Exception:
                logger.exception("OCR failed for %s; skipping", pdf_path.name)
                record = None
            if record is None:
                skipped += 1
                continue
            writer.write(record)
            written += 1
    finally:
        writer.close()

    summary = {"pdfs_found": len(pdf_paths), "documents_written": written, "documents_skipped": skipped}
    logger.info("OCR run complete: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="OCR a directory of PDFs into a manual-collection shard.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory containing *.pdf files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <lang>/data/raw/ocr/")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-confidence", type=float, default=40.0)
    args = parser.parse_args()

    if not is_tesseract_available():
        print(
            "ERROR: tesseract binary not found on PATH. Install it (Kaggle/Colab: "
            "`apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-nep`) before "
            "running this script. See lma/ocr/pdf_ocr.py's module docstring.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = args.out_dir if args.out_dir is not None else REPO_ROOT / args.lang / "data" / "raw" / "ocr"
    summary = run_ocr_directory(args.lang, args.pdf_dir, out_dir, dpi=args.dpi, min_confidence=args.min_confidence)
    print(summary)


if __name__ == "__main__":
    main()
