"""
Corpus-building entry point: dispatches one language to its own cleaning pipeline.

    python scripts/build_corpus.py --lang hindi
    python scripts/build_corpus.py --lang nepali

This used to hold the whole cascade. The cascade now lives in `hindi/clean/` and `nepali/clean/`,
one self-contained pipeline per language, because the thresholds are what have to differ and a
shared implementation made every divergence a special case instead of the default. Two concrete
examples of divergences that a single shared pipeline could not express honestly: Nepali runs
fastText's confidence bar at 0.50 rather than 0.65 (Hindi's bar rejects 17.9% of correctly-labelled
Nepali), and Nepali keeps the chandrabindu that Hindi folds to anusvara (folding it would misspell
गाउँ, ठाउँ, हुँदा). See each language's clean/config.py for the measurements behind every number.

What remains here is dispatch: the CLI, the language table, and the exit code. Both pipelines expose
the same `run()` signature, so this file has no per-language knowledge beyond the import.

Shared, language-neutral machinery stays in `lma/`: MinHash/LSH/union-find (`lma/dedup/`),
decontamination (`lma/dedup/decontaminate.py`), corpus statistics (`lma/text/stats.py`), plotting
(`lma/viz/`). Anything that encodes a fact about Hindi or Nepali lives in that language's directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.build_corpus")

LANGUAGES = ("hindi", "nepali")


def get_pipeline(language: str):
    """Returns the language's `run` module. Imported lazily so that a broken or half-written
    pipeline for one language cannot stop the other from building."""
    if language == "hindi":
        from hindi.clean import run as pipeline
    elif language == "nepali":
        from nepali.clean import run as pipeline
    else:
        raise ValueError(f"unknown language {language!r}; expected one of {LANGUAGES}")
    return pipeline


def build_corpus(
    language: str,
    lang_dir: Path | None = None,
    report_dir: Path | None = None,
    workers: int | None = None,
    limit: int | None = None,
    include_downloaded: bool = True,
    batch: int | None = None,
    from_interim: bool = False,
) -> dict:
    """Runs one language's cleaning pipeline and returns its stats dict.

    `lang_dir` and `report_dir` are passed straight through: scripts/reproduce.sh points them at an
    isolated demo directory so a demo run never reads the live corpus (which may be receiving new
    shards from a running crawler) or overwrites the real graded figures.
    """
    logger.info("=== build_corpus: %s ===", language)
    pipeline = get_pipeline(language)
    kwargs = dict(lang_dir=lang_dir, report_dir=report_dir, workers=workers, limit=limit,
                  include_downloaded=include_downloaded, from_interim=from_interim)
    if batch is not None:
        kwargs["batch"] = batch
    return pipeline.run(**kwargs)


def main():
    p = argparse.ArgumentParser(description="Build one language's cleaned corpus.")
    p.add_argument("--lang", required=True, choices=LANGUAGES)
    # --workers has been read by this script's main() since it was written, but was never
    # registered, so every invocation died with AttributeError -- including scripts/reproduce.sh.
    p.add_argument("--workers", type=int, default=None,
                   help="parallel workers for the per-document stages (default: cpu_count() - 1)")
    p.add_argument("--batch", type=int, default=None,
                   help="documents held in memory per parallel batch. Lower it if a run runs out of "
                        "memory; raise it for throughput on a machine with RAM to spare.")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N manual documents, spread across sources "
                        "(for a quick smoke run, not for a real build)")
    p.add_argument("--lang-dir", type=Path, default=None,
                   help="override the corpus directory (default: REPO_ROOT/<lang>)")
    p.add_argument("--report-dir", type=Path, default=None,
                   help="override where figures are written (default: REPO_ROOT/report/figures). "
                        "Used by scripts/reproduce.sh so a demo run never overwrites the real "
                        "graded figures.")
    p.add_argument("--no-downloaded", action="store_true",
                   help="clean the manual side only, skipping the downloaded pass-through")
    p.add_argument("--from-interim", action="store_true",
                   help="redo only the split/decontamination phase from a previous run's "
                        "data/interim/stage7.jsonl.gz, skipping the ~1.5h cascade")
    args = p.parse_args()

    stats = build_corpus(
        args.lang,
        lang_dir=args.lang_dir,
        report_dir=args.report_dir,
        workers=args.workers,
        limit=args.limit,
        include_downloaded=not args.no_downloaded,
        batch=args.batch,
        from_interim=args.from_interim,
    )
    # `corpus` holds the full per-host distribution, which is long; it is in stats.json either way.
    print(json.dumps({k: v for k, v in stats.items() if k != "corpus"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
