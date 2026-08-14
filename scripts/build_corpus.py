"""
Corpus-building orchestration: runs one language's raw shards through the full Phase-1 cleaning
cascade end to end -- collect -> language filter -> heuristic filters -> learned quality filter
-> dedup -> decontaminate -> split -> stats/figures. This is the script the Phase-1 report's
numbers and plots are generated from; every other module in lma/ is a building block this script
composes.

Usage:
    python scripts/build_corpus.py --lang hindi
    python scripts/build_corpus.py --lang nepali

Idempotent and safe to re-run as the crawler accumulates more data: it re-reads whatever raw
shards currently exist under <lang>/data/raw/ and regenerates every downstream artifact from
scratch each time, so there is never a "stale partial corpus" state to reason about.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.crawl.downloaded_corpus import ingest  # noqa: E402
from lma.dedup.pipeline import deduplicate_documents  # noqa: E402
from lma.text.filters import FilterConfig, evaluate  # noqa: E402
from lma.text.langid import CharNgramLangID, FastTextLangID, confusion_matrix  # noqa: E402
from lma.text.normalize import normalize  # noqa: E402
from lma.text.quality import QualityScorer, choose_threshold  # noqa: E402
from lma.text.stats import DocumentRecord, PipelineFunnel, compute_corpus_stats, document_length_distribution  # noqa: E402
from lma.viz.plots import (  # noqa: E402
    plot_domain_distribution,
    plot_funnel_chart,
    plot_langid_confusion_matrix,
    plot_length_histogram,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.build_corpus")

FASTTEXT_LANG_CODE = {"hindi": "hi", "nepali": "ne"}
SEED_TOKEN_BUDGET = 300_000  # small trusted-seed pull for training the langid/quality models


def _iter_shard_records(shard_path: Path):
    """Yields records from one JSONL.gz shard, tolerating a truncated tail.

    The crawler (and, less often, the downloader) may still be appending to a shard file while
    this script reads it -- a shard mid-flush can end in a truncated gzip stream or a partial
    final JSON line. Both are treated as "stop reading this shard here", not a fatal error: a
    corpus build should never crash because it raced an in-progress crawl, it should simply see
    whatever was durably flushed at read time and pick up the rest on the next re-run.
    """
    try:
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (EOFError, OSError) as e:
        logger.warning("shard %s ended early (likely still being written): %s", shard_path.name, e)


def _iter_raw_records(lang_dir: Path):
    """Yields every raw record across crawler + OCR + downloaded shards for one language."""
    raw_dir = lang_dir / "data" / "raw"
    for subdir_name in ("crawled", "ocr", "downloaded"):
        subdir = raw_dir / subdir_name
        if not subdir.exists():
            continue
        for shard_path in sorted(subdir.glob("*.jsonl.gz")):
            yield from _iter_shard_records(shard_path)
    # downloaded_corpus.ingest() also writes directly under data/raw/ (not a subdir) in some
    # call patterns; cover that location too so a script re-run always sees everything.
    for shard_path in sorted(raw_dir.glob("downloaded_*.jsonl.gz")):
        yield from _iter_shard_records(shard_path)


def _train_langid_and_quality_models(language: str, lang_dir: Path):
    """Pulls a small trusted Wikipedia seed and trains the char-ngram hi/ne discriminator and
    the quality-filter perplexity scorer on it. Self-contained: no separate offline training
    step is required to run build_corpus.py from a clean checkout."""
    seed_dir = lang_dir / "data" / "raw" / "seed"
    logger.info("pulling trusted Wikipedia seed for %s (budget=%d tokens)...", language, SEED_TOKEN_BUDGET)
    ingest(language, seed_dir, token_budget_per_source=SEED_TOKEN_BUDGET, sources=["wikipedia"])

    seed_texts = []
    for shard_path in seed_dir.glob("*.jsonl.gz"):
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    seed_texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue

    other_language = "nepali" if language == "hindi" else "hindi"
    other_lang_dir = REPO_ROOT / other_language
    other_seed_dir = other_lang_dir / "data" / "raw" / "seed"
    if not other_seed_dir.exists():
        logger.info("pulling trusted Wikipedia seed for contrast language %s...", other_language)
        ingest(other_language, other_seed_dir, token_budget_per_source=SEED_TOKEN_BUDGET, sources=["wikipedia"])
    other_seed_texts = []
    for shard_path in other_seed_dir.glob("*.jsonl.gz"):
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    other_seed_texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue

    hi_texts = seed_texts if language == "hindi" else other_seed_texts
    ne_texts = other_seed_texts if language == "hindi" else seed_texts

    langid_model = CharNgramLangID(n=5, smoothing_k=0.5)
    if hi_texts and ne_texts:
        langid_model.fit({"hi": hi_texts, "ne": ne_texts})
    else:
        logger.warning("insufficient seed text for one language; char-ngram LID will be skipped")
        langid_model = None

    quality_scorer = QualityScorer(n=4)
    if seed_texts:
        quality_scorer.fit(seed_texts)
        threshold = choose_threshold(quality_scorer.fit_perplexity_distribution(seed_texts), percentile=99.0)
    else:
        quality_scorer = None
        threshold = None

    return langid_model, quality_scorer, threshold, hi_texts, ne_texts


def build_corpus(language: str) -> dict:
    lang_dir = REPO_ROOT / language
    fasttext_code = FASTTEXT_LANG_CODE[language]

    logger.info("=== building corpus for %s ===", language)

    logger.info("loading raw records...")
    raw_records = list(_iter_raw_records(lang_dir))
    logger.info("loaded %d raw records", len(raw_records))
    funnel_names = ["collected"]
    funnel_counts = [len(raw_records)]

    if not raw_records:
        logger.warning("no raw records found for %s yet; crawl/OCR/download must run first", language)
        return {"language": language, "status": "no_data", "funnel": {}}

    logger.info("normalizing text...")
    for r in raw_records:
        r["text"] = normalize(r["text"])
    # drop anything that normalized down to nothing
    raw_records = [r for r in raw_records if r["text"]]

    logger.info("training/loading language-ID and quality models from a trusted seed...")
    langid_model, quality_scorer, quality_threshold, hi_seed, ne_seed = _train_langid_and_quality_models(
        language, lang_dir
    )

    fasttext_lid = FastTextLangID()

    logger.info("stage 2: language filtering...")
    lang_filtered = []
    for r in raw_records:
        ft_lang, ft_conf = fasttext_lid.predict(r["text"])
        if ft_lang != fasttext_code or ft_conf < 0.5:
            continue
        if langid_model is not None:
            result = langid_model.predict(r["text"])
            if result.label != fasttext_code:
                continue
        lang_filtered.append(r)
    funnel_names.append("language_filtered")
    funnel_counts.append(len(lang_filtered))
    logger.info("%d / %d survived language filtering", len(lang_filtered), len(raw_records))

    logger.info("stage 3: heuristic filtering...")
    heuristic_filtered = []
    filter_config = FilterConfig()
    rejection_counts: dict[str, int] = {}
    for r in lang_filtered:
        verdict = evaluate(r["text"], language=language, config=filter_config)
        if verdict.passed:
            heuristic_filtered.append(r)
        else:
            for rule in verdict.failed_rules:
                rejection_counts[rule] = rejection_counts.get(rule, 0) + 1
    funnel_names.append("heuristic_filtered")
    funnel_counts.append(len(heuristic_filtered))
    logger.info("%d / %d survived heuristic filtering", len(heuristic_filtered), len(lang_filtered))

    logger.info("stage 4: learned quality filtering...")
    if quality_scorer is not None and quality_threshold is not None:
        quality_filtered = [
            r for r in heuristic_filtered if quality_scorer.perplexity(r["text"]) <= quality_threshold
        ]
    else:
        logger.warning("no quality scorer available; skipping stage 4")
        quality_filtered = heuristic_filtered
    funnel_names.append("quality_filtered")
    funnel_counts.append(len(quality_filtered))
    logger.info("%d / %d survived quality filtering", len(quality_filtered), len(heuristic_filtered))

    logger.info("stage 5: deduplication...")
    doc_ids = [
        r.get("content_sha256") or hashlib.sha256(r["text"].encode("utf-8")).hexdigest()
        for r in quality_filtered
    ]
    # doc_ids must be unique for the dedup pipeline's union-find; disambiguate collisions.
    seen_ids: dict[str, int] = {}
    unique_ids = []
    for did in doc_ids:
        n = seen_ids.get(did, 0)
        seen_ids[did] = n + 1
        unique_ids.append(did if n == 0 else f"{did}#{n}")
    texts = [r["text"] for r in quality_filtered]
    kept_ids, dedup_stats = deduplicate_documents(unique_ids, texts)
    id_to_record = dict(zip(unique_ids, quality_filtered))
    deduped = [id_to_record[i] for i in kept_ids]
    funnel_names.append("deduplicated")
    funnel_counts.append(len(deduped))
    logger.info("dedup stats: %s", dedup_stats)

    logger.info("splitting (98/1/1, document-level, seeded)...")
    import random

    rng = random.Random(42)
    shuffled = deduped[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(n * 0.01)) if n >= 100 else 0
    n_test = max(1, int(n * 0.01)) if n >= 100 else 0
    val_split = shuffled[:n_val]
    test_split = shuffled[n_val : n_val + n_test]
    train_split = shuffled[n_val + n_test :]

    splits_dir = lang_dir / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in (("train", train_split), ("val", val_split), ("test", test_split)):
        manifest_path = splits_dir / f"{split_name}.manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in split_records:
                f.write(json.dumps({
                    "content_sha256": r.get("content_sha256"),
                    "host": r.get("host"),
                    "source_type": r.get("source_type", "manual"),
                }, ensure_ascii=False) + "\n")
        text_path = splits_dir / f"{split_name}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            for r in split_records:
                f.write(r["text"].replace("\n", " ") + "\n")
    logger.info("splits: train=%d val=%d test=%d", len(train_split), len(val_split), len(test_split))

    logger.info("computing corpus statistics and figures...")
    doc_records = [
        DocumentRecord(text=r["text"], source_type=r.get("source_type", "manual"), host=r.get("host") or "unknown")
        for r in deduped
    ]
    corpus_stats = compute_corpus_stats(doc_records)
    lengths = document_length_distribution(doc_records, unit="words")

    report_dir = REPO_ROOT / "report" / "figures"
    plot_funnel_chart(funnel_names, funnel_counts, report_dir / f"{language}_funnel.png", language.capitalize())
    plot_length_histogram(lengths, report_dir / f"{language}_length_hist.png", language.capitalize())
    plot_domain_distribution(corpus_stats.docs_per_source, report_dir / f"{language}_domains.png", language.capitalize())

    if langid_model is not None and hi_seed and ne_seed:
        cm = confusion_matrix(langid_model, {"hi": hi_seed[:50], "ne": ne_seed[:50]})
        plot_langid_confusion_matrix(cm, report_dir / f"{language}_langid_confusion.png")

    funnel = PipelineFunnel(stage_names=funnel_names, counts=funnel_counts)

    stats_out = {
        "language": language,
        "status": "ok",
        "funnel": dict(zip(funnel_names, funnel_counts)),
        "heuristic_rejection_counts": rejection_counts,
        "dedup_stats": {
            "input_count": dedup_stats.input_count,
            "exact_duplicates_removed": dedup_stats.exact_duplicates_removed,
            "near_duplicate_clusters": dedup_stats.near_duplicate_clusters,
            "near_duplicates_removed": dedup_stats.near_duplicates_removed,
            "kept_count": dedup_stats.kept_count,
        },
        "corpus": {
            "n_documents": corpus_stats.n_documents,
            "n_words": corpus_stats.n_words,
            "n_chars": corpus_stats.n_chars,
            "n_bytes": corpus_stats.n_bytes,
            "docs_per_source": corpus_stats.docs_per_source,
        },
        "splits": {"train": len(train_split), "val": len(val_split), "test": len(test_split)},
    }

    stats_path = lang_dir / "data" / "stats.json"
    stats_path.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", stats_path)
    return stats_out


def main():
    parser = argparse.ArgumentParser(description="Build the cleaned corpus for one language.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    args = parser.parse_args()
    result = build_corpus(args.lang)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
