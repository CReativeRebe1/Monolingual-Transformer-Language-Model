"""
Hindi cleaning pipeline orchestrator -- the Yi 5-stage cascade, end to end.

    python -m hindi.clean.run            (or: python scripts/build_corpus.py --lang hindi)

Stage order, and where it deviates from Yi Figure 1:

    1  normalise                     norm.py
    2  language filtering            lang.py
    3b rule-based CORRECTION         repair.py   <-- moved EARLIER than Yi's diagram, deliberately
    3a text metric filtering         rules.py
       repetitive document removal   rules.py
    3c perplexity / quality /
       coherence                     score.py
    4  paragraph -> exact -> near
       deduplication                 dedup.py
    5  semantic -> topic -> safety   safety.py
       decontamination               (beyond Yi; Lecture 3 slide 24)
       budget enforcement, splits, stats

**Why correction runs before filtering.** Yi's figure places Rule Based Correction after the metric
filters. Run in that order, a news article carrying three "विज्ञापन" furniture lines and a leaked JSON
metadata prefix is *rejected* before anything gets the chance to repair it. Repairing first keeps the
prose and discards only the debris. On a low-resource corpus that ordering is worth real tokens, and
repair.py's safety valve stops it from rescuing documents that are mostly boilerplate.

**Only the manual corpus goes through the cascade.** The downloaded side (IndicCorpV2) is already a
cleaned corpus and was deduplicated against the manual side at fetch time; re-filtering it would
waste hours and would blur the manual-vs-downloaded comparison the Phase-1 report has to make. It
also keeps 6.4M documents out of RAM: the cascade holds the manual corpus in memory (~290k documents
for Hindi) while the downloaded side is streamed twice from disk and never materialised.

**The trusted seed is our own Wikipedia dump text, not a network pull.** The perplexity model and
the quality classifier both need trusted in-language prose; Wikipedia is already in the corpus, so
the models train offline and a rerun cannot be broken by a dead endpoint. The quality classifier's
negative class is raw crawl from the same corpus, which is exactly the distribution it has to judge.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import random
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hindi.clean import dedup, lang, repair, rules, safety, score  # noqa: E402
from hindi.clean.config import DEFAULT, LANGUAGE, CleanConfig  # noqa: E402
from hindi.clean.norm import norm  # noqa: E402
from lma.dedup.decontaminate import decontaminate  # noqa: E402
from lma.text.quality import QualityScorer  # noqa: E402
from lma.text.stats import DocumentRecord, compute_corpus_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hindi.clean.run")

MANUAL_DIRS = ("crawled", "fast", "wiki")
SEED_DOCS = 4_000            # trusted Wikipedia documents used to fit the ppl model + classifier
SEED_NEG_DOCS = 4_000        # raw-crawl documents used as the classifier's negative class
MIN_SEED_DOCS = 200          # below this there is not enough trusted text to fit on
REAL_CORPUS_DOCS = 5_000     # at or above this, missing trusted text is an error, not a demo
STAGES = (
    "collected", "language", "repaired_and_rules", "quality_scored",
    "paragraph_deduped", "exact_deduped", "near_deduped", "topic_filtered", "safety_checked",
    "decontaminated",
)


# ----------------------------------------------------------------------------- loading

def read_shards(directory: Path, limit: int | None = None):
    """Yields records from every JSONL.gz shard in `directory`, tolerating torn tails.

    A shard being written by a still-running crawler, or one with a truncated gzip trailer, must
    not abort an hours-long corpus build -- the same tolerance scripts/build_corpus.py uses.
    """
    n = 0
    for shard in sorted(directory.glob("*.jsonl.gz")):
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (rec.get("text") or "").strip():
                        yield rec
                        n += 1
                        if limit and n >= limit:
                            return
        except (EOFError, OSError, zlib.error, UnicodeDecodeError) as e:
            logger.warning("shard %s ended early: %s", shard.name, e)


def load_manual(lang_dir: Path, limit: int | None = None) -> list[dict]:
    """Loads every manual document, or an evenly-spread sample of `limit` of them.

    A `limit` is divided across the source directories rather than filled from the first one. A
    smoke run that drew all its documents from `crawled/` would have no Wikipedia in it, which
    means no trusted text to fit the perplexity model and quality classifier on -- the run fails
    at `fit_models` instead of testing anything.
    """
    present = [s for s in MANUAL_DIRS if (lang_dir / "data" / "raw" / s).exists()]
    per_dir = None if limit is None else max(limit // max(len(present), 1), 1)
    out: list[dict] = []
    for sub in present:
        before = len(out)
        for rec in read_shards(lang_dir / "data" / "raw" / sub, limit=per_dir):
            rec.setdefault("source_type", "manual")
            rec["source_dir"] = sub
            out.append(rec)
        logger.info("  %s: %d documents", sub, len(out) - before)
    return out


# ----------------------------------------------------------------------------- per-document stage

_W: dict = {}


def _init(cfg: CleanConfig, ppl_scorer, ppl_threshold: float, qmodel, coh_threshold: float):
    _W.update(
        cfg=cfg, ppl=ppl_scorer, ppl_thr=ppl_threshold, qmodel=qmodel, coh_thr=coh_threshold
    )
    try:
        from lma.text.langid import FastTextLangID

        _W["lid"] = FastTextLangID()
    except Exception as e:                        # noqa: BLE001 - any failure means "no LID"
        # Degrading is correct here, but silently would not be: without fastText the script ratio
        # and the Hindi/Nepali marker vote still run, and the log line records that the run's
        # language filtering was weaker than usual.
        logger.warning("fastText LID unavailable (%s): using script ratio + marker vote only", e)
        _W["lid"] = None


@dataclass
class DocVerdict:
    idx: int
    keep: bool
    stage: str = ""
    reasons: list[str] = field(default_factory=list)
    text: str = ""
    ppl: float = 0.0
    quality: float = 0.0
    coherence: float | None = None
    repairs: list[str] = field(default_factory=list)


def _one(item: tuple[int, str]) -> DocVerdict:
    """The whole per-document cascade, fused into one pass.

    Fused rather than run as six separate parallel passes: each pass would re-pickle every document
    to the workers and re-read it from disk. One pass over the corpus, six checks per document.
    """
    idx, raw = item
    cfg: CleanConfig = _W["cfg"]

    text = norm(raw)

    ok, why = lang.check(text, config=cfg, lid=_W["lid"])
    if not ok:
        return DocVerdict(idx=idx, keep=False, stage="language", reasons=[why or "lang"])

    rep = repair.repair(text, config=cfg)
    text = rep.text

    verdict = rules.check(text, config=cfg)
    if not verdict.passed:
        return DocVerdict(idx=idx, keep=False, stage="repaired_and_rules",
                          reasons=verdict.failed, repairs=rep.actions)

    p = _W["ppl"].perplexity(text)
    if p > _W["ppl_thr"]:
        return DocVerdict(idx=idx, keep=False, stage="quality_scored",
                          reasons=["perplexity_too_high"], ppl=p, repairs=rep.actions)

    q_ok, q = score.quality(text, _W["qmodel"], config=cfg)
    if not q_ok:
        return DocVerdict(idx=idx, keep=False, stage="quality_scored",
                          reasons=["quality_too_low"], ppl=p, quality=q, repairs=rep.actions)

    c_ok, c = score.coherent(text, config=cfg)
    if not c_ok:
        return DocVerdict(idx=idx, keep=False, stage="quality_scored",
                          reasons=["incoherent"], ppl=p, quality=q, coherence=c,
                          repairs=rep.actions)

    return DocVerdict(idx=idx, keep=True, text=text, ppl=p, quality=q, coherence=c,
                      repairs=rep.actions)


# ----------------------------------------------------------------------------- helpers

def n_words(text: str) -> int:
    return len(text.split())


def fit_models(records: list[dict], cfg: CleanConfig):
    """Fits the perplexity model, quality classifier, and coherence threshold on our own corpus.

    Needs trusted in-language prose for the positive class, which here means Wikipedia. If there is
    not enough of it, what happens next depends on the size of the corpus:

    * **A real corpus** (>= REAL_CORPUS_DOCS documents) raises. Wikipedia missing from a 290k-document
      build means something is wrong with the data, and fitting the classifier on the very crawl it is
      supposed to judge would produce a filter that means nothing while looking like it works.
    * **A demo-scale corpus** degrades instead, with a warning: the perplexity threshold goes to
      infinity and the classifier stays unfitted (scoring a neutral 0.5, which passes). Every other
      stage still runs. scripts/reproduce.sh builds from a handful of bundled sample documents with no
      Wikipedia at all, and it exists to demonstrate the pipeline end to end, not to produce a corpus.
    """
    wiki = [r["text"] for r in records if r.get("source_dir") == "wiki"][:SEED_DOCS]
    crawl = [r["text"] for r in records if r.get("source_dir") != "wiki"][:SEED_NEG_DOCS]
    if len(wiki) < MIN_SEED_DOCS:
        if len(records) >= REAL_CORPUS_DOCS:
            raise RuntimeError(
                f"need >={MIN_SEED_DOCS} wiki documents to fit the learned filters, found "
                f"{len(wiki)} in a corpus of {len(records)} -- refusing to fit the quality "
                f"classifier on the crawl it is meant to judge"
            )
        logger.warning(
            "only %d wiki documents in a %d-document corpus: this is demo scale, so perplexity "
            "and quality filtering are DISABLED for this run (every other stage still runs)",
            len(wiki), len(records),
        )
        scorer = QualityScorer()
        scorer.fit([norm(t) for t in (wiki + crawl)[:SEED_DOCS]] or ["x"])
        return scorer, float("inf"), score.QualityModel(), cfg.coherence_min_cosine

    logger.info("fitting perplexity model on %d trusted wiki documents...", len(wiki))
    scorer, ppl_thr = score.fit_ppl([norm(t) for t in wiki], config=cfg)

    logger.info("fitting quality classifier (%d wiki vs %d crawl)...", len(wiki), len(crawl))
    qmodel = score.QualityModel()
    qmodel.fit([norm(t) for t in wiki], [norm(t) for t in crawl])

    coh_thr = score.calibrate_coherence([norm(t) for t in wiki], config=cfg)
    logger.info("thresholds: ppl<=%.1f quality>=%.2f coherence>=%.4f",
                ppl_thr, cfg.quality_min_score, coh_thr)
    return scorer, ppl_thr, qmodel, coh_thr


def write_split(path: Path, records: list[dict]) -> None:
    with open(path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(r["text"].replace("\n", " ") + "\n")
    with open(path.with_suffix(".manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "content_sha256": r.get("content_sha256"),
                "host": r.get("host"),
                "source_type": r.get("source_type", "manual"),
            }, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- orchestrator

def run(
    lang_dir: Path | None = None,
    report_dir: Path | None = None,
    workers: int | None = None,
    limit: int | None = None,
    config: CleanConfig | None = None,
    include_downloaded: bool = True,
) -> dict:
    """Runs the full cascade for Hindi and writes splits, stats.json, and figures.

    `lang_dir`/`report_dir` are overridable so scripts/reproduce.sh can run against a sample
    directory without touching the real corpus or the graded figures.
    """
    cfg = config or DEFAULT
    lang_dir = lang_dir or REPO_ROOT / LANGUAGE
    report_dir = report_dir or REPO_ROOT / "report" / "figures"
    workers = workers or max(1, mp.cpu_count() - 1)
    t_start = time.time()

    logger.info("=== cleaning %s (workers=%d) ===", LANGUAGE, workers)
    records = load_manual(lang_dir, limit=limit)
    if not records:
        logger.warning("no manual shards found under %s", lang_dir / "data" / "raw")
        return {"language": LANGUAGE, "status": "no_data"}
    logger.info("loaded %d manual documents", len(records))

    funnel: dict[str, int] = {"collected": len(records)}
    rejected: Counter[str] = Counter()
    by_stage: Counter[str] = Counter()
    repairs: Counter[str] = Counter()

    scorer, ppl_thr, qmodel, coh_thr = fit_models(records, cfg)

    logger.info("stages 1-3: normalise, language, repair, rules, ppl/quality/coherence...")
    survivors: list[dict] = []
    t0 = time.time()
    with mp.Pool(workers, initializer=_init,
                 initargs=(cfg, scorer, ppl_thr, qmodel, coh_thr)) as pool:
        items = ((i, r["text"]) for i, r in enumerate(records))
        for n_done, v in enumerate(pool.imap_unordered(_one, items, chunksize=200), start=1):
            for a in v.repairs:
                repairs[a] += 1
            if v.keep:
                rec = dict(records[v.idx])
                rec["text"] = v.text
                rec["ppl"] = v.ppl
                rec["quality"] = v.quality
                rec["coherence"] = v.coherence
                survivors.append(rec)
            else:
                by_stage[v.stage] += 1
                for r in v.reasons:
                    rejected[r] += 1
            if n_done % 50_000 == 0:
                logger.info("  %d/%d documents (%.0f docs/s)",
                            n_done, len(records), n_done / max(time.time() - t0, 1e-9))

    funnel["language"] = len(records) - by_stage["language"]
    funnel["repaired_and_rules"] = funnel["language"] - by_stage["repaired_and_rules"]
    funnel["quality_scored"] = len(survivors)
    logger.info("survived stages 1-3: %d/%d (%.1f%%) in %.0fs",
                len(survivors), len(records), 100 * len(survivors) / len(records), time.time() - t0)

    # ---- stage 4: dedup, paragraph-first (see dedup.py for why the order matters)
    logger.info("stage 4: paragraph dedup...")
    texts = [r["text"] for r in survivors]
    texts, paras_removed, docs_touched = dedup.dedup_paras(texts, config=cfg)
    for r, t in zip(survivors, texts):
        r["text"] = t
    # Paragraph dedup can empty a document or leave it too short to be worth keeping; re-checking
    # length here is cheaper than a second full rules pass and catches exactly that case.
    survivors = [r for r in survivors if len(r["text"]) >= cfg.min_chars]
    funnel["paragraph_deduped"] = len(survivors)
    logger.info("  removed %d paragraphs across %d documents; %d documents remain",
                paras_removed, docs_touched, len(survivors))

    ids = [r.get("content_sha256") or dedup.key(r["text"]) for r in survivors]
    keep, n_exact = dedup.dedup_exact(ids, [r["text"] for r in survivors])
    survivors = [survivors[i] for i in keep]
    funnel["exact_deduped"] = len(survivors)
    logger.info("  exact dedup removed %d; %d remain", n_exact, len(survivors))

    logger.info("  near dedup (MinHash-LSH) over %d documents...", len(survivors))
    t0 = time.time()
    keep, n_near, n_clusters = dedup.dedup_near(
        [str(i) for i in range(len(survivors))], [r["text"] for r in survivors], config=cfg
    )
    survivors = [survivors[i] for i in keep]
    funnel["near_deduped"] = len(survivors)
    logger.info("  near dedup removed %d in %d clusters (%.0fs); %d remain",
                n_near, n_clusters, time.time() - t0, len(survivors))

    # ---- stage 5: semantic -> topic -> safety
    logger.info("stage 5: semantic clustering + topic filtering...")
    cl = safety.cluster([r["text"] for r in survivors], config=cfg)
    keep, cl = safety.topics([r["text"] for r in survivors], cl, config=cfg)
    n_topic_dropped = len(survivors) - len(keep)
    survivors = [survivors[i] for i in keep]
    funnel["topic_filtered"] = len(survivors)
    logger.info("  %d clusters, %d ad-heavy, dropped %d documents",
                cl.n_clusters, len(cl.ad_clusters), n_topic_dropped)

    logger.info("stage 5: safety filtering + PII redaction...")
    safe: list[dict] = []
    n_unsafe = 0
    for r in survivors:
        ok, redacted, n_hits = safety.check(r["text"], config=cfg)
        if not ok:
            n_unsafe += 1
            rejected["safety_blocklist"] += 1
            continue
        r["text"] = redacted
        safe.append(r)
    survivors = safe
    funnel["safety_checked"] = len(survivors)
    logger.info("  dropped %d documents on the safety blocklist", n_unsafe)

    manual_words = sum(n_words(r["text"]) for r in survivors)
    logger.info("manual corpus after cleaning: %d documents, %d words", len(survivors), manual_words)

    # ---- splits (manual side), then decontamination
    rng = random.Random(cfg.seed)
    rng.shuffle(survivors)
    n = len(survivors)
    n_val = max(1, int(n * cfg.split_fractions[1])) if n >= 100 else 0
    n_test = max(1, int(n * cfg.split_fractions[2])) if n >= 100 else 0
    val = survivors[:n_val]
    test = survivors[n_val : n_val + n_test]
    train = survivors[n_val + n_test :]

    # ---- downloaded side: budget-limited pass-through, no cleaning
    downloaded_words = 0
    n_downloaded = 0
    dl_dir = lang_dir / "data" / "raw" / "downloaded"
    budget = int(manual_words * (1 - cfg.target_manual_fraction) / cfg.target_manual_fraction)
    if include_downloaded and dl_dir.exists():
        logger.info("downloaded side: budget %d words (manual %d at target manual share %.0f%%)",
                    budget, manual_words, 100 * cfg.target_manual_fraction)
        for rec in read_shards(dl_dir):
            if downloaded_words >= budget:
                break
            w = n_words(rec["text"])
            rec["source_type"] = "downloaded"
            downloaded_words += w
            n_downloaded += 1
            bucket = rng.random()
            if bucket < cfg.split_fractions[1]:
                val.append(rec)
            elif bucket < cfg.split_fractions[1] + cfg.split_fractions[2]:
                test.append(rec)
            else:
                train.append(rec)
        logger.info("  kept %d downloaded documents (%d words); %d words of the 600M pull unused",
                    n_downloaded, downloaded_words, max(600_000_000 - downloaded_words, 0))

    logger.info("decontaminating train against val/test (13-gram)...")
    n_before = len(train)
    train, n_contaminated, _ = decontaminate(train, val, test, n=13)
    logger.info("  dropped %d/%d train documents", n_contaminated, n_before)
    funnel["decontaminated"] = len(train) + len(val) + len(test)

    # ---- write splits
    splits_dir = lang_dir / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("val", val), ("test", test)):
        write_split(splits_dir / name, recs)
    logger.info("splits: train=%d val=%d test=%d", len(train), len(val), len(test))

    # ---- stats
    final = train + val + test
    manual_final = sum(n_words(r["text"]) for r in final if r.get("source_type") != "downloaded")
    total_final = sum(n_words(r["text"]) for r in final)
    corpus = compute_corpus_stats([
        DocumentRecord(text=r["text"], source_type=r.get("source_type", "manual"),
                       host=r.get("host") or "unknown")
        for r in final
    ])
    stats = {
        "language": LANGUAGE,
        "elapsed_s": round(time.time() - t_start, 1),
        "funnel": {k: funnel[k] for k in STAGES if k in funnel},
        "rejected_by_rule": dict(rejected.most_common()),
        "rejected_by_stage": dict(by_stage),
        "repairs_applied": dict(repairs.most_common()),
        "thresholds": {
            "perplexity": round(ppl_thr, 2),
            "quality_min": cfg.quality_min_score,
            "coherence_min": round(coh_thr, 5),
        },
        "dedup": {"paragraphs_removed": paras_removed, "exact_removed": n_exact,
                  "near_removed": n_near, "near_clusters": n_clusters},
        "clusters": {"n": cl.n_clusters, "ad_heavy": sorted(cl.ad_clusters),
                     "topic_dropped": n_topic_dropped},
        "words": {"manual": manual_final, "downloaded": total_final - manual_final,
                  "total": total_final,
                  "manual_fraction": round(manual_final / max(total_final, 1), 4)},
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "corpus": corpus.__dict__ if hasattr(corpus, "__dict__") else {},
    }
    (lang_dir / "data" / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _figures(report_dir, funnel, stats, final)
    logger.info("manual share of the final corpus: %.1f%% (floor is 20%%)",
                100 * stats["words"]["manual_fraction"])
    logger.info("done in %.0fs -> %s", time.time() - t_start, lang_dir / "data" / "stats.json")
    return stats


def _figures(report_dir: Path, funnel: dict[str, int], stats: dict, records: list[dict]) -> None:
    """Writes the funnel, length-distribution, per-source and manual-vs-downloaded figures.

    Never fatal: a missing plotting backend must not invalidate a completed multi-hour corpus build,
    and every number behind these figures is already in stats.json, so they can be redrawn later.
    The exception is logged rather than swallowed -- an earlier version of this function called
    plot_funnel_chart with the wrong signature and the only symptom was an empty figures directory.
    """
    try:
        from lma.text.stats import PipelineFunnel
        from lma.viz.plots import (
            plot_domain_distribution,
            plot_funnel_chart,
            plot_length_histogram,
            plot_manual_vs_downloaded,
        )

        report_dir.mkdir(parents=True, exist_ok=True)
        names = [k for k in STAGES if k in funnel]
        counts = [funnel[k] for k in names]
        PipelineFunnel(stage_names=names, counts=counts)      # validates the lengths match
        plot_funnel_chart(names, counts, report_dir / f"{LANGUAGE}_funnel.png", LANGUAGE)

        plot_length_histogram(
            [n_words(r["text"]) for r in records],
            report_dir / f"{LANGUAGE}_length_hist.png", LANGUAGE,
        )

        per_source: Counter[str] = Counter()
        for r in records:
            per_source[r.get("source_dir") or r.get("source_type") or "unknown"] += 1
        plot_domain_distribution(
            dict(per_source), report_dir / f"{LANGUAGE}_sources.png", LANGUAGE
        )

        plot_manual_vs_downloaded(
            stats["words"]["manual"], stats["words"]["downloaded"],
            report_dir / f"{LANGUAGE}_manual_vs_downloaded.png", LANGUAGE,
        )
        logger.info("figures -> %s", report_dir)
    except Exception as e:                        # noqa: BLE001
        logger.warning("figure generation skipped: %s: %s", type(e).__name__, e)


def main():
    p = argparse.ArgumentParser(description="Clean the Hindi corpus (Yi 5-stage cascade).")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N manual documents (for a smoke run)")
    p.add_argument("--lang-dir", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--no-downloaded", action="store_true",
                   help="clean the manual side only; skip the downloaded pass-through")
    args = p.parse_args()
    out = run(lang_dir=args.lang_dir, report_dir=args.report_dir, workers=args.workers,
              limit=args.limit, include_downloaded=not args.no_downloaded)
    print(json.dumps({k: v for k, v in out.items() if k != "corpus"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
