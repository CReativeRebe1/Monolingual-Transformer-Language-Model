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
import re
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
from lma.dedup.decontaminate import word_ngrams  # noqa: E402
from lma.dedup.lsh import LSHIndex  # noqa: E402
from lma.dedup.minhash import MinHasher, estimate_jaccard, word_shingles  # noqa: E402
from lma.dedup.union import UnionFind  # noqa: E402
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


BATCH_DOCS = 20_000          # documents held in memory per parallel batch
HOLDOUT_WINDOW = 2_000_000   # downloaded documents scanned when filling held-out caps
MANUAL_FLOOR = 0.20          # the assignment's hard requirement, enforced at the end of a run


def default_workers() -> int:
    """Workers to use when the caller does not say.

    Capped by available memory, not just by core count. This machine has 16 cores but 7.4 GB of RAM,
    and the first full run died with MemoryError while pickling a task to the pool: each spawned
    worker on Windows is a fresh interpreter with its own copy of the imports and the fitted models.
    Cores are the cheap resource here; memory is not.
    """
    cores = max(1, mp.cpu_count() - 1)
    try:
        import psutil

        budget = int(psutil.virtual_memory().available / (400 * 1024 * 1024))
    except Exception:                             # noqa: BLE001 - psutil is optional
        budget = 6
    return max(2, min(cores, budget, 8))


def batched(iterable, n: int):
    """Yields lists of up to n items. The batch boundary is where memory is reclaimed."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def load_manual_stream(lang_dir: Path, limit: int | None = None):
    """Streams manual records without building a list of the whole corpus.

    The list-building version of this (`load_manual`, kept below for tests and small corpora) holds
    every document's text at once: ~2 GB for Nepali, which does not fit here alongside the workers.

    A `limit` is divided across the source directories rather than filled from the first one, for the
    same reason as in load_manual -- a sample drawn entirely from crawled/ contains no Wikipedia, and
    the learned filters would have nothing trusted to fit on.
    """
    present = [s for s in MANUAL_DIRS if (lang_dir / "data" / "raw" / s).exists()]
    per_dir = None if limit is None else max(limit // max(len(present), 1), 1)
    for sub in present:
        n = 0
        for rec in read_shards(lang_dir / "data" / "raw" / sub, limit=per_dir):
            rec.setdefault("source_type", "manual")
            rec["source_dir"] = sub
            n += 1
            yield rec
        logger.info("  %s: %d documents", sub, n)


def collect_seeds(lang_dir: Path, limit: int | None = None) -> tuple[list[str], list[str], int]:
    """One pass to gather trusted (wiki) and untrusted (crawl) seed text for the learned filters.

    Returns (wiki_texts, crawl_texts, n_documents_seen). Deliberately a separate pass rather than a
    by-product of the main loop: the models must be fitted before the first document is judged, and
    holding the corpus until they are is what ran the machine out of memory.
    """
    wiki: list[str] = []
    neg: list[str] = []
    n_seen = 0
    for rec in load_manual_stream(lang_dir, limit=limit):
        n_seen += 1
        if rec.get("source_dir") == "wiki":
            if len(wiki) < SEED_DOCS:
                wiki.append(rec["text"])
        elif len(neg) < SEED_NEG_DOCS:
            neg.append(rec["text"])
    logger.info("seeds: %d trusted (wiki) + %d crawl documents from %d total",
                len(wiki), len(neg), n_seen)
    return wiki, neg, n_seen


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


def fit_models(records: list[dict], cfg: CleanConfig, corpus_size: int | None = None):
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
    n_docs = corpus_size if corpus_size is not None else len(records)
    wiki = [r["text"] for r in records if r.get("source_dir") == "wiki"][:SEED_DOCS]
    crawl = [r["text"] for r in records if r.get("source_dir") != "wiki"][:SEED_NEG_DOCS]
    if len(wiki) < MIN_SEED_DOCS:
        if n_docs >= REAL_CORPUS_DOCS:
            raise RuntimeError(
                f"need >={MIN_SEED_DOCS} wiki documents to fit the learned filters, found "
                f"{len(wiki)} in a corpus of {n_docs} -- refusing to fit the quality "
                f"classifier on the crawl it is meant to judge"
            )
        logger.warning(
            "only %d wiki documents in a %d-document corpus: this is demo scale, so perplexity "
            "and quality filtering are DISABLED for this run (every other stage still runs)",
            len(wiki), n_docs,
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


# Every character Python's universal-newline text mode treats as a line break, plus the Unicode
# separators. Replacing only the newline character is not enough: a document carrying a bare
# carriage return is written as one line and read back as TWO, which silently desynchronises the
# .txt file from its .manifest.jsonl and misattributes every document after it. That happened on
# the real corpus -- 21,823 Hindi documents -- and the downloaded side is stored raw by design, so
# it carries whatever line-break characters the source had.
_LINEBREAK_RE = re.compile("[\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029]+")


class SplitWriter:
    """Appends documents to one split's .txt and .manifest.jsonl as they arrive.

    Incremental rather than write-a-list, because neither side of this corpus fits in memory: the
    downloaded side is millions of documents and the manual side is ~2 GB of text on a machine with
    7.4 GB of RAM. Documents are written one per line with newlines flattened to spaces, the format
    every downstream consumer already expects (scripts/tokenizer_sweep.py,
    scripts/tokenizer_comparison.py, scripts/reproduce.sh).
    """

    def __init__(self, prefix: Path):
        self.n = 0
        self._text = open(prefix.with_suffix(".txt"), "w", encoding="utf-8")
        self._manifest = open(prefix.with_suffix(".manifest.jsonl"), "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._text.write(_LINEBREAK_RE.sub(" ", record["text"]) + "\n")
        self._manifest.write(json.dumps({
            "content_sha256": record.get("content_sha256"),
            "host": record.get("host"),
            "source_type": record.get("source_type", "manual"),
        }, ensure_ascii=False) + "\n")
        self.n += 1

    def write_all(self, records) -> None:
        for r in records:
            self.write(r)

    def close(self) -> None:
        self._text.close()
        self._manifest.close()


def split_of(key: str, cfg: CleanConfig) -> str:
    """Deterministic train/val/test assignment from a document's own hash.

    Derived from the document rather than drawn from an RNG so that repeated streaming passes agree
    exactly on which document goes where without sharing state, and so a rerun reproduces the same
    splits regardless of shard iteration order.
    """
    draw = int(key[:8], 16) / 0xFFFFFFFF
    if draw < cfg.split_fractions[1]:
        return "val"
    if draw < cfg.split_fractions[1] + cfg.split_fractions[2]:
        return "test"
    return "train"


# ----------------------------------------------------------------------------- interim files

def write_interim(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8")


def read_interim(path: Path):
    """Streams records back from an interim file. One document at a time, by design."""
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def slim(rec: dict, text: str, key: str) -> dict:
    """The subset of a record carried between stages. Dropping url/title keeps the interim files
    (and the per-record dicts) small; host and source_dir are kept because the report's provenance
    statistics need them."""
    return {
        "text": text,
        "content_sha256": key,
        "host": rec.get("host"),
        "source_dir": rec.get("source_dir"),
        "source_type": rec.get("source_type", "manual"),
    }


def gram_hash(gram) -> int:
    """8-byte digest of one 13-gram, as an int.

    The candidate index holds millions of these; storing the n-grams themselves (~90 bytes of
    Devanagari each) instead of an 8-byte digest would cost hundreds of megabytes on a machine that
    does not have them to spare.
    """
    return int.from_bytes(
        hashlib.blake2b(" ".join(gram).encode("utf-8"), digest_size=8).digest(), "big"
    )


def para_hash(text: str) -> int:
    """8-byte paragraph digest as an int, for the cross-document paragraph counter.

    8 bytes rather than a 64-character hex string because this counter holds one entry per DISTINCT
    paragraph in the corpus -- hundreds of thousands of them -- and the difference between an int key
    and a hex-string key is tens of megabytes of resident memory for no benefit.
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


class Holdout:
    """Decides which documents are held out, honouring both the fractions and the absolute caps.

    `split_of` alone decides membership by hash, which is order-independent and reproducible -- but
    it cannot enforce a cap, because a hash cannot know how many documents came before it. This class
    adds exactly that: it counts val/test assignments in stream order and sends the overflow to train.

    That makes the result order-dependent, so every pass that needs the same answer must replay the
    same counting over the same stream. `snapshot()`/`restore()` exist for that: the manual pass runs
    first, then the downloaded passes resume from the counts the manual pass finished with.
    """

    def __init__(self, cfg: CleanConfig, max_val: int | None = None, max_test: int | None = None):
        self.cfg = cfg
        self.max_val = cfg.max_val_docs if max_val is None else max(max_val, 0)
        self.max_test = cfg.max_test_docs if max_test is None else max(max_test, 0)
        self.n_val = 0
        self.n_test = 0

    def place(self, key: str) -> str:
        where = split_of(key, self.cfg)
        if where == "val":
            if self.n_val >= self.max_val:
                return "train"
            self.n_val += 1
            return "val"
        if where == "test":
            if self.n_test >= self.max_test:
                return "train"
            self.n_test += 1
            return "test"
        return "train"

    def full(self) -> bool:
        """True once both caps are filled -- the point where scanning for candidates can stop."""
        return self.n_val >= self.max_val and self.n_test >= self.max_test

    def snapshot(self) -> tuple[int, int]:
        return self.n_val, self.n_test

    def restore(self, snap: tuple[int, int]) -> None:
        self.n_val, self.n_test = snap


def save_state(interim: Path, state: dict) -> None:
    """Records the funnel and stage counters next to the interim files, so --from-interim can rebuild
    stats.json without rerunning the cascade that produced those numbers."""
    interim.mkdir(parents=True, exist_ok=True)
    (interim / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


def load_state(interim: Path) -> dict:
    path = interim / "state.json"
    if not path.exists():
        raise RuntimeError(
            f"--from-interim needs {path}, written when the cascade reaches stage 7. "
            "Run without --from-interim first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- orchestrator

def run(
    lang_dir: Path | None = None,
    report_dir: Path | None = None,
    workers: int | None = None,
    limit: int | None = None,
    config: CleanConfig | None = None,
    include_downloaded: bool = True,
    batch: int = BATCH_DOCS,
    from_interim: bool = False,
) -> dict:
    """Runs the full cascade for Hindi and writes splits, stats.json, and figures.

    Memory-bounded: document text is streamed through gzipped interim files under
    <lang_dir>/data/interim/ between stages, and only bounded state (hashes, signatures, cluster
    labels, the 1% val/test slices) is ever held in RAM.

    `lang_dir`/`report_dir` are overridable so scripts/reproduce.sh can run against a sample
    directory without touching the real corpus or the graded figures.
    """
    cfg = config or DEFAULT
    lang_dir = lang_dir or REPO_ROOT / LANGUAGE
    report_dir = report_dir or REPO_ROOT / "report" / "figures"
    workers = workers or default_workers()
    interim = lang_dir / "data" / "interim"
    t_start = time.time()

    logger.info("=== cleaning %s (workers=%d, batch=%d%s) ===", LANGUAGE, workers, batch,
                ", resuming from interim" if from_interim else "")

    funnel: dict[str, int] = {}
    rejected: Counter[str] = Counter()
    by_stage: Counter[str] = Counter()
    repairs: Counter[str] = Counter()

    if from_interim:
        # Resume from whichever interim stage is available, cheapest first. stage7 holds every
        # surviving manual document post-safety; stage6 holds them post-near-dedup, before stage 5.
        st = load_state(interim)
        stage6 = interim / "stage6.jsonl.gz"
        stage7 = interim / "stage7.jsonl.gz"
        funnel.update(st["funnel"])
        rejected.update(st["rejected"])
        by_stage.update(st["by_stage"])
        repairs.update(st["repairs"])
        resume_at = st.get("stage", "stage7")

        if resume_at == "stage7" and stage7.exists():
            logger.info("resuming from %s: %d manual documents, %d words",
                        stage7.name, st["n_after_safety"], st["manual_words"])
            return _finish(
                lang_dir, report_dir, cfg, interim, stage7, funnel, rejected, by_stage, repairs,
                st["manual_words"], include_downloaded, t_start,
                dict(paras_removed=st["paras_removed"], docs_touched=st["docs_touched"],
                     n_exact=st["n_exact"], n_near=st["n_near"], n_clusters=st["n_clusters"],
                     n_topic_dropped=st["n_topic_dropped"], ad_clusters=set(st["ad_clusters"]),
                     model=safety.ClusterModel(n_clusters=st["n_model_clusters"]),
                     ppl_thr=st["ppl_thr"] if st["ppl_thr"] is not None else float("inf"),
                     coh_thr=st["coh_thr"]))

        if stage6.exists():
            logger.info("resuming from %s: redoing stage 5 (clustering, topics, safety)",
                        stage6.name)
            return _stage5(lang_dir, report_dir, cfg, interim, stage6, funnel, rejected, by_stage,
                           repairs, st["n_after_near"], include_downloaded, t_start,
                           dict(paras_removed=st["paras_removed"], docs_touched=st["docs_touched"],
                                n_exact=st["n_exact"], n_near=st["n_near"],
                                n_clusters=st["n_clusters"],
                                ppl_thr=st["ppl_thr"] if st["ppl_thr"] is not None else float("inf"),
                                coh_thr=st["coh_thr"]))

        raise RuntimeError(
            f"--from-interim found neither {stage7} nor {stage6}; run without it first"
        )

    # ---- seeds and learned filters. A dedicated pass so the corpus itself is never resident.
    seed_wiki, seed_neg, n_seen = collect_seeds(lang_dir, limit=limit)
    scorer, ppl_thr, qmodel, coh_thr = fit_models(
        [{"text": t, "source_dir": "wiki"} for t in seed_wiki]
        + [{"text": t, "source_dir": "fast"} for t in seed_neg],
        cfg,
        corpus_size=n_seen,
    )
    del seed_wiki, seed_neg

    # ---- stages 1-3, batched: normalise, language, repair, rules, ppl/quality/coherence
    stage3 = interim / "stage3.jsonl.gz"
    n_in = n_kept = 0
    t0 = time.time()
    with mp.Pool(workers, initializer=_init,
                 initargs=(cfg, scorer, ppl_thr, qmodel, coh_thr)) as pool, \
            write_interim(stage3) as out:
        for chunk in batched(load_manual_stream(lang_dir, limit=limit), batch):
            texts = [r["text"] for r in chunk]
            for v in pool.imap_unordered(_one, list(enumerate(texts)), chunksize=50):
                n_in += 1
                for a in v.repairs:
                    repairs[a] += 1
                if v.keep:
                    rec = slim(chunk[v.idx], v.text, dedup.key(v.text))
                    rec["ppl"] = round(v.ppl, 3)
                    rec["quality"] = round(v.quality, 4)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_kept += 1
                else:
                    by_stage[v.stage] += 1
                    for r in v.reasons:
                        rejected[r] += 1
            del texts, chunk
            if n_in % 50_000 < batch:
                logger.info("  %d documents (%d kept, %.0f docs/s)",
                            n_in, n_kept, n_in / max(time.time() - t0, 1e-9))
    del scorer, qmodel

    funnel["collected"] = n_in
    funnel["language"] = n_in - by_stage["language"]
    funnel["repaired_and_rules"] = funnel["language"] - by_stage["repaired_and_rules"]
    funnel["quality_scored"] = n_kept
    logger.info("stages 1-3: %d/%d survived (%.1f%%) in %.0fs",
                n_kept, n_in, 100 * n_kept / max(n_in, 1), time.time() - t0)
    if not n_kept:
        logger.warning("nothing survived the per-document stages; stopping")
        return {"language": LANGUAGE, "status": "no_survivors", "funnel": funnel}

    # ---- stage 4a: cross-document paragraph dedup (two streaming passes)
    logger.info("stage 4: paragraph dedup...")
    counts: Counter[int] = Counter()
    for rec in read_interim(stage3):
        for h in {para_hash(p) for p in rules.paras(rec["text"])
                  if len(p) >= cfg.para_dup_min_chars}:
            counts[h] += 1
    boilerplate = {h for h, c in counts.items() if c >= cfg.para_dup_min_docs}
    del counts
    logger.info("  %d paragraphs recur in >=%d documents", len(boilerplate), cfg.para_dup_min_docs)

    stage4 = interim / "stage4.jsonl.gz"
    paras_removed = docs_touched = n_after_para = 0
    with write_interim(stage4) as out:
        for rec in read_interim(stage3):
            kept, dropped = [], 0
            for p in rules.paras(rec["text"]):
                if len(p) >= cfg.para_dup_min_chars and para_hash(p) in boilerplate:
                    dropped += 1
                    continue
                kept.append(p)
            text = "\n\n".join(kept)
            paras_removed += dropped
            docs_touched += 1 if dropped else 0
            # Paragraph dedup can leave a document too short to be worth keeping.
            if len(text) < cfg.min_chars:
                rejected["empty_after_paragraph_dedup"] += 1
                continue
            rec["text"] = text
            rec["content_sha256"] = dedup.key(text)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_after_para += 1
    del boilerplate
    funnel["paragraph_deduped"] = n_after_para
    logger.info("  removed %d paragraphs across %d documents; %d remain",
                paras_removed, docs_touched, n_after_para)

    # ---- stage 4b: exact dedup (streaming, 16-byte keys)
    stage5 = interim / "stage5.jsonl.gz"
    seen: set[bytes] = set()
    n_exact = n_after_exact = 0
    with write_interim(stage5) as out:
        for rec in read_interim(stage4):
            k = bytes.fromhex(rec["content_sha256"][:32])   # half of SHA-256 is ample here
            if k in seen:
                n_exact += 1
                continue
            seen.add(k)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_after_exact += 1
    del seen
    funnel["exact_deduped"] = n_after_exact
    logger.info("  exact dedup removed %d; %d remain", n_exact, n_after_exact)

    # ---- stage 4c: near dedup. Signatures stay in RAM (128 uint64 per document, ~1 KB); text does not.
    logger.info("  near dedup (MinHash-LSH) over %d documents...", n_after_exact)
    t0 = time.time()
    hasher = MinHasher(num_perm=cfg.num_perm, seed=cfg.seed)
    lsh = LSHIndex(num_perm=cfg.num_perm, b=cfg.lsh_b, r=cfg.lsh_r)
    sigs: list = []
    for i, rec in enumerate(read_interim(stage5)):
        sig = hasher.signature(word_shingles(rec["text"], k=cfg.shingle_k))
        sigs.append(sig)
        lsh.insert(str(i), sig)
    uf = UnionFind(range(len(sigs)))
    for a, b in lsh.iter_pairs():
        ia, ib = int(a), int(b)
        if estimate_jaccard(sigs[ia], sigs[ib]) >= cfg.near_dup_threshold:
            uf.union(ia, ib)
    reps: dict[int, int] = {}
    sizes: Counter[int] = Counter()
    for i in range(len(sigs)):
        root = uf.find(i)
        sizes[root] += 1
        if root not in reps or i < reps[root]:
            reps[root] = i
    keep_near = set(reps.values())
    n_clusters = sum(1 for s in sizes.values() if s > 1)
    n_near = len(sigs) - len(keep_near)
    del sigs, lsh, uf, reps, sizes

    stage6 = interim / "stage6.jsonl.gz"
    with write_interim(stage6) as out:
        for i, rec in enumerate(read_interim(stage5)):
            if i in keep_near:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_after_near = len(keep_near)
    del keep_near
    funnel["near_deduped"] = n_after_near
    logger.info("  near dedup removed %d in %d clusters (%.0fs); %d remain",
                n_near, n_clusters, time.time() - t0, n_after_near)

    save_state(interim, {
        "stage": "stage6", "funnel": funnel, "rejected": dict(rejected),
        "by_stage": dict(by_stage), "repairs": dict(repairs), "n_after_near": n_after_near,
        "paras_removed": paras_removed, "docs_touched": docs_touched, "n_exact": n_exact,
        "n_near": n_near, "n_clusters": n_clusters,
        "ppl_thr": ppl_thr if ppl_thr != float("inf") else None, "coh_thr": coh_thr,
    })

    return _stage5(lang_dir, report_dir, cfg, interim, stage6, funnel, rejected, by_stage,
                   repairs, n_after_near, include_downloaded, t_start,
                   dict(paras_removed=paras_removed, docs_touched=docs_touched, n_exact=n_exact,
                        n_near=n_near, n_clusters=n_clusters, ppl_thr=ppl_thr, coh_thr=coh_thr))


def _stage5(lang_dir: Path, report_dir: Path, cfg: CleanConfig, interim: Path, stage6: Path,
            funnel: dict, rejected: Counter, by_stage: Counter, repairs: Counter,
            n_after_near: int, include_downloaded: bool, t_start: float, st: dict) -> dict:
    """Stage 5 -- semantic clustering, topic filtering, safety -- then the split phase.

    Separated from run() so `--from-interim` can redo it against an existing stage6 without repeating
    stages 1-4, and so the resumed path and the full path are literally the same code.
    """
    paras_removed, docs_touched = st["paras_removed"], st["docs_touched"]
    n_exact, n_near, n_clusters = st["n_exact"], st["n_near"], st["n_clusters"]
    ppl_thr, coh_thr = st["ppl_thr"], st["coh_thr"]

    # ---- stage 5: semantic clustering + topic filtering + safety, all streaming
    logger.info("stage 5: semantic clustering (fitting on <=%d sampled documents)...",
                cfg.cluster_sample)
    sample = [rec["text"] for _, rec in zip(range(cfg.cluster_sample), read_interim(stage6))]
    model = safety.fit_clusters(sample, config=cfg)
    del sample

    labels: list[int] = []
    ad_sum: dict[int, float] = {}
    ad_n: Counter[int] = Counter()
    for chunk in batched(read_interim(stage6), 20_000):
        for lab, rec in zip(safety.assign(model, [r["text"] for r in chunk]), chunk):
            labels.append(lab)
            ad_sum[lab] = ad_sum.get(lab, 0.0) + safety.ad_rate(rec["text"])
            ad_n[lab] += 1
    rates = {lab: ad_sum[lab] / ad_n[lab] for lab in ad_sum}
    ad_clusters = safety.ad_heavy(rates, config=cfg)
    logger.info("  %d clusters, %d ad-heavy%s", model.n_clusters, len(ad_clusters),
                (" -> downsampling to %.0f%%" % (100 * cfg.ad_cluster_keep_fraction))
                if ad_clusters else "")

    # ---- final cleaning pass: topic downsample, then safety. Split assignment happens later, at
    # write time, so stage7 holds EVERY surviving manual document -- which is what makes
    # --from-interim able to resume from here and re-split without redoing the cascade.
    stage7 = interim / "stage7.jsonl.gz"
    stride = max(int(1 / cfg.ad_cluster_keep_fraction), 1) if cfg.ad_cluster_keep_fraction else 1
    seen_in_cluster: Counter[int] = Counter()
    n_topic_dropped = n_unsafe = n_after_safety = 0
    manual_words = 0
    with write_interim(stage7) as out:
        for i, rec in enumerate(read_interim(stage6)):
            lab = labels[i]
            if lab in ad_clusters:
                # Deterministic: every stride-th document of the cluster, by position.
                position = seen_in_cluster[lab]
                seen_in_cluster[lab] += 1
                if position % stride:
                    n_topic_dropped += 1
                    continue
            ok, redacted, _ = safety.check(rec["text"], config=cfg)
            if not ok:
                n_unsafe += 1
                rejected["safety_blocklist"] += 1
                continue
            rec["text"] = redacted
            n_after_safety += 1
            manual_words += n_words(redacted)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    del labels
    funnel["topic_filtered"] = n_after_near - n_topic_dropped
    funnel["safety_checked"] = n_after_safety
    logger.info("  topic filtering dropped %d, safety dropped %d; %d remain",
                n_topic_dropped, n_unsafe, n_after_safety)
    logger.info("manual corpus after cleaning: %d documents, %d words", n_after_safety, manual_words)

    save_state(interim, {
        "funnel": funnel, "rejected": dict(rejected), "by_stage": dict(by_stage),
        "repairs": dict(repairs), "manual_words": manual_words,
        "n_after_safety": n_after_safety, "n_topic_dropped": n_topic_dropped,
        "paras_removed": paras_removed, "docs_touched": docs_touched, "n_exact": n_exact,
        "n_near": n_near, "n_clusters": n_clusters, "n_model_clusters": model.n_clusters,
        "ad_clusters": sorted(ad_clusters), "ppl_thr": ppl_thr if ppl_thr != float("inf") else None,
        "coh_thr": coh_thr,
    })

    return _finish(lang_dir, report_dir, cfg, interim, stage7, funnel, rejected, by_stage,
                   repairs, manual_words, include_downloaded, t_start,
                   dict(paras_removed=paras_removed, docs_touched=docs_touched, n_exact=n_exact,
                        n_near=n_near, n_clusters=n_clusters, n_topic_dropped=n_topic_dropped,
                        ad_clusters=ad_clusters, model=model, ppl_thr=ppl_thr, coh_thr=coh_thr))


def _finish(lang_dir: Path, report_dir: Path, cfg: CleanConfig, interim: Path, stage7: Path,
            funnel: dict, rejected: Counter, by_stage: Counter, repairs: Counter,
            manual_words: int, include_downloaded: bool, t_start: float, st: dict) -> dict:
    """Holdout selection, decontamination, split writing, stats and figures.

    Split out so `--from-interim` can run exactly this phase against an existing stage7 without
    repeating the cascade -- and so the resumed path and the full path cannot drift apart, which they
    would if resume were a second copy of this logic.
    """
    paras_removed, docs_touched = st["paras_removed"], st["docs_touched"]
    n_exact, n_near, n_clusters = st["n_exact"], st["n_near"], st["n_clusters"]
    n_topic_dropped, ad_clusters = st["n_topic_dropped"], st["ad_clusters"]
    model, ppl_thr, coh_thr = st["model"], st["ppl_thr"], st["coh_thr"]

    # ---- held-out selection, decontaminated on the HELD-OUT side (see module notes).
    #
    # Candidates are oversampled, then any candidate sharing a 13-gram with a training document is
    # discarded and replaced from the clean remainder. Train loses nothing; val/test are still
    # 13-gram-disjoint from train.
    n_val_cand = int(cfg.max_val_docs * cfg.holdout_oversample)
    n_test_cand = int(cfg.max_test_docs * cfg.holdout_oversample)
    manual_val_cand = int(n_val_cand * cfg.holdout_manual_fraction)
    manual_test_cand = int(n_test_cand * cfg.holdout_manual_fraction)

    manual_pick = Holdout(cfg, max_val=manual_val_cand, max_test=manual_test_cand)
    cand_val: list[dict] = []
    cand_test: list[dict] = []
    cand_keys: set[str] = set()
    for rec in read_interim(stage7):
        where = manual_pick.place(rec["content_sha256"])
        if where == "val":
            cand_val.append(rec)
            cand_keys.add(rec["content_sha256"])
        elif where == "test":
            cand_test.append(rec)
            cand_keys.add(rec["content_sha256"])
    logger.info("held-out candidates from the manual side: %d val + %d test",
                len(cand_val), len(cand_test))

    dl_dir = lang_dir / "data" / "raw" / "downloaded"
    budget = int(manual_words * (1 - cfg.target_manual_fraction) / cfg.target_manual_fraction)
    stream_downloaded = include_downloaded and dl_dir.exists()
    dl_docs = 0
    if stream_downloaded:
        logger.info("downloaded side: budget %d words WRITTEN (manual %d at target manual share "
                    "%.0f%%)", budget, manual_words, 100 * cfg.target_manual_fraction)
        dl_pick = Holdout(cfg, max_val=n_val_cand - len(cand_val),
                          max_test=n_test_cand - len(cand_test))
        t0 = time.time()
        for rec in read_shards(dl_dir):
            if dl_docs >= HOLDOUT_WINDOW or dl_pick.full():
                break
            key = rec.get("content_sha256") or dedup.key(rec["text"])
            dl_docs += 1
            where = dl_pick.place(key)
            if where != "train":
                slimmed = slim(rec, rec["text"], key) | {"source_type": "downloaded"}
                (cand_val if where == "val" else cand_test).append(slimmed)
                cand_keys.add(key)
        logger.info("  candidate scan: %d documents in %.0fs -> %d val + %d test candidates",
                    dl_docs, time.time() - t0, len(cand_val), len(cand_test))

    # ---- index the candidates, then mark any that a training document collides with
    logger.info("decontaminating the held-out candidates against train (13-gram)...")
    owners: dict[int, list[int]] = {}
    for i, rec in enumerate(cand_val + cand_test):
        for g in word_ngrams(rec["text"], 13):
            owners.setdefault(gram_hash(g), []).append(i)
    logger.info("  candidate index: %d distinct 13-grams from %d candidates",
                len(owners), len(cand_val) + len(cand_test))

    dirty: set[int] = set()
    splits_dir = lang_dir / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    writers = {name: SplitWriter(splits_dir / name) for name in ("train", "val", "test")}

    manual_train_words = manual_train_docs = 0
    for rec in read_interim(stage7):
        if rec["content_sha256"] in cand_keys:
            continue                          # candidates are never training data
        for g in word_ngrams(rec["text"], 13):
            hit = owners.get(gram_hash(g))
            if hit:
                dirty.update(hit)
        writers["train"].write(rec)
        manual_train_docs += 1
        manual_train_words += n_words(rec["text"])
    logger.info("  manual train: %d documents written, 0 dropped (%d candidates marked so far)",
                manual_train_docs, len(dirty))

    downloaded_words = 0
    dl_train_docs = 0
    if stream_downloaded:
        t0 = time.time()
        for rec in read_shards(dl_dir):
            if downloaded_words >= budget:    # WRITTEN words, so no loss is silently absorbed
                break
            key = rec.get("content_sha256") or dedup.key(rec["text"])
            if key in cand_keys:
                continue
            for g in word_ngrams(rec["text"], 13):
                hit = owners.get(gram_hash(g))
                if hit:
                    dirty.update(hit)
            rec["source_type"] = "downloaded"
            writers["train"].write(rec)
            dl_train_docs += 1
            downloaded_words += n_words(rec["text"])
        logger.info("  downloaded train: %d documents, %d words (%.1f%% of budget) in %.0fs",
                    dl_train_docs, downloaded_words, 100 * downloaded_words / max(budget, 1),
                    time.time() - t0)
    del owners

    # ---- keep the clean candidates, up to the caps
    val = [r for i, r in enumerate(cand_val) if i not in dirty][: cfg.max_val_docs]
    offset = len(cand_val)
    test = [r for i, r in enumerate(cand_test, start=offset) if i not in dirty][: cfg.max_test_docs]
    n_dirty_val = sum(1 for i in range(len(cand_val)) if i in dirty)
    n_dirty_test = sum(1 for i in range(offset, offset + len(cand_test)) if i in dirty)
    writers["val"].write_all(val)
    writers["test"].write_all(test)
    logger.info("  held out %d val + %d test after discarding %d + %d contaminated candidates "
                "(%.1f%% of candidates collided with train)",
                len(val), len(test), n_dirty_val, n_dirty_test,
                100 * len(dirty) / max(len(cand_val) + len(cand_test), 1))
    if len(val) < cfg.max_val_docs or len(test) < cfg.max_test_docs:
        logger.warning("held-out set is smaller than the caps: raise holdout_oversample if this "
                       "matters (val %d/%d, test %d/%d)",
                       len(val), cfg.max_val_docs, len(test), cfg.max_test_docs)
    manual_contaminated = 0                   # train is never dropped now, by construction
    dl_contaminated = n_dirty_val + n_dirty_test

    split_counts = {name: w.n for name, w in writers.items()}
    for w in writers.values():
        w.close()
    funnel["decontaminated"] = sum(split_counts.values())
    logger.info("splits: train=%d val=%d test=%d",
                split_counts["train"], split_counts["val"], split_counts["test"])

    # ---- stats
    manual_final = manual_train_words + sum(
        n_words(r["text"]) for r in val + test if r.get("source_type") != "downloaded"
    )
    downloaded_final = downloaded_words + sum(
        n_words(r["text"]) for r in val + test if r.get("source_type") == "downloaded"
    )
    dl_docs_written = dl_train_docs
    holdout_counts = {"val": len(val), "test": len(test),
                      "val_manual": sum(1 for r in val if r.get("source_type") != "downloaded"),
                      "test_manual": sum(1 for r in test if r.get("source_type") != "downloaded")}
    total_final = manual_final + downloaded_final
    # Per-host statistics come from the manual side: it has hundreds of hosts and its provenance is
    # the interesting part, while the downloaded side is one source by construction.
    corpus = compute_corpus_stats([
        DocumentRecord(text=r["text"], source_type=r.get("source_type", "manual"),
                       host=r.get("host") or "unknown")
        for r in val + test
    ])
    stats = {
        "language": LANGUAGE,
        "elapsed_s": round(time.time() - t_start, 1),
        "funnel": {k: funnel[k] for k in STAGES if k in funnel},
        "rejected_by_rule": dict(rejected.most_common()),
        "rejected_by_stage": dict(by_stage),
        "repairs_applied": dict(repairs.most_common()),
        "thresholds": {
            "perplexity": round(ppl_thr, 2) if ppl_thr != float("inf") else None,
            "quality_min": cfg.quality_min_score,
            "coherence_min": round(coh_thr, 5),
            "lid_min_confidence": cfg.lid_min_confidence,
        },
        "dedup": {"paragraphs_removed": paras_removed, "documents_touched": docs_touched,
                  "exact_removed": n_exact, "near_removed": n_near, "near_clusters": n_clusters},
        "clusters": {"n": model.n_clusters, "ad_heavy": sorted(ad_clusters),
                     "topic_dropped": n_topic_dropped},
        "words": {"manual": manual_final, "downloaded": downloaded_final, "total": total_final,
                  "manual_fraction": round(manual_final / max(total_final, 1), 4)},
        "downloaded": {"documents_streamed": dl_docs, "words_budget": budget,
                       "dropped_contaminated": dl_contaminated},
        "decontamination": {"direction": "held-out side (train keeps every document)",
                            "train_documents_dropped": 0,
                            "candidates_discarded": dl_contaminated,
                            "candidates_offered": n_val_cand + n_test_cand,
                            "holdout": holdout_counts},
        "splits": split_counts,
        "corpus": corpus.__dict__ if hasattr(corpus, "__dict__") else {},
    }
    (lang_dir / "data" / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _figures(report_dir, funnel, stats, val + test)
    share = stats["words"]["manual_fraction"]
    logger.info("manual share of the final corpus: %.1f%% (floor is 20%%)", 100 * share)
    if share < MANUAL_FLOOR:
        # Raised, not warned. A corpus under the floor does not satisfy the assignment, and an
        # earlier run produced exactly that (15.0%) while logging it as an ordinary line that was
        # easy to read past. stats.json is already written above, so the funnel is available for
        # diagnosis; what must not happen is the run reporting success.
        raise RuntimeError(
            f"manual share {100 * share:.1f}% is below the required {100 * MANUAL_FLOOR:.0f}% floor "
            f"({stats['words']['manual']:,} manual of {stats['words']['total']:,} words). "
            f"Raise target_manual_fraction in {LANGUAGE}/clean/config.py and re-run with "
            f"--from-interim."
        )
    logger.info("done in %.0fs -> %s", time.time() - t_start, lang_dir / "data" / "stats.json")
    return stats


def _figures(report_dir: Path, funnel: dict[str, int], stats: dict, sample: list[dict]) -> None:
    """Writes the funnel, length-distribution, per-source and manual-vs-downloaded figures.

    Never fatal: a missing plotting backend must not invalidate a completed multi-hour corpus build,
    and every number behind these figures is in stats.json, so they can be redrawn. The exception is
    logged with its type -- an earlier version called plot_funnel_chart with the wrong signature and
    the only symptom was an empty figures directory.

    The length histogram and per-source chart are drawn from the held-out splits, which are the only
    documents still in memory at this point and are a uniform 2% sample of the corpus by construction.
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

        if sample:
            plot_length_histogram([n_words(r["text"]) for r in sample],
                                  report_dir / f"{LANGUAGE}_length_hist.png", LANGUAGE)
            per_source: Counter[str] = Counter()
            for r in sample:
                per_source[r.get("source_dir") or r.get("source_type") or "unknown"] += 1
            plot_domain_distribution(dict(per_source),
                                     report_dir / f"{LANGUAGE}_sources.png", LANGUAGE)

        plot_manual_vs_downloaded(stats["words"]["manual"], stats["words"]["downloaded"],
                                  report_dir / f"{LANGUAGE}_manual_vs_downloaded.png", LANGUAGE)
        logger.info("figures -> %s", report_dir)
    except Exception as e:                        # noqa: BLE001
        logger.warning("figure generation skipped: %s: %s", type(e).__name__, e)


def main():
    p = argparse.ArgumentParser(description="Clean the Hindi corpus (Yi 5-stage cascade).")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch", type=int, default=BATCH_DOCS,
                   help="documents held in memory per parallel batch (memory/throughput trade-off)")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N manual documents (for a smoke run)")
    p.add_argument("--lang-dir", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--no-downloaded", action="store_true",
                   help="clean the manual side only; skip the downloaded pass-through")
    p.add_argument("--from-interim", action="store_true",
                   help="skip the cascade and redo only the split/decontamination phase from "
                        "data/interim/stage7.jsonl.gz (written by a previous run)")
    args = p.parse_args()
    out = run(lang_dir=args.lang_dir, report_dir=args.report_dir, workers=args.workers,
              limit=args.limit, include_downloaded=not args.no_downloaded, batch=args.batch,
              from_interim=args.from_interim)
    print(json.dumps({k: v for k, v in out.items() if k != "corpus"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
