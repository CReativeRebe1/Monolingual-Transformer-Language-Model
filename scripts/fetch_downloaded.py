"""
Collect the "downloaded" share of the corpus from ai4bharat/IndicCorpV2, deduplicated against
everything we collected ourselves.

Target is ~600M tokens per language. That number is set by the assignment's >=20%-manual floor,
not picked arbitrarily: with ~207.6M raw manual Hindi tokens the downloaded side may reach 828M
before the floor breaks, and with ~222.0M Nepali it may reach 888M. 600M leaves real headroom,
which matters because the manual side shrinks during cleaning while this side does not.

NO CLEANING, NO PREPROCESSING. IndicCorpV2 is already a cleaned corpus, and this is the
"downloaded" 80% -- running our own filter cascade over it would waste time and would blur the
manual-vs-downloaded comparison the Phase-1 report has to make. The only gate applied here is the
duplicate check.

Dedup direction is deliberate and one-way: **duplicates are dropped from the downloaded side,
never from the manual side.** Manual tokens are the scarce, graded resource; downloaded tokens are
effectively unlimited, so discarding one costs nothing but discarding a manual one costs against
the >=20% requirement. The target counts only what is actually WRITTEN, so dedup losses are
absorbed by streaming further rather than by quietly shrinking the corpus.

The reference side is the RAW manual shards, not cleaned splits -- those do not exist yet. This is
deliberately conservative: a downloaded document matching a manual document that cleaning later
discards is dropped slightly unnecessarily, which is free given downloaded data is abundant, and it
avoids making this step wait on the cleaning pipeline.

IndicCorpV2 is public -- no HuggingFace token is required (verified by streaming it with no
credentials present). If it ever becomes gated, open_stream reports that with instructions.

Usage:
    python scripts/fetch_downloaded.py --lang hindi --benchmark      # measure speed first
    python scripts/fetch_downloaded.py --lang hindi
    python scripts/fetch_downloaded.py --lang nepali --target-tokens 600000000
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import logging
import multiprocessing as mp
import sqlite3
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.crawl.async_crawler import ShardWriter  # noqa: E402
from lma.crawl.downloaded_corpus import SOURCE_CONFIGS  # noqa: E402
from lma.dedup.index import DupIndex, hash_key  # noqa: E402
from lma.dedup.minhash import MinHasher, word_shingles  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.fetch_downloaded")

DEFAULT_TARGET_TOKENS = 600_000_000
MIN_CHARS = 200                     # matches the floor every other source already applies
MANUAL_DIRS = ("crawled", "fast", "wiki")
BENCHMARK_DOCS = 20_000
PARALLEL_MIN_DOCS = 5_000       # below this, pool startup costs more than it saves


# ----------------------------------------------------------------------------- manual reference

def iter_manual(lang_dir: Path):
    """Yields text from every manually-collected shard (crawler, fast crawler, wiki dumps)."""
    raw = lang_dir / "data" / "raw"
    for sub in MANUAL_DIRS:
        d = raw / sub
        if not d.exists():
            continue
        for shard in sorted(d.glob("*.jsonl.gz")):
            try:
                with gzip.open(shard, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = (rec.get("text") or "").strip()
                        if text:
                            yield text
            except (EOFError, OSError, zlib.error, UnicodeDecodeError) as e:
                # Same tolerance the corpus builder uses: a shard still being written, or one
                # with a torn gzip tail, must not abort an hours-long index build.
                logger.warning("shard %s ended early: %s", shard.name, e)


def _prep(text: str):
    """Pool worker: text -> (exact key, MinHash signature).

    Safe to run in a worker process only because the shingle hash is process-stable -- see
    lma/dedup/minhash.py's `_base_hash`. With the previous builtin-hash() implementation,
    signatures computed here would not have matched the parent's and every near-duplicate
    comparison would have silently returned garbage.
    """
    return hash_key(text), _WORKER["hasher"].signature(word_shingles(text, k=3))


_WORKER: dict = {}


def _init_worker():
    _WORKER["hasher"] = MinHasher()


def build_index(lang_dir: Path, workers: int | None = None, rebuild: bool = False) -> DupIndex:
    """Indexes the manual corpus so downloaded documents can be checked against it.

    Cached to disk because this is pure overhead on every rerun: the manual corpus does not change
    between pulls, and an interrupted run previously threw away several minutes of work. Profiled
    at ~1.08ms/doc single-threaded (hash_key and signature costing about the same), i.e. ~4 minutes
    for Hindi's ~230k documents before a single byte is streamed. Parallel + cached, a resumed run
    starts in seconds.
    """
    cache = lang_dir / "data" / "dup_index.npz"
    index = DupIndex()

    if cache.exists() and not rebuild:
        t0 = time.time()
        blob = np.load(cache, allow_pickle=False)
        keys, sigs = blob["keys"], blob["sigs"]
        for k, sig in zip(keys, sigs):
            index.add_precomputed(k.decode("ascii"), sig)
        logger.info("loaded cached index: %d docs in %.1fs (--rebuild-index to refresh)",
                    len(index), time.time() - t0)
        return index

    workers = workers or max(1, mp.cpu_count() - 1)
    t0 = time.time()
    keys: list[bytes] = []
    sigs: list[np.ndarray] = []

    # Peek before committing to a pool. On Windows each worker is a fresh interpreter costing
    # a second or two to spawn, so for a small manual corpus -- and for the test suite, which
    # loads this file by path and therefore cannot be re-imported by a spawned child at all --
    # sequential is both faster and the only thing that works.
    stream = iter_manual(lang_dir)
    head = list(itertools.islice(stream, PARALLEL_MIN_DOCS))
    parallel = workers > 1 and len(head) == PARALLEL_MIN_DOCS
    logger.info("building duplicate index over the manual corpus (%s)...",
                f"{workers} workers" if parallel else "sequential")

    def _record(i, pair):
        key, sig = pair
        index.add_precomputed(key, sig)
        keys.append(key.encode("ascii"))
        sigs.append(sig)
        if i % 20_000 == 0:
            dt = max(time.time() - t0, 1e-9)
            logger.info("  %d docs (%.0fs, %.0f docs/s)", i, dt, i / dt)

    if parallel:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            for i, pair in enumerate(
                pool.imap(_prep, itertools.chain(head, stream), chunksize=200), start=1
            ):
                _record(i, pair)
    else:
        _init_worker()
        for i, text in enumerate(head, start=1):
            _record(i, _prep(text))

    logger.info("manual index: %d documents in %.0fs", len(index), time.time() - t0)
    if keys:
        np.savez_compressed(cache, keys=np.array(keys, dtype="S64"), sigs=np.array(sigs))
        logger.info("cached index -> %s", cache.name)
    return index


# ----------------------------------------------------------------------------- resume state

class State:
    """Tracks tokens written and which documents were already stored, so an interrupted pull
    resumes instead of restarting. Same SQLite pattern as the crawler frontier."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS written (key TEXT PRIMARY KEY);"
            "CREATE TABLE IF NOT EXISTS progress (k TEXT PRIMARY KEY, v INTEGER NOT NULL);"
        )
        self.conn.commit()
        self._seen = {r[0] for r in self.conn.execute("SELECT key FROM written")}
        row = self.conn.execute("SELECT v FROM progress WHERE k='tokens'").fetchone()
        self.tokens = row[0] if row else 0

    def already(self, key: str) -> bool:
        return key in self._seen

    def record(self, key: str, tokens: int) -> None:
        self._seen.add(key)
        self.tokens += tokens
        self.conn.execute("INSERT OR IGNORE INTO written (key) VALUES (?)", (key,))

    def commit(self) -> None:
        self.conn.execute(
            "INSERT INTO progress (k, v) VALUES ('tokens', ?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (self.tokens,)
        )
        self.conn.commit()

    def close(self) -> None:
        self.commit()
        self.conn.close()


# ----------------------------------------------------------------------------- streaming

def check_auth() -> None:
    """No-op by design.

    IndicCorpV2 is **not** gated -- verified directly by streaming a document from
    split="hin_Deva" with no token present at all. An earlier version of this script exited here
    demanding `huggingface-cli login`, which would have blocked a pull that works fine
    unauthenticated. Auth is only surfaced now if the stream itself actually fails (see
    open_stream), so a credential problem is reported when it is real rather than assumed.
    """
    return None


def open_stream(language: str):
    """Opens the IndicCorpV2 stream for one language."""
    from datasets import load_dataset

    cfg = next(c for c in SOURCE_CONFIGS[language] if c["name"] == "indiccorp_v2")
    logger.info("streaming %s split=%s", cfg["repo_id"], cfg["split"])
    try:
        ds = load_dataset(cfg["repo_id"], cfg.get("config"), split=cfg["split"], streaming=True)
    except Exception as e:
        logger.error(
            "Could not open %s (split=%s): %s\n"
            "This dataset is normally public and needs no token. If the error mentions auth or "
            "403, either the dataset became gated (accept its terms at "
            "https://huggingface.co/datasets/ai4bharat/IndicCorpV2) or set HF_TOKEN / run "
            "`huggingface-cli login`. Never put a token in a file inside this repo.",
            cfg["repo_id"], cfg["split"], e,
        )
        raise
    return ds, cfg


def benchmark(language: str) -> None:
    """Streams a bounded sample and reports real throughput + an ETA, writing nothing."""
    check_auth()
    ds, cfg = open_stream(language)
    t0 = time.time()
    docs = chars = 0
    for ex in ds:
        text = (ex.get(cfg["text_field"]) or "").strip()
        if not text:
            continue
        docs += 1
        chars += len(text)
        if docs >= BENCHMARK_DOCS:
            break
    dt = max(time.time() - t0, 1e-9)
    mb = chars / 1e6
    words = chars / 5.5                       # rough Devanagari chars-per-word
    tok_per_s = (words * 1.31) / dt
    print(f"\nstreamed {docs:,} docs, {mb:.1f} MB, in {dt:.1f}s")
    print(f"  {mb/dt:.2f} MB/s   ~{tok_per_s:,.0f} tokens/s")
    if tok_per_s > 0:
        print(f"  ETA for 600M tokens: {600e6/tok_per_s/3600:.1f} hours")


def fetch(language: str, target_tokens: int, workers: int | None = None,
          rebuild_index: bool = False) -> dict:
    """Streams IndicCorpV2, drops anything duplicating the manual corpus, writes the rest."""
    check_auth()
    lang_dir = REPO_ROOT / language

    index = build_index(lang_dir, workers=workers, rebuild=rebuild_index)

    state = State(lang_dir / "data" / "downloaded_state.sqlite3")
    if state.tokens:
        logger.info("resuming: %d tokens already written", state.tokens)

    ds, cfg = open_stream(language)
    writer = ShardWriter(lang_dir / "data" / "raw" / "downloaded")

    streamed = dup_exact = dup_near = written = skipped = 0
    t0 = time.time()
    try:
        for ex in ds:
            if state.tokens >= target_tokens:
                logger.info("reached target of %d tokens", target_tokens)
                break
            text = (ex.get(cfg["text_field"]) or "").strip()
            streamed += 1
            if len(text) < MIN_CHARS:
                continue

            key = hash_key(text)
            if state.already(key):
                skipped += 1            # already written by an earlier, interrupted run
                continue

            is_dup, kind = index.seen(text)
            if is_dup:
                if kind == "exact":
                    dup_exact += 1
                else:
                    dup_near += 1
                continue

            n_tokens = len(text.split())
            writer.write({
                "url": None,
                "title": None,
                "text": text,                      # stored raw: no cleaning, by design
                "content_sha256": key,
                "host": cfg["repo_id"],
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_type": "downloaded",
            })
            state.record(key, n_tokens)
            written += 1

            if written % 5_000 == 0:
                state.commit()
                rate = state.tokens / max(time.time() - t0, 1e-9)
                logger.info(
                    "written=%d tokens=%d/%d dup(exact=%d near=%d) %.0f tok/s",
                    written, state.tokens, target_tokens, dup_exact, dup_near, rate,
                )
    finally:
        writer.close()
        tokens = state.tokens
        state.close()

    total_dup = dup_exact + dup_near
    summary = {
        "language": language,
        "streamed": streamed,
        "dropped_exact": dup_exact,
        "dropped_near": dup_near,
        "dropped_total": total_dup,
        "overlap_rate": round(total_dup / streamed, 4) if streamed else 0.0,
        "already_written_skipped": skipped,
        "written": written,
        "tokens_total": tokens,
        "elapsed_s": round(time.time() - t0, 1),
    }
    logger.info("done: %s", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description="Fetch downloaded-corpus data, deduped against manual.")
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    p.add_argument("--benchmark", action="store_true",
                   help="measure streaming throughput and exit, writing nothing")
    p.add_argument("--workers", type=int, default=None,
                   help="index-building workers (default: cpu_count() - 1)")
    p.add_argument("--rebuild-index", action="store_true",
                   help="ignore the cached manual index and rebuild it")
    args = p.parse_args()
    
    if args.benchmark:
        benchmark(args.lang)
        return
    print(json.dumps(fetch(args.lang, args.target_tokens, workers=args.workers,
                          rebuild_index=args.rebuild_index), indent=2))


if __name__ == "__main__":
    main()
