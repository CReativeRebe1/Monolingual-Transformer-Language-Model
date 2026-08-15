"""
High-throughput crawler CLI.

Differs from scripts/run_crawl.py in one decisive way: that crawler allows a single in-flight
request per host, so a domain is capped near 0.67 req/s and total throughput is bounded by the
number of live domains. This one allows `--per-domain` concurrent requests per host spaced
`--delay` apart at their *starts*, which lifts the per-host ceiling roughly an order of magnitude
while still honouring robots.txt and any site-declared Crawl-delay.

Writes JSONL.gz shards in the same schema scripts/build_corpus.py already consumes, into
<lang>/data/raw/fast/, and keeps its own SQLite state so it can run alongside (or instead of) the
original crawler without touching that frontier.

Usage:
    python scripts/fast_crawl.py --lang hindi --target-words 60000000
    python scripts/fast_crawl.py --lang nepali --concurrency 96 --per-domain 6
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.crawl.fast import CrawlConfig, run_fast_crawl  # noqa: E402
from lma.crawl.robots import USER_AGENT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.fast_crawl")


def load_seeds(language: str) -> tuple[list[str], dict[str, str]]:
    """Reuses each language's existing seed_domains.py -- one source of truth for both crawlers."""
    mod = importlib.import_module(f"{language}.data.seed_domains")
    seeds = [f"https://{d}/" for d in mod.ALL_DOMAINS]
    prefixes = {}
    for domain, substring in getattr(mod, "DOMAIN_URL_FILTERS", {}).items():
        host = domain[4:] if domain.startswith("www.") else domain
        prefixes[host] = substring
    return seeds, prefixes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="High-throughput resumable crawler.")
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument("--concurrency", type=int, default=128, help="global concurrent requests")
    p.add_argument("--per-domain", type=int, default=8, help="concurrent requests per host")
    p.add_argument("--delay", type=float, default=0.35, help="min seconds between host request starts")
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--max-bytes", type=int, default=3_000_000)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--max-pages", type=int, default=1_000_000, help="fetches this run; 0 = unlimited")
    p.add_argument("--max-urls", type=int, default=5_000_000, help="frontier cap; 0 = unlimited")
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-chars", type=int, default=500)
    p.add_argument("--min-hindi-ratio", type=float, default=0.55, dest="min_ratio",
                   help="minimum Devanagari share of alphabetic characters")
    p.add_argument("--min-markers", type=int, default=2,
                   help="minimum distinct target-language function words")
    p.add_argument("--target-words", type=int, default=0,
                   help="persistent corpus target across runs; 0 = disabled")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--state", type=Path, default=None)
    return p


def validate(a: argparse.Namespace) -> None:
    if not 1 <= a.concurrency <= 1000:
        raise SystemExit("--concurrency must be 1..1000")
    if not 1 <= a.per_domain <= 32:
        raise SystemExit("--per-domain must be 1..32")
    if a.per_domain > a.concurrency:
        raise SystemExit("--per-domain cannot exceed --concurrency")
    if a.delay < 0:
        raise SystemExit("--delay must be >= 0")
    if a.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if a.max_bytes < 1024:
        raise SystemExit("--max-bytes must be >= 1024")
    if not 0.0 <= a.min_ratio <= 1.0:
        raise SystemExit("--min-hindi-ratio must be 0..1")
    for name in ("max_attempts", "max_pages", "max_urls", "max_depth", "min_chars",
                 "min_markers", "target_words", "log_every"):
        if getattr(a, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")


def main() -> None:
    args = build_parser().parse_args()
    validate(args)

    seeds, prefixes = load_seeds(args.lang)
    cfg = CrawlConfig(
        language=args.lang,
        seeds=seeds,
        host_prefixes=prefixes,
        out_dir=args.out_dir or REPO_ROOT / args.lang / "data" / "raw" / "fast",
        state_path=args.state or REPO_ROOT / args.lang / "data" / "fast_state.sqlite3",
        concurrency=args.concurrency,
        per_domain=args.per_domain,
        delay=args.delay,
        timeout=args.timeout,
        max_bytes=args.max_bytes,
        max_attempts=args.max_attempts,
        max_pages=args.max_pages,
        max_urls=args.max_urls,
        max_depth=args.max_depth,
        min_chars=args.min_chars,
        min_ratio=args.min_ratio,
        min_markers=args.min_markers,
        target_words=args.target_words,
        user_agent=USER_AGENT,
        log_every=args.log_every,
    )
    logger.info("seeds=%d concurrency=%d per_domain=%d delay=%.2fs",
                len(seeds), cfg.concurrency, cfg.per_domain, cfg.delay)

    async def runner():
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(run_fast_crawl(cfg))
        # Ctrl-C cancels in-flight work; startup recovery resets those rows to pending, so a
        # signal never loses frontier progress (CorpusStore.__init__ requeues 'fetching' rows).
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, task.cancel)
            except NotImplementedError:  # Windows: add_signal_handler is unsupported
                signal.signal(sig, lambda *_: task.cancel())
        try:
            return await task
        except asyncio.CancelledError:
            logger.info("interrupted; frontier state preserved")
            return {}

    summary = asyncio.run(runner())
    if summary:
        print(summary)


if __name__ == "__main__":
    main()
