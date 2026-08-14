"""
CLI entry point for the manual-collection crawl of one language.

Usage:
    python scripts/run_crawl.py --lang hindi  --max-docs 200000 --concurrency 8
    python scripts/run_crawl.py --lang nepali --max-docs 200000 --concurrency 8

Designed to be left running for hours/days and safely re-invoked with the exact same command if
interrupted -- see lma/crawl/frontier.py for the resume mechanism. Progress and errors are logged
to <lang>/data/crawl.log so a run can be monitored without keeping a terminal attached.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.crawl.async_crawler import run_crawl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the manual-collection crawl for one language.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--max-docs", type=int, default=200_000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--min-delay", type=float, default=1.0, help="floor seconds between requests to the same host")
    args = parser.parse_args()

    lang_dir = REPO_ROOT / args.lang
    data_dir = lang_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    log_path = data_dir / "crawl.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )

    seed_module = importlib.import_module(f"{args.lang}.data.seed_domains")
    domains = seed_module.ALL_DOMAINS

    frontier_db = data_dir / "crawl_state.sqlite3"
    out_dir = data_dir / "raw" / "crawled"

    logging.getLogger("lma.crawl").info(
        "starting crawl: lang=%s domains=%d max_docs=%d concurrency=%d",
        args.lang, len(domains), args.max_docs, args.concurrency,
    )

    asyncio.run(
        run_crawl(
            domains=domains,
            frontier_db=frontier_db,
            out_dir=out_dir,
            max_docs=args.max_docs,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            min_delay=args.min_delay,
        )
    )


if __name__ == "__main__":
    main()
