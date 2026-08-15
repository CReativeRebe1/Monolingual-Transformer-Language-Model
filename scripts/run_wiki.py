"""
Bulk-ingest Wikipedia for one language from the processed Wikimedia dump.

Why the dump and not the MediaWiki API: the API looks like the fast path, but for whole-article
plain-text extracts MediaWiki silently caps `exlimit` at 1 -- confirmed live against
hi.wikipedia.org, which returned the warning *"exlimit was too large for a whole article extracts
request, lowered to 1"* -- so an API walk is one article per HTTP round-trip, no faster than just
crawling the HTML. Only intro-snippet extracts (`exintro`) allow batching, and snippets are not
what a pretraining corpus wants. The Wikimedia dump exposed through HuggingFace is already
plain-text, already deduplicated by revision, and streams in bulk, so it is the genuinely fast
route.

This reuses lma/crawl/downloaded_corpus.py's streaming ingester rather than duplicating it: that
path was already verified working against the live Hub earlier in Phase 1. Records land tagged
source_type="downloaded" (Wikipedia is a bulk export offered by the site, not pages we discovered
and scraped ourselves), so the >=20%-manual claim continues to rest only on the genuine crawl.

Runs independently of the main crawler -- different host, no shared politeness budget -- so both
can run at the same time.

Usage:
    python scripts/run_wiki.py --lang hindi
    python scripts/run_wiki.py --lang nepali --max-tokens 50000000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.crawl.downloaded_corpus import ingest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Ingest Wikipedia from the Wikimedia dump.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument(
        "--max-tokens", type=int, default=200_000_000,
        help="whitespace-token budget; the default is high enough to pull the whole of either "
             "language's Wikipedia, which is far smaller than this cap",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <lang>/data/raw/wiki/")
    args = parser.parse_args()

    out_dir = args.out_dir or REPO_ROOT / args.lang / "data" / "raw" / "wiki"
    reports = ingest(args.lang, out_dir, token_budget_per_source=args.max_tokens, sources=["wikipedia"])
    for r in reports:
        print(f"{r.source_name}: status={r.status} docs={r.documents_written} tokens={r.approx_tokens_written}")


if __name__ == "__main__":
    main()
