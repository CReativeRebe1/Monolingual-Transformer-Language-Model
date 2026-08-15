"""
Bulk-ingest Wikipedia for one language from the official Wikimedia dump.

This is the fastest legitimate route to bulk Devanagari prose -- see lma/crawl/wiki_dump.py for
why the dump beats both the MediaWiki API (capped at 1 article per request for full extracts) and
crawling the HTML. Streams the compressed archive without decompressing to disk, and strips
wikitext markup across all CPU cores.

Runs independently of the main crawler (different host, no shared politeness budget), so both can
run at the same time.

Usage:
    python scripts/run_wiki.py --lang hindi
    python scripts/run_wiki.py --lang nepali --workers 8
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.crawl.async_crawler import ShardWriter  # noqa: E402
from lma.crawl.wiki_dump import DUMP_URL, WIKI_CODE, parse_dump  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.run_wiki")


def download(url: str, dest: Path) -> Path:
    """Downloads the dump if not already present, resuming a partial download if one exists.

    Resumable by HTTP Range: these archives are ~1GB for Hindi, and a connection drop 900MB in
    should not mean starting over. A complete file is left alone entirely, so re-running this
    script never re-downloads.
    """
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        total = int(client.head(url).headers.get("content-length", 0))
        have = dest.stat().st_size if dest.exists() else 0
        if total and have == total:
            logger.info("dump already complete: %s (%.1f MB)", dest.name, total / 1e6)
            return dest
        if have:
            logger.info("resuming download at %.1f/%.1f MB", have / 1e6, total / 1e6)

        headers = {"Range": f"bytes={have}-"} if have else {}
        with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            with open(dest, "ab" if have else "wb") as fh:
                done = have
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):
                        logger.info("  %.0f/%.0f MB", done / 1e6, total / 1e6)
    logger.info("downloaded %s", dest.name)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Ingest Wikipedia from the Wikimedia dump.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--workers", type=int, default=None, help="default: cpu_count() - 1")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--dump-path", type=Path, default=None,
                        help="use an already-downloaded dump instead of fetching one")
    args = parser.parse_args()

    wiki = WIKI_CODE[args.lang]
    site = f"{wiki[:2]}.wikipedia.org"
    out_dir = REPO_ROOT / args.lang / "data" / "raw" / "wiki"

    dump_path = args.dump_path or download(
        DUMP_URL.format(wiki=wiki), REPO_ROOT / args.lang / "data" / "raw" / f"{wiki}.xml.bz2"
    )

    writer = ShardWriter(out_dir)
    try:
        n = parse_dump(dump_path, site, writer, workers=args.workers, max_docs=args.max_docs)
    finally:
        writer.close()
    print(f"{site}: {n} documents written to {out_dir}")


if __name__ == "__main__":
    main()
