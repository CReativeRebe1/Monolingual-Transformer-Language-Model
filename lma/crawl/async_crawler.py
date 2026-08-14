"""
Polite async crawler: seeds -> sitemap discovery -> fetch -> extract -> JSONL.gz shards.

Designed to run unattended for days across many process restarts. All state that must survive a
restart (which URLs are done/pending/failed) lives in the SQLite frontier, not in memory: kill
the process at any point and rerunning the same command resumes exactly where it left off,
re-fetching nothing that already succeeded.

Output: append-only JSONL.gz shards under <out_dir>/shard_XXXXX.jsonl.gz, one JSON object per
line: {url, host, title, text, content_sha256, fetched_at, lang_hint}. `lang_hint` is None here;
language filtering is a separate downstream pipeline stage (lma/text/langid.py), deliberately
kept out of the crawler so the raw-collection stage stays simple and independently re-runnable.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from lma.crawl.extractors import extract
from lma.crawl.frontier import Frontier
from lma.crawl.robots import USER_AGENT, PolitenessManager
from lma.crawl.sitemap import discover_sitemap_urls, fetch_urls_from_sitemap

logger = logging.getLogger("lma.crawl")

MAX_DOC_BYTES = 2_000_000  # skip anything absurdly large (not article text)
MIN_TEXT_CHARS = 200  # below this, almost certainly a stub/nav page, not real content


class ShardWriter:
    """Rolls JSONL.gz shards at a target size so no single file becomes unwieldy."""

    def __init__(self, out_dir: Path, max_docs_per_shard: int = 2000):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_docs_per_shard = max_docs_per_shard
        self._shard_idx = self._next_shard_index()
        self._doc_count_in_shard = 0
        self._fh = None
        self._open_new_shard()

    def _next_shard_index(self) -> int:
        existing = sorted(self.out_dir.glob("shard_*.jsonl.gz"))
        if not existing:
            return 0
        last = existing[-1].name
        idx = int(last.split("_")[1].split(".")[0])
        return idx + 1

    def _open_new_shard(self):
        if self._fh:
            self._fh.close()
        path = self.out_dir / f"shard_{self._shard_idx:05d}.jsonl.gz"
        self._fh = gzip.open(path, "at", encoding="utf-8")
        self._doc_count_in_shard = 0

    def write(self, record: dict):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._doc_count_in_shard += 1
        if self._doc_count_in_shard >= self.max_docs_per_shard:
            self._shard_idx += 1
            self._open_new_shard()
        else:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


async def _seed_frontier_from_domains(client: httpx.AsyncClient, frontier: Frontier, domains: list[str]):
    for domain in domains:
        sitemap_candidates = await discover_sitemap_urls(client, domain)
        found_any = False
        for sm_url in sitemap_candidates:
            urls = await fetch_urls_from_sitemap(client, sm_url)
            if urls:
                added = frontier.add_many(urls, depth=1)
                logger.info("seeded %d urls (%d new) from %s", len(urls), added, sm_url)
                found_any = True
        if not found_any:
            # Fall back to the bare homepage so the frontier is not empty for this domain.
            # Generic link-following from here is deliberately NOT implemented (sitemap-first
            # policy keeps us on-target and polite; see module docstring).
            frontier.add(f"https://{domain}/", depth=0)
            logger.warning("no sitemap found for %s; seeded homepage only", domain)


async def _fetch_one(
    client: httpx.AsyncClient,
    politeness: PolitenessManager,
    url: str,
) -> tuple[int | None, str | None, str | None]:
    """Returns (http_status, html, error)."""
    if not await politeness.allowed(url):
        return None, None, "disallowed_by_robots"
    await politeness.wait_turn(url)
    try:
        resp = await client.get(url, timeout=20.0, follow_redirects=True)
        if len(resp.content) > MAX_DOC_BYTES:
            return resp.status_code, None, "too_large"
        if resp.status_code != 200:
            return resp.status_code, None, f"http_{resp.status_code}"
        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            return resp.status_code, None, f"non_html_content_type:{ctype}"
        return resp.status_code, resp.text, None
    except httpx.TimeoutException:
        return None, None, "timeout"
    except httpx.HTTPError as e:
        return None, None, f"http_error:{e.__class__.__name__}"


async def run_crawl(
    domains: list[str],
    frontier_db: Path,
    out_dir: Path,
    max_docs: int = 200_000,
    concurrency: int = 8,
    batch_size: int = 50,
    min_delay: float = 1.0,
) -> None:
    """Main entry point. Safe to interrupt and rerun; the frontier makes progress durable."""
    headers = {"User-Agent": USER_AGENT}
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=headers, limits=limits, http2=True) as client:
        politeness = PolitenessManager(client, default_min_delay=min_delay)
        with Frontier(frontier_db) as frontier:
            stats = frontier.stats()
            if stats.total == 0:
                logger.info("empty frontier; discovering sitemaps for %d domains", len(domains))
                await _seed_frontier_from_domains(client, frontier, domains)
            else:
                logger.info("resuming frontier: %s", stats)

            writer = ShardWriter(out_dir)
            fetched_this_run = 0
            sem = asyncio.Semaphore(concurrency)

            async def worker(url: str):
                nonlocal fetched_this_run
                async with sem:
                    status, html, error = await _fetch_one(client, politeness, url)
                    if error or html is None:
                        frontier.mark_failed(url, status, error or "unknown")
                        return
                    doc = extract(html, url)
                    if doc is None or len(doc.text) < MIN_TEXT_CHARS:
                        frontier.mark_skipped(url, "extraction_empty_or_too_short")
                        return
                    writer.write(
                        {
                            **asdict(doc),
                            "host": urlsplit(url).netloc,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    frontier.mark_done(url, status, doc.content_sha256)
                    fetched_this_run += 1

            try:
                while fetched_this_run < max_docs:
                    batch = frontier.claim_batch(n=batch_size)
                    if not batch:
                        requeued = frontier.requeue_failed(max_attempts=3)
                        if requeued == 0:
                            logger.info("frontier exhausted (no pending, nothing to requeue)")
                            break
                        continue
                    await asyncio.gather(*(worker(url) for url, _, _ in batch))
                    if fetched_this_run % 500 < batch_size:
                        logger.info(
                            "progress: %d fetched this run | frontier=%s",
                            fetched_this_run,
                            frontier.stats(),
                        )
            finally:
                writer.close()
                logger.info(
                    "crawl run ended: %d docs fetched | frontier=%s",
                    fetched_this_run,
                    frontier.stats(),
                )
