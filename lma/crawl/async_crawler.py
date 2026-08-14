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
from lma.crawl.linkextract import extract_same_domain_links
from lma.crawl.robots import USER_AGENT, PolitenessManager
from lma.crawl.sitemap import discover_sitemap_urls, fetch_urls_from_sitemap

logger = logging.getLogger("lma.crawl")

MAX_DOC_BYTES = 2_000_000  # skip anything absurdly large (not article text)
MIN_TEXT_CHARS = 200  # below this, almost certainly a stub/nav page, not real content
MAX_LINK_FOLLOW_DEPTH = 4  # bounds the BFS fallback for domains without a usable sitemap


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


async def _seed_frontier_from_domains(
    client: httpx.AsyncClient,
    frontier: Frontier,
    domains: list[str],
    url_filters: dict[str, str] | None = None,
):
    """Discover sitemap URLs per domain and add them to the frontier.

    url_filters: optional {domain: required_substring}. Some domains (bbc.com in particular)
    share one sitemap tree across every language edition of the site; without a filter, sitemap
    discovery would seed the frontier mostly with off-language URLs from that shared index. See
    hindi/data/seed_domains.py and nepali/data/seed_domains.py for the concrete values.
    """
    url_filters = url_filters or {}
    for domain in domains:
        if frontier.is_domain_seeded(domain):
            # Per-domain tracking (not "is the frontier empty") is what makes this safe to call
            # on every crawler startup: a restart after domain 1 of 8 finished seeding correctly
            # resumes seeding domains 2-8, rather than skipping seeding entirely because the
            # frontier already has domain 1's URLs in it. See _SCHEMA's seeded_domains comment.
            logger.info("domain %s already seeded; skipping", domain)
            continue

        required_substring = url_filters.get(domain)
        sitemap_candidates = await discover_sitemap_urls(client, domain)
        found_any = False
        total_seeded_for_domain = 0
        for sm_url in sitemap_candidates:
            urls = await fetch_urls_from_sitemap(client, sm_url)
            if required_substring is not None:
                before = len(urls)
                urls = [u for u in urls if required_substring in u]
                logger.info(
                    "filtered %s by substring %r: %d -> %d urls",
                    sm_url, required_substring, before, len(urls),
                )
            if urls:
                added = frontier.add_many(urls, depth=1)
                logger.info("seeded %d urls (%d new) from %s", len(urls), added, sm_url)
                found_any = True
                total_seeded_for_domain += added
        if not found_any:
            # Fall back to the bare homepage so the frontier is not empty for this domain.
            # Generic link-following from here is deliberately NOT implemented (sitemap-first
            # policy keeps us on-target and polite; see module docstring).
            seed_url = f"https://{domain}/"
            if required_substring is None or required_substring in seed_url:
                if frontier.add(seed_url, depth=0):
                    total_seeded_for_domain += 1
                logger.warning("no sitemap found for %s; seeded homepage only", domain)
            else:
                logger.warning(
                    "no sitemap found for %s and homepage does not match filter %r; skipping",
                    domain, required_substring,
                )
        frontier.mark_domain_seeded(domain, total_seeded_for_domain)


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
    url_filters: dict[str, str] | None = None,
) -> None:
    """Main entry point. Safe to interrupt and rerun; the frontier makes progress durable."""
    headers = {"User-Agent": USER_AGENT}
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=headers, limits=limits, http2=True) as client:
        politeness = PolitenessManager(client, default_min_delay=min_delay)
        with Frontier(frontier_db) as frontier:
            stats = frontier.stats()
            logger.info("frontier at startup: %s", stats)
            # Always attempt seeding, for every run: _seed_frontier_from_domains skips any
            # domain already marked seeded (see its docstring for why this must be per-domain,
            # not gated on "is the whole frontier empty" -- that check silently starved every
            # domain past the first one across crawler restarts, a real bug caught by noticing
            # the Hindi frontier was 100% one domain after several restarts).
            await _seed_frontier_from_domains(client, frontier, domains, url_filters=url_filters)

            writer = ShardWriter(out_dir)
            fetched_this_run = 0
            sem = asyncio.Semaphore(concurrency)

            async def worker(url: str, depth: int):
                nonlocal fetched_this_run
                async with sem:
                    status, html, error = await _fetch_one(client, politeness, url)
                    if error or html is None:
                        frontier.mark_failed(url, status, error or "unknown")
                        return
                    doc = extract(html, url)
                    if doc is None or len(doc.text) < MIN_TEXT_CHARS:
                        frontier.mark_skipped(url, "extraction_empty_or_too_short")
                    else:
                        writer.write(
                            {
                                **asdict(doc),
                                "host": urlsplit(url).netloc,
                                "fetched_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        frontier.mark_done(url, status, doc.content_sha256)
                        fetched_this_run += 1
                    # Link-following fallback: even a skipped page (e.g. a category/listing page
                    # that itself has too little text to count as an article) is still useful as
                    # a source of further same-domain links, which is exactly how a domain with
                    # no working sitemap gets past its single seeded homepage URL. Every fetched
                    # page (whether or not it counted as a document) contributes links up to the
                    # depth cap, so BFS naturally reaches deeper into a site's archive over time.
                    if depth < MAX_LINK_FOLLOW_DEPTH:
                        links = extract_same_domain_links(html, url)
                        frontier.add_many(links, depth=depth + 1)

            try:
                while fetched_this_run < max_docs:
                    batch = frontier.claim_batch(n=batch_size)
                    if not batch:
                        requeued = frontier.requeue_failed(max_attempts=3)
                        if requeued == 0:
                            logger.info("frontier exhausted (no pending, nothing to requeue)")
                            break
                        continue
                    await asyncio.gather(*(worker(url, depth) for url, _, depth in batch))
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
