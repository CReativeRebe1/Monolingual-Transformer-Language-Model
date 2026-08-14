"""
Sitemap discovery: turns a handful of seed domains into thousands of article URLs.

News/archive sites (our main manual-collection sources) almost universally expose XML sitemaps,
often nested (a sitemap index pointing at dated sub-sitemaps). Walking these is far higher-yield
and far more polite than generic link-following crawling, since it hits exactly the URLs the
site wants indexed rather than probing navigation/category pages.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
_COMMON_PATHS = ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml", "/news-sitemap.xml")


async def discover_sitemap_urls(client: httpx.AsyncClient, domain: str) -> list[str]:
    """Return candidate sitemap URLs for a domain: robots.txt-declared ones plus common paths."""
    candidates: list[str] = []
    try:
        resp = await client.get(f"https://{domain}/robots.txt", timeout=10.0, follow_redirects=True)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.strip().lower().startswith("sitemap:"):
                    candidates.append(line.split(":", 1)[1].strip())
    except (httpx.HTTPError, httpx.TimeoutException):
        pass
    for path in _COMMON_PATHS:
        candidates.append(f"https://{domain}{path}")
    # de-dupe, preserve order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


async def fetch_urls_from_sitemap(
    client: httpx.AsyncClient,
    sitemap_url: str,
    max_depth: int = 3,
    max_sitemaps_total: int = 40,
    _budget: list[int] | None = None,
) -> list[str]:
    """Recursively resolve a sitemap (or sitemap index) into leaf page URLs.

    Handles both <urlset> (leaf URLs) and <sitemapindex> (pointers to further sitemaps),
    which is the structure news sites typically use to shard sitemaps by month/year.

    max_sitemaps_total bounds the total number of sitemap documents fetched across the whole
    recursion (not just per level): some large news sites (bbc.com in particular) expose a
    single global archive index sharded into 100+ dated sub-sitemaps going back years, across
    every language edition. Walking all of them before seeding a single other domain would
    burn most of a crawl session's startup time on one site. Capping the total lets bootstrap
    seeding finish in a bounded amount of wall-clock time; later crawler runs against the same
    frontier can raise the cap to pick up more of a large site's archive once other domains are
    already flowing.
    """
    if _budget is None:
        _budget = [max_sitemaps_total]
    if max_depth <= 0 or _budget[0] <= 0:
        return []
    _budget[0] -= 1
    try:
        resp = await client.get(sitemap_url, timeout=15.0, follow_redirects=True)
        if resp.status_code != 200:
            return []
        content = resp.content
        if sitemap_url.endswith(".gz") or resp.headers.get("content-type", "").endswith("gzip"):
            import gzip

            content = gzip.decompress(content)
        root = ET.fromstring(content)
    except (httpx.HTTPError, httpx.TimeoutException, ET.ParseError, OSError):
        return []

    urls: list[str] = []
    sub_sitemaps: list[str] = []
    for child in root:
        tag = child.tag.replace(_SITEMAP_NS, "")
        if tag == "url":
            loc = child.find(f"{_SITEMAP_NS}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
        elif tag == "sitemap":
            loc = child.find(f"{_SITEMAP_NS}loc")
            if loc is not None and loc.text:
                sub_sitemaps.append(loc.text.strip())

    for sub in sub_sitemaps:
        if _budget[0] <= 0:
            break
        urls.extend(
            await fetch_urls_from_sitemap(
                client, sub, max_depth=max_depth - 1, _budget=_budget
            )
        )
    return urls
