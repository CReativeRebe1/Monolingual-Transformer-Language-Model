"""
Same-domain link extraction: the fallback path when a site exposes no usable sitemap.

Sitemap discovery (lma/crawl/sitemap.py) is the primary, higher-yield, more polite seeding
strategy. In practice several Nepali news sites in this project return 404/502 for every
standard sitemap path, which sitemap discovery alone cannot work around. Rather than leave
those domains starved at a single homepage URL, the crawler falls back to breadth-first
link-following from whatever pages it has already fetched -- exactly what a generic web
crawler does, scoped defensively so it stays on the article-like parts of a site rather than
wandering into navigation, search, or account pages.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

from selectolax.parser import HTMLParser

# Path segments that reliably indicate a non-article page on a news/CMS site: navigation,
# account/auth flows, search, tag/category index pages, and non-HTML asset links. Filtering
# these out keeps link-following crawl budget spent on article-shaped URLs.
_DENYLIST_SEGMENTS = re.compile(
    r"/(login|signin|signup|register|search|tag|tags|category|categories|author|authors|"
    r"video|videos|live|livetv|photo|photos|gallery|about|contact|privacy|terms|rss|feed|"
    r"subscribe|newsletter|advertise|careers|jobs|sitemap|events?)(/|$|\?)",
    re.IGNORECASE,
)
# Substring (not exact-segment) match: news-portal "section index / archive listing" pages --
# e.g. amarujala.com's /news-archives, /news-archives/auto-archives -- have no article body of
# their own (confirmed empirically: they extracted as pure cookie-notice boilerplate, see
# lma/text/filters.py's BOILERPLATE_PHRASES comment) and are worth excluding from link-following
# even though "archives" never appears as a standalone path segment on its own.
_ARCHIVE_INDEX_RE = re.compile(r"archives?", re.IGNORECASE)
_ASSET_EXT_RE = re.compile(
    r"\.(jpg|jpeg|png|gif|svg|webp|css|js|pdf|zip|mp4|mp3|ico|woff2?|ttf)(\?|$)", re.IGNORECASE
)


def extract_links(html: str, base_url: str, max_links: int = 30) -> list[str]:
    """Return up to max_links same-domain, article-shaped absolute URLs found in the page.

    Deliberately conservative: this is a fallback for sites without a working sitemap, not a
    general-purpose crawler, so it stays same-domain (never follows offsite links, which would
    turn a scoped manual-collection crawl into an open-ended one) and filters out the page
    types the denylist above identifies as structurally not article content.
    """
    base_host = urlsplit(base_url).netloc
    try:
        tree = HTMLParser(html)
    except Exception:
        return []

    found: list[str] = []
    seen: set[str] = set()
    for node in tree.css("a[href]"):
        href = node.attributes.get("href")
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parts = urlsplit(absolute)
        if parts.scheme not in ("http", "https"):
            continue
        if parts.netloc != base_host:
            continue  # same-domain only
        if _ASSET_EXT_RE.search(parts.path):
            continue
        if _DENYLIST_SEGMENTS.search(parts.path) or _ARCHIVE_INDEX_RE.search(parts.path):
            continue
        clean = f"{parts.scheme}://{parts.netloc}{parts.path}"  # drop query/fragment noise
        if clean in seen:
            continue
        seen.add(clean)
        found.append(clean)
        if len(found) >= max_links:
            break
    return found
