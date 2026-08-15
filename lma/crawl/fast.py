"""
High-throughput resumable crawler.

Why this exists alongside lma/crawl/async_crawler.py: that crawler serialises per host (one
in-flight request, 1.5s apart), which caps a domain at ~0.67 req/s no matter how high global
concurrency goes. Total throughput is then `live_domains / delay`, and with ~30 domains that is
the binding constraint on collecting tens of millions of manual tokens before a deadline.

This one lifts the per-host cap instead of the domain count:
  * up to `per_domain` concurrent requests per host (default 8) rather than one;
  * `delay` seconds between request *starts* to a host (default 0.35), not between completions,
    so a slow response never idles the host;
  * a priority frontier, so sitemaps and article-shaped URLs are fetched before tag/author
    listing pages that yield no prose;
  * a stdlib HTMLParser instead of trafilatura -- extraction stops being the CPU bottleneck when
    fetches arrive an order of magnitude faster;
  * language scoring and exact dedup at crawl time, so obviously-useless pages never reach disk.

Politeness is preserved, not traded away: robots.txt is honoured per host, a robots-declared
Crawl-delay always overrides the CLI delay when it is larger, and per-host concurrency stays
bounded. The regime is closer to what a browser does loading a page than to a single-threaded
1.5s crawl, and these are large commercial news sites.

Output records are written in the same JSONL.gz shard schema the existing crawler emits, so
scripts/build_corpus.py consumes them with no changes.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import html
import json
import logging
import re
import sqlite3
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import aiohttp

logger = logging.getLogger("lma.fast")

# --------------------------------------------------------------------------------------
# Policy constants
# --------------------------------------------------------------------------------------

SITEMAP_PATHS = (
    "/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
    "/wp-sitemap.xml", "/sitemap/sitemap.xml", "/news-sitemap.xml", "/sitemap-news.xml",
)

TRACKING_PARAMS = frozenset({
    "gclid", "fbclid", "igshid", "mc_cid", "mc_eid", "ref", "referrer",
    "share", "source", "amp", "output", "_ga",
})

SKIP_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff",
    ".mp3", ".mp4", ".m4a", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".epub",
    ".zip", ".gz", ".tar", ".rar", ".7z", ".bz2",
    ".css", ".js", ".json", ".woff", ".woff2", ".ttf", ".eot", ".apk", ".exe", ".dmg",
)

SKIP_PATH_FRAGMENTS = (
    "/login", "/signin", "/signup", "/register", "/logout", "/account", "/subscribe",
    "/search", "/privacy", "/terms", "/disclaimer", "/contact", "/advertise",
    "/feed", "/rss", "/comment", "/newsletter", "/web-stories", "/webstory",
    "/cdn-cgi/", "/wp-admin", "/wp-login", "/wp-json", "/xmlrpc",
)

LOW_PRIORITY_FRAGMENTS = (
    "/tag/", "/tags/", "/author/", "/authors/", "/category/", "/categories/",
    "/topic/", "/topics/", "/video/", "/videos/", "/photo/", "/photos/",
    "/gallery/", "/galleries/", "/podcast/", "/page/", "/archive",
)

# Frequent function words. Devanagari alone cannot separate Hindi from Nepali/Marathi, so a
# marker count in the *target* language is what actually discriminates (see PHASE1_LOG.md's
# language-ID section for the same problem in the offline pipeline).
LANG_MARKERS = {
    "hindi": frozenset(
        "और के की का को है हैं में से पर एक यह वह हो था थी थे कि लिए भी नहीं ने तो जो "
        "किया गया रहे कहा साथ बाद लेकिन अपने दिया करने वाले".split()
    ),
    "nepali": frozenset(
        "र को की का छ छन् थियो थिए भएको लागि तर यदि पनि होइन ले मा हो यो त्यो अनि जुन "
        "गरेको भएका हुन् गर्ने रहेको भन्ने साथै बताए अनुसार".split()
    ),
}

_WS_RE = re.compile(r"[ \t  -   　]+")
_WORD_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_ZERO_WIDTH_RE = re.compile(r"[​-‏﻿⁠]")
_DATE_PATH_RE = re.compile(r"/(19|20)\d{2}[/-]\d{1,2}([/-]\d{1,2})?")
_LONG_ID_RE = re.compile(r"[-/](\d{6,})(?:[./]|$)")
_SITEMAPish_RE = re.compile(r"sitemap|\.xml(\.gz)?$", re.IGNORECASE)
_BOILERPLATE_RE = re.compile(
    r"copyright|all rights reserved|subscribe|newsletter|follow us|share this|advertisement|"
    r"सर्वाधिकार सुरक्षित|विज्ञापन|सब्सक्राइब|फॉलो कर", re.IGNORECASE
)

PRIO_SITEMAP, PRIO_SM_LINK, PRIO_ARTICLE, PRIO_SLUG, PRIO_ROOT, PRIO_PLAIN, PRIO_LOW = (
    100, 70, 60, 45, 30, 20, 5
)

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


# --------------------------------------------------------------------------------------
# Data contracts
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class FetchResult:
    requested_url: str
    final_url: str
    depth: int
    priority: int
    ok: bool
    status: int | None = None
    content_type: str = ""
    body: bytes = b""
    error: str = ""
    retry_after: float = 0.0


@dataclass(slots=True)
class ExtractedPage:
    title: str
    text: str
    published_at: str | None
    canonical_url: str | None
    links: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# URL handling
# --------------------------------------------------------------------------------------

def registered_host(host: str) -> str:
    """Pragmatic host key: lowercase, trailing dot and leading 'www.' removed.

    Deliberately not a Public Suffix List lookup -- it only needs to group a site's pages for
    scope and rate-limiting, and PSL parsing would add a dependency for no crawl benefit.
    """
    host = (host or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Canonicalises a URL, or returns None if it should never enter the frontier."""
    try:
        if base:
            url = urljoin(base, url)
        parts = urlsplit(url)
    except (ValueError, UnicodeError):
        return None

    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:
        return None

    host = parts.hostname.lower().rstrip(".")
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    lower = path.lower()
    if lower.endswith(SKIP_SUFFIXES) or any(f in lower for f in SKIP_PATH_FRAGMENTS):
        return None

    query = urlencode(
        sorted(
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
        )
    )
    return urlunsplit((parts.scheme, netloc, path, query, ""))


def in_scope(url: str, allowed: frozenset[str], host_prefixes: dict[str, str]) -> bool:
    """True if url belongs to a seed domain (or subdomain) and satisfies any path prefix."""
    host = registered_host(urlsplit(url).hostname or "")
    if not any(host == a or host.endswith("." + a) for a in allowed):
        return False
    prefix = host_prefixes.get(host)
    if prefix and not _SITEMAPish_RE.search(url):
        return urlsplit(url).path.startswith(prefix)
    return True


def url_priority(url: str, from_sitemap: bool = False) -> int:
    """Higher is fetched first. Article-shaped URLs beat listing surfaces."""
    parts = urlsplit(url)
    path = parts.path.lower()
    if _SITEMAPish_RE.search(url):
        return PRIO_SITEMAP
    if any(f in path for f in LOW_PRIORITY_FRAGMENTS):
        return PRIO_LOW
    if from_sitemap:
        return PRIO_SM_LINK
    if _DATE_PATH_RE.search(path) or _LONG_ID_RE.search(path):
        return PRIO_ARTICLE
    if path in ("", "/"):
        return PRIO_ROOT
    slug = path.rstrip("/").rsplit("/", 1)[-1]
    return PRIO_SLUG if len(slug) > 25 or slug.count("-") >= 3 else PRIO_PLAIN


# --------------------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------------------

_IGNORE_TAGS = frozenset({
    "script", "style", "noscript", "svg", "canvas", "nav", "footer", "aside",
    "form", "button", "select", "textarea", "iframe", "header",
})
_BLOCK_TAGS = frozenset({"p", "h1", "h2", "h3", "h4", "blockquote", "li"})


class ArticleParser(HTMLParser):
    """Streaming extractor: title, headings, prose blocks, links, canonical, meta, JSON-LD."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.headings: list[str] = []
        self.blocks: list[str] = []
        self.links: list[str] = []
        self.canonical: str | None = None
        self.meta: dict[str, str] = {}
        self.ld_json: list[str] = []
        self._ignore_depth = 0
        self._in_title = False
        self._in_ld = False
        self._buf: list[str] = []
        self._buf_tag: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in _IGNORE_TAGS:
            # Track nesting so a nested ignored element does not re-enable capture on its close.
            if not (tag == "script" and (a.get("type") or "").lower() == "application/ld+json"):
                self._ignore_depth += 1
                return
            self._in_ld = True
            return
        if self._ignore_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"])
        elif tag == "link" and (a.get("rel") or "").lower() == "canonical" and a.get("href"):
            self.canonical = a["href"]
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"])
        elif tag in _BLOCK_TAGS:
            self._buf, self._buf_tag = [], tag

    def handle_endtag(self, tag):
        if tag in _IGNORE_TAGS:
            if self._in_ld and tag == "script":
                self._in_ld = False
            elif self._ignore_depth:
                self._ignore_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        elif tag == self._buf_tag:
            text = clean_inline("".join(self._buf))
            if text:
                (self.headings if tag in ("h1", "h2", "h3") else self.blocks).append(text)
                if tag in ("h1", "h2", "h3"):
                    self.blocks.append(text)
            self._buf, self._buf_tag = [], None

    def handle_data(self, data):
        if self._in_ld:
            self.ld_json.append(data)
        elif self._in_title:
            self.title += data
        elif not self._ignore_depth and self._buf_tag:
            self._buf.append(data)


def clean_inline(value: str) -> str:
    value = html.unescape(value or "")
    value = unicodedata.normalize("NFC", value)
    value = _ZERO_WIDTH_RE.sub("", value)
    return _WS_RE.sub(" ", value).strip()


def clean_text(title: str, blocks: list[str]) -> str:
    """Drops short fragments, exact repeats, title echoes, and short boilerplate lines."""
    seen, kept = set(), []
    for raw in blocks:
        block = clean_inline(raw)
        if len(block) < 20 or block in seen or block == title:
            continue
        if len(block) < 200 and _BOILERPLATE_RE.search(block):
            continue
        seen.add(block)
        kept.append(block)
    return "\n\n".join(kept)


def iter_json_objects(value):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from iter_json_objects(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_json_objects(v)


def json_ld_article(parser: ArticleParser) -> tuple[str, str, str | None]:
    """Returns (headline, body, published) from the longest JSON-LD articleBody found."""
    best_body, headline, published = "", "", None
    for raw in parser.ld_json:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        for obj in iter_json_objects(data):
            types = obj.get("@type", "")
            types = " ".join(types) if isinstance(types, list) else str(types)
            if "Article" not in types:
                continue
            body = obj.get("articleBody")
            if isinstance(body, str) and len(body) > len(best_body):
                best_body = body
                headline = obj.get("headline") if isinstance(obj.get("headline"), str) else ""
                pub = obj.get("datePublished")
                published = pub if isinstance(pub, str) else None
    return headline, best_body, published


def extract_page(body: bytes, final_url: str, content_type: str) -> ExtractedPage:
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip().strip('"\'') or "utf-8"
    try:
        text = body.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = body.decode("utf-8", errors="replace")

    parser = ArticleParser()
    try:
        parser.feed(text)
    except Exception:  # noqa: BLE001 - malformed markup must not kill the worker
        pass

    ld_head, ld_body, ld_date = json_ld_article(parser)
    html_text = clean_text(clean_inline(parser.title), parser.blocks)
    title = (
        parser.meta.get("og:title") or ld_head
        or (parser.headings[0] if parser.headings else "") or clean_inline(parser.title)
    )
    published = (
        parser.meta.get("article:published_time") or parser.meta.get("datepublished")
        or parser.meta.get("date") or ld_date
    )
    ld_clean = clean_inline(ld_body)
    return ExtractedPage(
        title=clean_inline(title),
        text=ld_clean if len(ld_clean) > len(html_text) else html_text,
        published_at=published,
        canonical_url=urljoin(final_url, parser.canonical) if parser.canonical else None,
        links=[urljoin(final_url, h) for h in parser.links],
    )


# --------------------------------------------------------------------------------------
# Sitemaps
# --------------------------------------------------------------------------------------

def maybe_gunzip(raw: bytes, url: str) -> bytes:
    if raw[:2] == b"\x1f\x8b" or url.lower().endswith(".gz"):
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError):
            return raw
    return raw


def extract_xml_links(raw: bytes, url: str) -> list[str]:
    raw = maybe_gunzip(raw, url)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1].lower()
        if tag in ("loc", "link"):
            val = (el.text or el.get("href") or "").strip()
            if val.startswith(("http://", "https://")):
                out.append(val)
    return out


def looks_like_xml(body: bytes, content_type: str, url: str) -> bool:
    head = body[:200].lstrip().lower()
    return (
        "xml" in content_type
        or bool(_SITEMAPish_RE.search(url))
        or head.startswith((b"<?xml", b"<urlset", b"<sitemapindex"))
        or body[:2] == b"\x1f\x8b"
    )


# --------------------------------------------------------------------------------------
# Language scoring and hashing
# --------------------------------------------------------------------------------------

def devanagari_score(text: str, language: str) -> tuple[float, int]:
    """Returns (devanagari_ratio_over_alphabetic, count_of_distinct_target_language_markers)."""
    alpha = deva = 0
    for ch in text:
        if ch.isalpha():
            alpha += 1
            if "ऀ" <= ch <= "ॿ":
                deva += 1
    ratio = (deva / alpha) if alpha else 0.0
    tokens = set(_WORD_RE.findall(text))
    return ratio, len(tokens & LANG_MARKERS[language])


def content_hash(text: str) -> str:
    """Stable exact-dedup key: NFC + casefold + word tokens joined by single spaces.

    This definition is a persistence contract -- changing it invalidates every hash already
    stored in an existing state DB, so previously-seen articles would be re-saved.
    """
    norm = unicodedata.normalize("NFC", text).casefold()
    return hashlib.sha256(" ".join(_WORD_RE.findall(norm)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Persistent store
# --------------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    depth INTEGER NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    discovered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
    content_hash TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    char_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    collected_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS urls_claim_idx ON urls(status, available_at, priority DESC, attempts);
CREATE INDEX IF NOT EXISTS urls_domain_claim_idx
    ON urls(domain, status, available_at, priority DESC, attempts);
"""


class CorpusStore:
    """SQLite frontier + exact-dedup index. Touched only from the coordinator task.

    Keeping every mutation on one task is what makes batched commits safe: fetch workers return
    plain results and never write, so there is no cross-task transaction interleaving to reason
    about.
    """

    def __init__(self, path: Path, max_urls: int = 0, commit_every: int = 1000,
                 commit_seconds: float = 5.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "temp_store=MEMORY",
                       "cache_size=-65536", "busy_timeout=5000"):
            self.conn.execute(f"PRAGMA {pragma}")
        self.conn.executescript(_SCHEMA)
        self.max_urls = max_urls
        self.commit_every = commit_every
        self.commit_seconds = commit_seconds
        self._dirty = 0
        self._last_commit = time.monotonic()
        # Any row left 'fetching' belongs to a previous run that died mid-flight; recovering it
        # here is what makes a hard kill (or power loss) lose no frontier work.
        self.conn.execute("UPDATE urls SET status='pending' WHERE status='fetching'")
        self.conn.commit()
        self._count = self.conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
        self._domains = [r[0] for r in self.conn.execute("SELECT DISTINCT domain FROM urls")]
        self._cursor = 0

    def add_urls(self, items: list[tuple[str, str, int, int]]) -> int:
        """items: (url, domain, depth, priority). Returns rows actually inserted."""
        if not items:
            return 0
        if self.max_urls:
            room = self.max_urls - self._count
            if room <= 0:
                return 0
            items = items[:room]
        seen, batch = set(), []
        for url, domain, depth, prio in items:
            if url not in seen:
                seen.add(url)
                batch.append((url, domain, depth, prio, _now()))
        before = self.conn.total_changes
        self.conn.executemany(
            "INSERT OR IGNORE INTO urls (url,domain,depth,priority,discovered_at) "
            "VALUES (?,?,?,?,?)", batch,
        )
        inserted = self.conn.total_changes - before
        self._count += inserted
        self._dirty += inserted
        for _, domain, _, _ in items:
            if domain not in self._domains:
                self._domains.append(domain)
        return inserted

    def claim(self, limit: int, per_domain_room: dict[str, int], max_attempts: int) -> list[tuple]:
        """Claims up to `limit` ready URLs, rotating across domains.

        Querying per domain rather than one global ORDER BY is deliberate: a single site with a
        million-URL sitemap would otherwise fill every global priority window and starve every
        other domain indefinitely.
        """
        if not self._domains or limit <= 0:
            return []
        now = time.time()
        claimed: list[tuple] = []
        domains = sorted(self._domains)
        # Fair share first, leftovers second. Without the share cap, the first domain visited
        # simply takes `limit` rows and every other domain waits for the next call -- which is
        # precisely the starvation this per-domain query exists to prevent, and is invisible
        # until a domain with a huge sitemap meets a domain with three URLs.
        with_room = [d for d in domains if per_domain_room.get(d, 0) > 0]
        if not with_room:
            return []
        share = max(1, limit // len(with_room))
        for cap in (share, limit):          # pass 1: fair share; pass 2: fill spare capacity
            for i in range(len(domains)):
                if len(claimed) >= limit:
                    break
                domain = domains[(self._cursor + i) % len(domains)]
                room = per_domain_room.get(domain, 0) - sum(1 for c in claimed if c[1] == domain)
                if room <= 0:
                    continue
                take = min(room, cap, limit - len(claimed))
                if take <= 0:
                    continue
                rows = self._claim_rows(domain, now, max_attempts, take)
                for url, depth, prio in rows:
                    self.conn.execute(
                        "UPDATE urls SET status='fetching', attempts=attempts+1 WHERE url=?",
                        (url,),
                    )
                    claimed.append((url, domain, depth, prio))
                    self._dirty += 1
            if len(claimed) >= limit:
                break
        self._cursor = (self._cursor + 1) % max(1, len(domains))
        return claimed

    def _claim_rows(self, domain, now, max_attempts, take):
        return self.conn.execute(
            "SELECT url,depth,priority FROM urls WHERE domain=? AND status='pending' "
            "AND available_at<=? AND attempts<? ORDER BY priority DESC, rowid LIMIT ?",
            (domain, now, max_attempts, take),
        ).fetchall()

    def finish(self, url: str, ok: bool, error: str = "") -> None:
        self.conn.execute(
            "UPDATE urls SET status=?, last_error=?, available_at=0 WHERE url=?",
            ("done" if ok else "failed", (error or "")[:300], url),
        )
        self._dirty += 1

    def requeue(self, url: str, delay: float) -> None:
        self.conn.execute(
            "UPDATE urls SET status='pending', available_at=? WHERE url=?",
            (time.time() + max(0.0, delay), url),
        )
        self._dirty += 1

    def add_article(self, h: str, url: str, chars: int, words: int) -> bool:
        """Returns True if this is new content (by hash and by canonical URL)."""
        before = self.conn.total_changes
        self.conn.execute(
            "INSERT OR IGNORE INTO articles (content_hash,url,char_count,word_count,collected_at)"
            " VALUES (?,?,?,?,?)", (h, url, chars, words, _now()),
        )
        added = self.conn.total_changes > before
        self._dirty += int(added)
        return added

    def domains(self) -> list[str]:
        return list(self._domains)

    def totals(self) -> tuple[int, int, int]:
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(char_count),0), COALESCE(SUM(word_count),0) FROM articles"
        ).fetchone()
        return row[0], row[1], row[2]

    def pending_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM urls WHERE status='pending'"
        ).fetchone()[0]

    def next_retry_in(self) -> float | None:
        row = self.conn.execute(
            "SELECT MIN(available_at) FROM urls WHERE status='pending' AND available_at>?",
            (time.time(),),
        ).fetchone()
        return None if not row or row[0] is None else max(0.0, row[0] - time.time())

    def maybe_commit(self, force: bool = False) -> None:
        if force or self._dirty >= self.commit_every or (
            time.monotonic() - self._last_commit >= self.commit_seconds and self._dirty
        ):
            self.conn.commit()
            self._dirty = 0
            self._last_commit = time.monotonic()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------
# Robots + per-host pacing
# --------------------------------------------------------------------------------------

class RobotsAndRateLimiter:
    """Per-host robots rules, concurrency semaphore, and request-start spacing.

    Spacing is applied to request *starts*, not completions: with `per_domain` requests allowed
    in flight, waiting for a response before starting the next one would collapse throughput back
    to the serial case whenever a host is slow.
    """

    def __init__(self, session: aiohttp.ClientSession, user_agent: str, min_delay: float,
                 per_domain: int):
        self._session = session
        self._ua = user_agent
        self._min_delay = min_delay
        self._per_domain = per_domain
        self._robots: dict[str, tuple[RobotFileParser, float]] = {}
        self._robot_locks: dict[str, asyncio.Lock] = {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._next_at: dict[str, float] = {}
        self._pace_locks: dict[str, asyncio.Lock] = {}

    def _sem(self, host: str) -> asyncio.Semaphore:
        if host not in self._sems:
            self._sems[host] = asyncio.Semaphore(self._per_domain)
        return self._sems[host]

    async def _rules(self, origin: str) -> tuple[RobotFileParser, float]:
        if origin in self._robots:
            return self._robots[origin]
        lock = self._robot_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin in self._robots:
                return self._robots[origin]
            rp = RobotFileParser()
            try:
                async with self._session.get(
                    f"{origin}/robots.txt", timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    body = (await resp.content.read(512 * 1024)).decode("utf-8", errors="replace")
                    rp.parse(body.splitlines() if resp.status == 200 else [])
            except Exception:  # noqa: BLE001 - unreachable robots is treated as no restrictions
                rp.parse([])
            declared = rp.crawl_delay(self._ua) or rp.crawl_delay("*") or 0.0
            # A site-declared Crawl-delay always wins when it is stricter than our own floor.
            effective = max(float(declared or 0.0), self._min_delay)
            self._robots[origin] = (rp, effective)
            return self._robots[origin]

    async def allowed(self, url: str) -> tuple[bool, float]:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        rp, delay = await self._rules(origin)
        try:
            return rp.can_fetch(self._ua, url), delay
        except Exception:  # noqa: BLE001
            return True, delay

    async def acquire(self, host: str, delay: float):
        """Reserves this host's next start slot, then waits for it. Returns the held semaphore."""
        sem = self._sem(host)
        await sem.acquire()
        lock = self._pace_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            start_at = max(now, self._next_at.get(host, 0.0))
            self._next_at[host] = start_at + delay
        wait = start_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)   # deliberately outside the lock: never block other tasks
        return sem


# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------

def parse_retry_after(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return min(900.0, max(0.0, float(value)))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return min(900.0, max(0.0, (dt - datetime.now(timezone.utc)).total_seconds()))
    except Exception:  # noqa: BLE001
        return 0.0


_OK_TYPES = ("text/", "application/xml", "application/xhtml", "application/rss",
             "application/atom", "application/gzip", "application/x-gzip",
             "application/octet-stream", "")


async def fetch(session, limiter, url, depth, prio, max_bytes, timeout) -> FetchResult:
    """Never raises: every outcome, including errors, comes back as a FetchResult."""
    host = registered_host(urlsplit(url).hostname or "")
    try:
        ok, delay = await limiter.allowed(url)
        if not ok:
            return FetchResult(url, url, depth, prio, False, error="robots_disallowed")
        sem = await limiter.acquire(host, delay)
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True,
                max_redirects=8,
            ) as resp:
                ctype = (resp.headers.get("content-type") or "").lower()
                if resp.status >= 400:
                    return FetchResult(
                        url, str(resp.url), depth, prio, False, status=resp.status,
                        error=f"http_{resp.status}",
                        retry_after=parse_retry_after(resp.headers.get("retry-after")),
                    )
                if not any(t in ctype for t in _OK_TYPES if t) and ctype:
                    return FetchResult(url, str(resp.url), depth, prio, False,
                                       status=resp.status, error=f"content_type:{ctype[:40]}")
                declared = int(resp.headers.get("content-length") or 0)
                if declared and declared > max_bytes:
                    return FetchResult(url, str(resp.url), depth, prio, False,
                                       status=resp.status, error="too_large")
                chunks, total = [], 0
                async for chunk in resp.content.iter_chunked(65536):
                    total += len(chunk)
                    if total > max_bytes:
                        return FetchResult(url, str(resp.url), depth, prio, False,
                                           status=resp.status, error="too_large")
                    chunks.append(chunk)
                return FetchResult(url, str(resp.url), depth, prio, True, status=resp.status,
                                   content_type=ctype, body=b"".join(chunks))
        finally:
            sem.release()
    except asyncio.TimeoutError:
        return FetchResult(url, url, depth, prio, False, error="timeout")
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - network/DNS/redirect errors all become results
        return FetchResult(url, url, depth, prio, False, error=f"{type(e).__name__}")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

@dataclass
class CrawlConfig:
    language: str
    seeds: list[str]
    out_dir: Path
    state_path: Path
    host_prefixes: dict[str, str] = field(default_factory=dict)
    concurrency: int = 128
    per_domain: int = 8
    delay: float = 0.35
    timeout: float = 12.0
    max_bytes: int = 3_000_000
    max_attempts: int = 3
    max_pages: int = 1_000_000
    max_urls: int = 5_000_000
    max_depth: int = 10
    min_chars: int = 500
    min_ratio: float = 0.55
    min_markers: int = 2
    target_words: int = 0
    user_agent: str = ""
    log_every: int = 500


def _backoff(res: FetchResult) -> float:
    """Retry delay: honour Retry-After when the server gave one, else exponential backoff."""
    if res.retry_after:
        return res.retry_after
    return 30.0


def _handle(res: FetchResult, cfg: CrawlConfig, store: CorpusStore, writer, allowed) -> int:
    """Processes one fetch result: enqueue links, extract/accept an article. Returns 0 or 1."""
    if not res.ok:
        # Robots denials and oversize bodies are permanent; transport errors and 5xx/429 are not.
        permanent = res.error in ("robots_disallowed", "too_large") or res.error.startswith(
            "content_type:"
        )
        retryable = (not permanent) and (
            res.status is None or res.status in RETRY_STATUSES
        )
        if retryable:
            store.requeue(res.requested_url, _backoff(res))
        else:
            store.finish(res.requested_url, False, res.error)
        return 0

    host = registered_host(urlsplit(res.final_url).hostname or "")

    if looks_like_xml(res.body, res.content_type, res.final_url):
        if res.depth < cfg.max_depth:
            batch = []
            for link in extract_xml_links(res.body, res.final_url):
                n = normalize_url(link, res.final_url)
                if n and in_scope(n, allowed, cfg.host_prefixes):
                    h = registered_host(urlsplit(n).hostname or "")
                    batch.append((n, h, res.depth + 1, url_priority(n, from_sitemap=True)))
            store.add_urls(batch)
        store.finish(res.requested_url, True)
        return 0

    page = extract_page(res.body, res.final_url, res.content_type)

    # Links are enqueued before the acceptance checks: a listing page that yields no article of
    # its own is still how a sitemap-less site's archive gets discovered.
    if res.depth < cfg.max_depth:
        batch = []
        for link in page.links:
            n = normalize_url(link, res.final_url)
            if n and in_scope(n, allowed, cfg.host_prefixes):
                h = registered_host(urlsplit(n).hostname or "")
                batch.append((n, h, res.depth + 1, url_priority(n)))
        store.add_urls(batch)

    store.finish(res.requested_url, True)

    if len(page.text) < cfg.min_chars:
        return 0
    ratio, markers = devanagari_score(page.text, cfg.language)
    if ratio < cfg.min_ratio or markers < cfg.min_markers:
        return 0

    canonical = normalize_url(page.canonical_url or res.final_url) or res.final_url
    digest = content_hash(page.text)
    n_words = len(_WORD_RE.findall(page.text))
    if not store.add_article(digest, canonical, len(page.text), n_words):
        return 0

    writer.write({
        "url": canonical,
        "title": page.title,
        "text": page.text,
        "content_sha256": digest,
        "host": host,
        "fetched_at": _now(),
        "source_type": "manual",
        "published_at": page.published_at,
        "devanagari_ratio": round(ratio, 4),
        "word_count": n_words,
    })
    return 1


async def run_fast_crawl(cfg: CrawlConfig) -> dict:
    """Runs until max_pages, target_words, or frontier exhaustion. Safe to interrupt and rerun."""
    from lma.crawl.async_crawler import ShardWriter  # same shard schema as the other crawler

    allowed = frozenset(registered_host(urlsplit(s).hostname or "") for s in cfg.seeds)
    store = CorpusStore(cfg.state_path, max_urls=cfg.max_urls)
    writer = ShardWriter(cfg.out_dir)

    # Bootstrap: seed roots plus each origin's conventional sitemap locations.
    boot: list[tuple[str, str, int, int]] = []
    for seed in cfg.seeds:
        n = normalize_url(seed)
        if not n:
            continue
        parts = urlsplit(n)
        host = registered_host(parts.hostname or "")
        boot.append((n, host, 0, PRIO_ROOT))
        origin = f"{parts.scheme}://{parts.netloc}"
        for path in SITEMAP_PATHS:
            sm = normalize_url(origin + path)
            if sm:
                boot.append((sm, host, 0, PRIO_SITEMAP))
    store.add_urls(boot)
    store.maybe_commit(force=True)

    words0 = store.totals()[2]
    docs = pages = 0
    n_art = n_words = 0
    connector = aiohttp.TCPConnector(
        limit=cfg.concurrency, limit_per_host=cfg.per_domain,
        ttl_dns_cache=600, keepalive_timeout=30, enable_cleanup_closed=True,
    )
    headers = {
        "User-Agent": cfg.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "hi,ne,en;q=0.7",
    }
    inflight: dict[asyncio.Task, tuple[str, str]] = {}
    started = time.monotonic()

    try:
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            limiter = RobotsAndRateLimiter(session, cfg.user_agent, cfg.delay, cfg.per_domain)

            while True:
                n_art, _, n_words = store.totals()
                if cfg.target_words and n_words >= cfg.target_words:
                    logger.info("reached target_words=%d", cfg.target_words)
                    break
                if cfg.max_pages and pages >= cfg.max_pages:
                    logger.info("reached max_pages=%d", cfg.max_pages)
                    break

                free = cfg.concurrency - len(inflight)
                if cfg.max_pages:
                    free = min(free, cfg.max_pages - pages - len(inflight))
                if free > 0:
                    per_host: dict[str, int] = {}
                    for _, h in inflight.values():
                        per_host[h] = per_host.get(h, 0) + 1
                    room = {d: cfg.per_domain - per_host.get(d, 0) for d in store.domains()}
                    for url, _dom, depth, prio in store.claim(free, room, cfg.max_attempts):
                        task = asyncio.create_task(
                            fetch(session, limiter, url, depth, prio, cfg.max_bytes, cfg.timeout)
                        )
                        inflight[task] = (url, registered_host(urlsplit(url).hostname or ""))

                if not inflight:
                    wait = store.next_retry_in()
                    if wait is not None and wait <= 30.0:
                        store.maybe_commit(force=True)
                        await asyncio.sleep(min(wait + 0.1, 30.0))
                        continue
                    logger.info("frontier exhausted")
                    break

                done, _pending = await asyncio.wait(
                    set(inflight.keys()), timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    url, _host = inflight.pop(task)
                    try:
                        res = task.result()
                    except asyncio.CancelledError:
                        store.requeue(url, 0)
                        continue
                    except Exception as e:  # noqa: BLE001 - a worker bug must not lose the row
                        store.finish(url, False, f"worker:{type(e).__name__}")
                        continue
                    pages += 1
                    docs += _handle(res, cfg, store, writer, allowed)

                store.maybe_commit()
                if cfg.log_every and pages and pages % cfg.log_every < len(done):
                    n_art, _, n_words = store.totals()
                    rate = pages / max(1e-9, time.monotonic() - started)
                    logger.info(
                        "pages=%d articles=%d words=%d pending=%d %.1f pages/s",
                        pages, n_art, n_words, store.pending_count(), rate,
                    )
    finally:
        for task in inflight:
            task.cancel()
        store.maybe_commit(force=True)
        n_art, _n_chars, n_words = store.totals()
        store.close()
        writer.close()

    elapsed = time.monotonic() - started
    return {
        "pages_fetched": pages,
        "articles_new": docs,
        "articles_total": n_art,
        "words_total": n_words,
        "words_this_run": n_words - words0,
        "elapsed_s": round(elapsed, 1),
        "pages_per_s": round(pages / max(1e-9, elapsed), 2),
    }
