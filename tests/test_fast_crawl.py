"""Tests for lma.crawl.fast -- the high-throughput crawler.

Pure-unit tests for URL/priority/extraction/scoring logic, store tests against a temp SQLite
file, and an end-to-end integration test against a local aiohttp server so the default suite
never touches a public site.
"""
import asyncio
import gzip
import json
from pathlib import Path

import pytest
from aiohttp import web

from lma.crawl.fast import (
    CorpusStore, CrawlConfig, PRIO_ARTICLE, PRIO_LOW, PRIO_SITEMAP, PRIO_SM_LINK,
    content_hash, devanagari_score, extract_page, extract_xml_links, in_scope, looks_like_xml,
    maybe_gunzip, normalize_url, registered_host, run_fast_crawl, url_priority,
)

HINDI = ("यह एक समाचार लेख है और इसमें कई वाक्य हैं। यह लेख भारत की राजधानी दिल्ली के बारे में है "
         "और यह बताता है कि वहाँ का मौसम कैसा है। इस लेख को पढ़ने वाले लोग जान पाएंगे कि दिल्ली "
         "में सर्दियों में क्या होता है और गर्मियों में क्या होता है। यह जानकारी उपयोगी है। ") * 4


# ---------------------------------------------------------------- URL handling

def test_normalize_strips_tracking_and_fragment_and_default_port():
    u = normalize_url("https://Example.COM:443/a//b?utm_source=x&b=2&a=1#frag")
    assert u == "https://example.com/a/b?a=1&b=2"


def test_normalize_rejects_bad_schemes_ports_media_and_skip_paths():
    assert normalize_url("ftp://example.com/x") is None
    assert normalize_url("https://example.com:notaport/x") is None
    assert normalize_url("https://example.com/img.JPG") is None
    assert normalize_url("https://example.com/login/form") is None


def test_normalize_resolves_relative_against_base():
    assert normalize_url("/story-1", "https://example.com/section/") == "https://example.com/story-1"


def test_registered_host_drops_www():
    assert registered_host("WWW.Example.com.") == "example.com"


def test_scope_allows_subdomains_but_not_suffix_confusion():
    allowed = frozenset({"example.com"})
    assert in_scope("https://news.example.com/a", allowed, {})
    assert in_scope("https://example.com/a", allowed, {})
    assert not in_scope("https://notexample.com/a", allowed, {})


def test_scope_applies_host_path_prefix_but_sitemaps_bypass_it():
    allowed, prefixes = frozenset({"bbc.com"}), {"bbc.com": "/hindi/"}
    assert in_scope("https://bbc.com/hindi/story", allowed, prefixes)
    assert not in_scope("https://bbc.com/arabic/story", allowed, prefixes)
    assert in_scope("https://bbc.com/sitemap.xml", allowed, prefixes)


def test_priority_ordering():
    assert url_priority("https://a.com/sitemap.xml") == PRIO_SITEMAP
    assert url_priority("https://a.com/x", from_sitemap=True) == PRIO_SM_LINK
    assert url_priority("https://a.com/2024/05/story") == PRIO_ARTICLE
    assert url_priority("https://a.com/news-1234567.html") == PRIO_ARTICLE
    assert url_priority("https://a.com/tag/politics") == PRIO_LOW


# ---------------------------------------------------------------- sitemaps

def test_extract_xml_links_handles_namespaces_and_gzip():
    xml = (b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           b"<url><loc>https://a.com/1</loc></url><url><loc>https://a.com/2</loc></url></urlset>")
    assert extract_xml_links(xml, "https://a.com/sitemap.xml") == ["https://a.com/1", "https://a.com/2"]
    assert extract_xml_links(gzip.compress(xml), "https://a.com/sitemap.xml.gz") == [
        "https://a.com/1", "https://a.com/2"]


def test_extract_xml_links_on_malformed_returns_empty():
    assert extract_xml_links(b"<urlset><broken", "https://a.com/sitemap.xml") == []


def test_maybe_gunzip_passes_through_bad_gzip():
    assert maybe_gunzip(b"\x1f\x8bnotgzip", "https://a.com/x.gz") == b"\x1f\x8bnotgzip"


def test_looks_like_xml_detects_by_body_and_url():
    assert looks_like_xml(b"<?xml version='1.0'?>", "", "https://a.com/x")
    assert looks_like_xml(b"<html>", "", "https://a.com/sitemap.xml")
    assert not looks_like_xml(b"<html>", "text/html", "https://a.com/story")


# ---------------------------------------------------------------- extraction

def test_extract_page_pulls_prose_links_and_canonical():
    html = f"""<html><head><title>T</title>
      <link rel="canonical" href="/canon"/><meta property="og:title" content="OG Title"/></head>
      <body><nav><a href="/nav">nav</a></nav><script>var x=1;</script>
      <article><p>{HINDI}</p></article><a href="/story-2">next</a></body></html>""".encode()
    page = extract_page(html, "https://a.com/s", "text/html; charset=utf-8")
    assert page.title == "OG Title"
    assert "दिल्ली" in page.text
    assert "var x=1" not in page.text
    assert page.canonical_url == "https://a.com/canon"
    assert "https://a.com/story-2" in page.links


def test_extract_page_prefers_longer_json_ld_body():
    long_body = HINDI * 3
    ld = json.dumps({"@type": "NewsArticle", "headline": "H", "articleBody": long_body,
                     "datePublished": "2024-01-01"})
    html = f'<html><body><p>{HINDI}</p><script type="application/ld+json">{ld}</script></body></html>'.encode()
    page = extract_page(html, "https://a.com/s", "text/html")
    assert len(page.text) > len(HINDI) * 2
    assert page.published_at == "2024-01-01"


def test_extract_page_survives_broken_markup():
    page = extract_page(b"<html><p>hi<div><span>", "https://a.com/s", "text/html")
    assert isinstance(page.text, str)


# ---------------------------------------------------------------- language + hashing

def test_devanagari_score_accepts_hindi_rejects_english():
    ratio, markers = devanagari_score(HINDI, "hindi")
    assert ratio > 0.9 and markers >= 2
    ratio_en, markers_en = devanagari_score("This is an ordinary English article.", "hindi")
    assert ratio_en == 0.0 and markers_en == 0


def test_content_hash_is_stable_across_case_and_whitespace():
    assert content_hash("Hello   World") == content_hash("hello world")


# ---------------------------------------------------------------- store

def test_store_dedups_urls_and_respects_max_urls(tmp_path):
    s = CorpusStore(tmp_path / "s.db", max_urls=3)
    assert s.add_urls([("https://a.com/1", "a.com", 0, 20)] * 2) == 1   # batch-local dedup
    assert s.add_urls([("https://a.com/1", "a.com", 0, 20)]) == 0       # already stored
    s.add_urls([(f"https://a.com/{i}", "a.com", 0, 20) for i in range(2, 10)])
    assert s.pending_count() == 3                                       # max_urls honoured
    s.close()


def test_store_claims_by_priority_and_per_domain_room(tmp_path):
    s = CorpusStore(tmp_path / "s.db")
    s.add_urls([("https://a.com/low", "a.com", 0, 5), ("https://a.com/high", "a.com", 0, 100)])
    claimed = s.claim(10, {"a.com": 1}, max_attempts=3)
    assert len(claimed) == 1 and claimed[0][0].endswith("/high")   # priority wins
    s.close()


def test_store_rotates_across_domains(tmp_path):
    s = CorpusStore(tmp_path / "s.db")
    s.add_urls([(f"https://a.com/{i}", "a.com", 0, 20) for i in range(50)])
    s.add_urls([("https://b.com/1", "b.com", 0, 20)])
    hosts = {d for _, d, _, _ in s.claim(4, {"a.com": 8, "b.com": 8}, 3)}
    assert hosts == {"a.com", "b.com"}   # a big domain must not starve a small one
    s.close()


def test_store_requeue_delays_eligibility(tmp_path):
    s = CorpusStore(tmp_path / "s.db")
    s.add_urls([("https://a.com/1", "a.com", 0, 20)])
    s.claim(1, {"a.com": 8}, 3)
    s.requeue("https://a.com/1", 60)
    assert s.claim(1, {"a.com": 8}, 3) == []
    assert 0 < s.next_retry_in() <= 60
    s.close()


def test_store_rejects_duplicate_hash_and_url(tmp_path):
    s = CorpusStore(tmp_path / "s.db")
    assert s.add_article("h1", "https://a.com/1", 10, 2) is True
    assert s.add_article("h1", "https://a.com/2", 10, 2) is False   # same content
    assert s.add_article("h2", "https://a.com/1", 10, 2) is False   # same canonical URL
    s.close()


def test_store_recovers_interrupted_fetching_rows(tmp_path):
    path = tmp_path / "s.db"
    s = CorpusStore(path)
    s.add_urls([("https://a.com/1", "a.com", 0, 20)])
    s.claim(1, {"a.com": 8}, 3)
    s.maybe_commit(force=True)
    s.close()
    s2 = CorpusStore(path)          # simulates a crash mid-fetch
    assert s2.pending_count() == 1  # the claimed row came back as pending
    s2.close()


# ---------------------------------------------------------------- integration

@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _start_server():
    routes = web.RouteTableDef()

    @routes.get("/robots.txt")
    async def robots(_):
        return web.Response(text="User-agent: *\nDisallow: /private\n")

    @routes.get("/sitemap.xml")
    async def sitemap(request):
        # Absolute URLs built from the live request: sitemaps carry absolute <loc> values, and
        # extract_xml_links deliberately drops anything that is not http(s)-absolute.
        b = str(request.url.origin())
        body = ('<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                f"<url><loc>{b}/2024/01/good</loc></url>"
                f"<url><loc>{b}/short</loc></url>"
                f"<url><loc>{b}/english</loc></url>"
                f"<url><loc>{b}/private/secret</loc></url></urlset>")
        return web.Response(text=body, content_type="application/xml")

    @routes.get("/2024/01/good")
    async def good(_):
        return web.Response(text=f"<html><body><article><p>{HINDI}</p></article></body></html>",
                            content_type="text/html")

    @routes.get("/dupe")
    async def dupe(_):
        return web.Response(text=f"<html><body><article><p>{HINDI}</p></article></body></html>",
                            content_type="text/html")

    @routes.get("/short")
    async def short(_):
        return web.Response(text="<html><body><p>छोटा</p></body></html>", content_type="text/html")

    @routes.get("/english")
    async def english(_):
        return web.Response(text="<html><body><p>" + ("English text here. " * 60) + "</p></body></html>",
                            content_type="text/html")

    @routes.get("/private/secret")
    async def secret(_):
        return web.Response(text="should never be fetched", content_type="text/html")

    @routes.get("/")
    async def root(_):
        return web.Response(text='<html><body><a href="/dupe">d</a></body></html>',
                            content_type="text/html")

    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, port


@pytest.mark.anyio
async def test_end_to_end_crawl_against_local_server(tmp_path):
    runner, port = await _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        cfg = CrawlConfig(
            language="hindi", seeds=[f"{base}/"],
            out_dir=tmp_path / "shards", state_path=tmp_path / "state.db",
            concurrency=8, per_domain=4, delay=0.0, timeout=10,
            max_pages=50, min_chars=200, min_ratio=0.55, min_markers=2,
            user_agent="test-agent",
        )
        summary = await run_fast_crawl(cfg)
    finally:
        await runner.cleanup()

    assert summary["pages_fetched"] > 0
    # /2024/01/good and /dupe have identical text -> exact dedup keeps exactly one.
    assert summary["articles_total"] == 1

    shard = next((tmp_path / "shards").glob("shard_*.jsonl.gz"))
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["source_type"] == "manual"          # counts toward the manual quota
    assert "दिल्ली" in rec["text"]
    assert set(("url", "title", "text", "content_sha256", "host", "fetched_at")) <= set(rec)


@pytest.mark.anyio
async def test_robots_disallowed_path_is_never_fetched(tmp_path):
    runner, port = await _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        cfg = CrawlConfig(
            language="hindi", seeds=[f"{base}/"],
            out_dir=tmp_path / "shards", state_path=tmp_path / "state.db",
            concurrency=4, per_domain=2, delay=0.0, timeout=10, max_pages=50,
            min_chars=200, user_agent="test-agent",
        )
        await run_fast_crawl(cfg)
    finally:
        await runner.cleanup()

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    row = conn.execute(
        "SELECT status,last_error FROM urls WHERE url LIKE '%/private/secret'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "failed" and row[1] == "robots_disallowed"


@pytest.mark.anyio
async def test_crawl_resumes_without_refetching(tmp_path):
    runner, port = await _start_server()
    base = f"http://127.0.0.1:{port}"
    try:
        common = dict(
            language="hindi", seeds=[f"{base}/"], out_dir=tmp_path / "shards",
            state_path=tmp_path / "state.db", concurrency=4, per_domain=2, delay=0.0,
            timeout=10, min_chars=200, user_agent="test-agent",
        )
        first = await run_fast_crawl(CrawlConfig(max_pages=2, **common))
        second = await run_fast_crawl(CrawlConfig(max_pages=50, **common))
    finally:
        await runner.cleanup()

    assert first["pages_fetched"] == 2
    # The second run continues the frontier rather than restarting it, and the corpus total is
    # cumulative across runs (article totals live in the persistent store).
    assert second["articles_total"] >= first["articles_total"]
    assert second["articles_total"] == 1
