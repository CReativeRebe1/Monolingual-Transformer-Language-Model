"""Tests for lma.crawl.async_crawler._seed_frontier_from_domains -- specifically the per-domain
seeding-skip fix (see lma/crawl/frontier.py's seeded_domains table): a domain already marked
seeded must be skipped entirely (zero HTTP requests), while an unseeded domain must still be
seeded even if OTHER domains already have URLs in the frontier -- the exact bug found live,
where every domain past the first in a multi-domain list was silently starved across restarts.
"""
import httpx
import pytest

from lma.crawl.async_crawler import _seed_frontier_from_domains
from lma.crawl.frontier import Frontier


def _urlset_xml(urls):
    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    entries = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return f'<?xml version="1.0"?><urlset xmlns="{ns}">{entries}</urlset>'.encode()


def _make_transport(request_log, domain_urls: dict[str, list[str]]):
    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(str(request.url))
        host = request.url.host
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, content=b"")
        if path == "/sitemap.xml" and host in domain_urls:
            return httpx.Response(200, content=_urlset_xml(domain_urls[host]))
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.anyio
async def test_already_seeded_domain_makes_zero_requests(tmp_path):
    request_log = []
    transport = _make_transport(request_log, {"a.com": ["https://a.com/article-1"]})
    async with httpx.AsyncClient(transport=transport) as client:
        with Frontier(tmp_path / "f.sqlite3") as frontier:
            frontier.mark_domain_seeded("a.com", urls_found=0)
            await _seed_frontier_from_domains(client, frontier, ["a.com"])
            assert request_log == []  # never touched -- already seeded


@pytest.mark.anyio
async def test_unseeded_domain_is_seeded_even_when_another_domain_already_has_urls(tmp_path):
    # This is the exact bug: domain "a.com" was seeded in an earlier run (frontier already has
    # its URLs), but "b.com" was never reached. A restart must still seed "b.com".
    request_log = []
    transport = _make_transport(
        request_log, {"a.com": ["https://a.com/article-1"], "b.com": ["https://b.com/article-2"]}
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with Frontier(tmp_path / "f.sqlite3") as frontier:
            frontier.add("https://a.com/article-1", depth=1)
            frontier.mark_domain_seeded("a.com", urls_found=1)

            await _seed_frontier_from_domains(client, frontier, ["a.com", "b.com"])

            assert frontier.is_domain_seeded("b.com") is True
            assert frontier.seen("https://b.com/article-2") is True
            # a.com's sitemap must NOT have been re-fetched.
            assert not any("a.com" in url for url in request_log)
            assert any("b.com" in url for url in request_log)


@pytest.mark.anyio
async def test_domain_marked_seeded_after_successful_seeding(tmp_path):
    request_log = []
    transport = _make_transport(request_log, {"a.com": ["https://a.com/article-1"]})
    async with httpx.AsyncClient(transport=transport) as client:
        with Frontier(tmp_path / "f.sqlite3") as frontier:
            assert frontier.is_domain_seeded("a.com") is False
            await _seed_frontier_from_domains(client, frontier, ["a.com"])
            assert frontier.is_domain_seeded("a.com") is True
            assert frontier.seeded_domains()["a.com"] == 1


@pytest.mark.anyio
async def test_domain_with_no_sitemap_is_still_marked_seeded_via_homepage_fallback(tmp_path):
    request_log = []
    transport = _make_transport(request_log, {})  # no sitemap for anyone -> homepage fallback
    async with httpx.AsyncClient(transport=transport) as client:
        with Frontier(tmp_path / "f.sqlite3") as frontier:
            await _seed_frontier_from_domains(client, frontier, ["nosite.com"])
            assert frontier.is_domain_seeded("nosite.com") is True
            assert frontier.seen("https://nosite.com/") is True
