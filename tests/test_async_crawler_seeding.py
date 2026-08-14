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


@pytest.mark.anyio
async def test_sitemap_budget_is_shared_across_a_domains_multiple_candidates(tmp_path):
    # Reproduces the real bug: a domain whose robots.txt declares its own sitemap AND has one
    # of the 4 common-path candidates also resolve to a large index must not get a fresh
    # max_sitemaps_total budget for EACH candidate -- the budget must be shared per-domain.
    import xml.etree.ElementTree as ET

    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"

    def _index_xml(n_sub):
        entries = "".join(f"<sitemap><loc>https://big.com/sub_{i}.xml</loc></sitemap>" for i in range(n_sub))
        return f'<?xml version="1.0"?><sitemapindex xmlns="{ns}">{entries}</sitemapindex>'.encode()

    def _leaf_xml(url):
        return f'<?xml version="1.0"?><urlset xmlns="{ns}"><url><loc>{url}</loc></url></urlset>'.encode()

    request_log = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(str(request.url))
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, content=b"Sitemap: https://big.com/sitemap-a.xml\n")
        if path == "/sitemap-a.xml":
            return httpx.Response(200, content=_index_xml(200))  # huge index, candidate #1
        if path == "/sitemap.xml":
            return httpx.Response(200, content=_index_xml(200))  # ALSO huge, candidate #2
        if path.startswith("/sub_"):
            return httpx.Response(200, content=_leaf_xml(f"https://big.com{path}/page"))
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with Frontier(tmp_path / "f.sqlite3") as frontier:
            await _seed_frontier_from_domains(client, frontier, ["big.com"])

    sitemap_requests = [u for u in request_log if "sitemap" in u or "sub_" in u]
    # With a shared budget of 40, total sitemap-document fetches across BOTH large-index
    # candidates must stay bounded near 40, not balloon toward 2 x 40 (one per candidate) or
    # worse. Generous upper bound to allow for the two initial index fetches themselves.
    assert len(sitemap_requests) <= 45, f"expected a shared, bounded budget; got {len(sitemap_requests)} requests"
