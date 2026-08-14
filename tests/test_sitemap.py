"""Tests for lma.crawl.sitemap -- offline, using httpx.MockTransport (no real network calls).

The property under test is the total-sitemap-fetch budget: a deeply nested sitemap index (as
bbc.com's real global archive is) must not cause fetch_urls_from_sitemap to make more than
max_sitemaps_total requests in total, regardless of how many sub-sitemaps it links to.
"""
import xml.etree.ElementTree as ET

import httpx
import pytest

from lma.crawl.sitemap import fetch_urls_from_sitemap

NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def _sitemap_index_xml(sub_urls: list[str]) -> bytes:
    entries = "".join(f"<sitemap><loc>{u}</loc></sitemap>" for u in sub_urls)
    return f'<?xml version="1.0"?><sitemapindex xmlns="{NS}">{entries}</sitemapindex>'.encode()


def _urlset_xml(page_urls: list[str]) -> bytes:
    entries = "".join(f"<url><loc>{u}</loc></url>" for u in page_urls)
    return f'<?xml version="1.0"?><urlset xmlns="{NS}">{entries}</urlset>'.encode()


@pytest.fixture
def request_counter():
    return {"count": 0}


def _make_deeply_nested_transport(request_counter, n_subsitemaps: int = 200):
    """Simulates a top-level index fanning out to n_subsitemaps leaf sitemaps, each with 1 URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        request_counter["count"] += 1
        url = str(request.url)
        if url == "https://example.com/sitemap_index.xml":
            subs = [f"https://example.com/sub_{i}.xml" for i in range(n_subsitemaps)]
            return httpx.Response(200, content=_sitemap_index_xml(subs))
        # any sub_N.xml
        return httpx.Response(200, content=_urlset_xml([f"{url}/page"]))

    return httpx.MockTransport(handler)


@pytest.mark.anyio
async def test_sitemap_fetch_respects_total_budget(request_counter):
    transport = _make_deeply_nested_transport(request_counter, n_subsitemaps=200)
    async with httpx.AsyncClient(transport=transport) as client:
        urls = await fetch_urls_from_sitemap(
            client, "https://example.com/sitemap_index.xml", max_sitemaps_total=40
        )
    # 1 request for the top-level index + at most 39 sub-sitemap requests = 40 total.
    assert request_counter["count"] <= 40
    # Each successfully-fetched sub-sitemap yields exactly one URL.
    assert len(urls) <= 39


@pytest.mark.anyio
async def test_sitemap_fetch_under_budget_gets_everything(request_counter):
    transport = _make_deeply_nested_transport(request_counter, n_subsitemaps=5)
    async with httpx.AsyncClient(transport=transport) as client:
        urls = await fetch_urls_from_sitemap(
            client, "https://example.com/sitemap_index.xml", max_sitemaps_total=40
        )
    assert len(urls) == 5


@pytest.mark.anyio
async def test_flat_urlset_returns_all_urls(request_counter):
    def handler(request: httpx.Request) -> httpx.Response:
        request_counter["count"] += 1
        return httpx.Response(
            200, content=_urlset_xml(["https://example.com/a", "https://example.com/b"])
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        urls = await fetch_urls_from_sitemap(client, "https://example.com/sitemap.xml")
    assert urls == ["https://example.com/a", "https://example.com/b"]
    assert request_counter["count"] == 1


@pytest.mark.anyio
async def test_non_200_response_returns_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        urls = await fetch_urls_from_sitemap(client, "https://example.com/missing.xml")
    assert urls == []


@pytest.mark.anyio
async def test_malformed_xml_returns_empty_list_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not valid xml <<<")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        urls = await fetch_urls_from_sitemap(client, "https://example.com/broken.xml")
    assert urls == []
