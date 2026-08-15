"""Tests for lma.crawl.linkextract -- the BFS fallback for sites with no usable sitemap."""
from lma.crawl.linkextract import extract_links

BASE = "https://example.com/news/"


def test_extracts_same_domain_links_only():
    html = """
    <html><body>
        <a href="/news/article-1">A</a>
        <a href="https://example.com/news/article-2">B</a>
        <a href="https://other.com/article-3">C (offsite, must be excluded)</a>
    </body></html>
    """
    links = extract_links(html, BASE)
    assert "https://example.com/news/article-1" in links
    assert "https://example.com/news/article-2" in links
    assert not any("other.com" in u for u in links)


def test_filters_denylisted_path_segments():
    html = """
    <html><body>
        <a href="/login">Login</a>
        <a href="/search?q=x">Search</a>
        <a href="/tag/politics">Tag</a>
        <a href="/category/sports">Category</a>
        <a href="/news/real-article-slug">Real article</a>
    </body></html>
    """
    links = extract_links(html, BASE)
    assert links == ["https://example.com/news/real-article-slug"]


def test_filters_asset_links():
    html = """
    <html><body>
        <a href="/images/photo.jpg">img</a>
        <a href="/style.css">css</a>
        <a href="/news/article-1">article</a>
    </body></html>
    """
    links = extract_links(html, BASE)
    assert links == ["https://example.com/news/article-1"]


def test_ignores_non_http_and_fragment_links():
    html = """
    <html><body>
        <a href="#top">anchor</a>
        <a href="mailto:x@example.com">mail</a>
        <a href="javascript:void(0)">js</a>
        <a href="/news/article-1">article</a>
    </body></html>
    """
    links = extract_links(html, BASE)
    assert links == ["https://example.com/news/article-1"]


def test_resolves_relative_urls_against_base():
    html = '<html><body><a href="../about">About</a><a href="article-2">Rel</a></body></html>'
    links = extract_links(html, BASE)
    assert "https://example.com/news/article-2" in links
    assert not any("/about" in u for u in links)  # denylisted


def test_deduplicates_query_string_variants():
    html = """
    <html><body>
        <a href="/news/article-1?utm_source=fb">a</a>
        <a href="/news/article-1?utm_source=tw">b</a>
    </body></html>
    """
    links = extract_links(html, BASE)
    assert links == ["https://example.com/news/article-1"]


def test_respects_max_links_cap():
    html = "<html><body>" + "".join(f'<a href="/news/a{i}">x</a>' for i in range(100)) + "</body></html>"
    links = extract_links(html, BASE, max_links=10)
    assert len(links) == 10


def test_malformed_html_does_not_raise():
    assert extract_links("<html><body><a href=", BASE) == []


def test_archive_index_and_events_hub_pages_are_filtered():
    # Regression test: amarujala.com's /news-archives, /news-archives/auto-archives, and
    # /events hub pages have no article body of their own -- confirmed empirically to extract
    # as pure cookie-notice boilerplate (see test_filters.py's matching regression test).
    html = """
    <html><body>
        <a href="/news-archives">Archives index</a>
        <a href="/news-archives/auto-archives">Auto archives</a>
        <a href="/events">Events</a>
        <a href="/uttar-pradesh/meerut/some-real-article-slug-12345">Real article</a>
    </body></html>
    """
    links = extract_links(html, BASE)
    assert links == ["https://example.com/uttar-pradesh/meerut/some-real-article-slug-12345"]
