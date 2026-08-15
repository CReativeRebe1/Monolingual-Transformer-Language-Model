"""Tests for lma.crawl.wiki_dump -- streaming Wikipedia dump parsing.

Builds a small synthetic pages-articles dump (same XML shape and bz2 compression as the real
Wikimedia export) so the streaming reader, namespace/redirect filtering, wikitext stripping, and
the multiprocessing path are all exercised without downloading a ~1GB archive.
"""
import bz2
from pathlib import Path

from lma.crawl.wiki_dump import iter_pages, parse_dump, strip_wikitext

PAGE_TMPL = """  <page>
    <title>{title}</title>
    <ns>{ns}</ns>
    {redirect}
    <revision><text xml:space="preserve">{text}</text></revision>
  </page>
"""

ARTICLE = (
    "'''दिल्ली''' [[भारत]] की {{राजधानी}} है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं। "
    "यह शहर अपनी ऐतिहासिक इमारतों और संस्कृति के लिए जाना जाता है और हर साल लाखों पर्यटक यहाँ आते हैं। "
    "दिल्ली में मौसम गर्मियों में बहुत गर्म और सर्दियों में काफी ठंडा रहता है इसलिए यात्रा का समय महत्वपूर्ण है।\n"
    "== सन्दर्भ ==\n"
    "* [http://example.com कड़ी एक]\n"
    "* [http://example.org कड़ी दो]\n"
)


def _write_dump(path: Path, pages: list[str]) -> Path:
    xml = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">\n' + "".join(pages) + "</mediawiki>\n"
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        fh.write(xml)
    return path


def test_strip_wikitext_removes_markup():
    out = strip_wikitext(ARTICLE)
    assert "[[" not in out and "]]" not in out
    assert "{{" not in out and "}}" not in out
    assert "'''" not in out
    assert "दिल्ली" in out and "भारत" in out


def test_strip_wikitext_drops_reference_sections():
    out = strip_wikitext(ARTICLE)
    # The "सन्दर्भ" (References) section and its external-link list must not survive.
    assert "example.com" not in out
    assert "सन्दर्भ" not in out


def test_strip_wikitext_survives_malformed_input():
    assert strip_wikitext("{{{{{unclosed [[[[") is not None  # must not raise


def test_iter_pages_yields_main_namespace_articles(tmp_path):
    dump = _write_dump(tmp_path / "d.xml.bz2", [
        PAGE_TMPL.format(title="दिल्ली", ns="0", redirect="", text=ARTICLE),
    ])
    pages = list(iter_pages(dump, "hi.wikipedia.org"))
    assert len(pages) == 1
    site, title, wikitext = pages[0]
    assert site == "hi.wikipedia.org"
    assert title == "दिल्ली"
    assert "भारत" in wikitext


def test_iter_pages_skips_redirects_and_other_namespaces(tmp_path):
    dump = _write_dump(tmp_path / "d.xml.bz2", [
        PAGE_TMPL.format(title="Article", ns="0", redirect="", text=ARTICLE),
        PAGE_TMPL.format(title="Redir", ns="0", redirect='<redirect title="Article" />', text=ARTICLE),
        PAGE_TMPL.format(title="Talk:Article", ns="1", redirect="", text=ARTICLE),
        PAGE_TMPL.format(title="Template:X", ns="10", redirect="", text=ARTICLE),
    ])
    titles = [t for _, t, _ in iter_pages(dump, "hi.wikipedia.org")]
    assert titles == ["Article"]


def test_parse_dump_writes_records_via_pool(tmp_path):
    dump = _write_dump(tmp_path / "d.xml.bz2", [
        PAGE_TMPL.format(title=f"लेख{i}", ns="0", redirect="", text=ARTICLE) for i in range(5)
    ])

    written = []

    class FakeWriter:
        def write(self, rec):
            written.append(rec)

    n = parse_dump(dump, "hi.wikipedia.org", FakeWriter(), workers=2)
    assert n == 5
    assert len(written) == 5
    for rec in written:
        assert rec["source_type"] == "downloaded"
        assert rec["host"] == "hi.wikipedia.org"
        assert rec["url"].startswith("https://hi.wikipedia.org/wiki/")
        assert "[[" not in rec["text"]
        assert "content_sha256" in rec


def test_parse_dump_respects_max_docs(tmp_path):
    dump = _write_dump(tmp_path / "d.xml.bz2", [
        PAGE_TMPL.format(title=f"लेख{i}", ns="0", redirect="", text=ARTICLE) for i in range(20)
    ])

    class FakeWriter:
        def __init__(self): self.n = 0
        def write(self, rec): self.n += 1

    w = FakeWriter()
    n = parse_dump(dump, "hi.wikipedia.org", w, workers=2, max_docs=6)
    assert n == 6


def test_parse_dump_skips_stub_articles(tmp_path):
    # Below MIN_CHARS after stripping -> must not be written.
    dump = _write_dump(tmp_path / "d.xml.bz2", [
        PAGE_TMPL.format(title="Stub", ns="0", redirect="", text="छोटा लेख।"),
    ])

    class FakeWriter:
        def __init__(self): self.n = 0
        def write(self, rec): self.n += 1

    w = FakeWriter()
    assert parse_dump(dump, "hi.wikipedia.org", w, workers=2) == 0
