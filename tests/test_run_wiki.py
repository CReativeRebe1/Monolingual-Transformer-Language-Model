"""Tests for scripts/run_wiki.py's project selection -- specifically that an unavailable
project/language pair (e.g. ne.wikisource, which has no dump at all -- verified live, see
lma/crawl/wiki_dump.py's WIKI_CODE comment) fails with a clear message instead of building a
URL for a dump that does not exist and failing deep inside the download with a raw 404.
"""
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_wiki.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wiki", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hindi_supports_all_three_projects():
    module = _load_module()
    assert set(module.WIKI_CODE["hindi"]) == {"wikipedia", "wikisource", "wiktionary"}


def test_nepali_wikisource_is_absent():
    module = _load_module()
    assert "wikisource" not in module.WIKI_CODE["nepali"]


def test_main_rejects_unavailable_project_with_clear_error(monkeypatch, capsys):
    module = _load_module()
    monkeypatch.setattr(
        "sys.argv", ["run_wiki.py", "--lang", "nepali", "--project", "wikisource"]
    )
    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "nepali" in captured.err and "wikisource" in captured.err


def test_site_domain_reflects_project_not_hardcoded_to_wikipedia():
    module = _load_module()
    assert module.LANG_CODE["hindi"] == "hi"
    assert module.LANG_CODE["nepali"] == "ne"


def test_dump_url_uses_project_specific_code():
    module = _load_module()
    assert module.WIKI_CODE["hindi"]["wikisource"] == "hiwikisource"
    assert module.WIKI_CODE["hindi"]["wikipedia"] == "hiwiki"
    assert module.DUMP_URL.format(wiki="hiwikisource") == (
        "https://dumps.wikimedia.org/hiwikisource/latest/hiwikisource-latest-pages-articles.xml.bz2"
    )
