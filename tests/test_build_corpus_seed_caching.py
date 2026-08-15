"""Tests for scripts/build_corpus.py's _train_models -- specifically that the
trusted Wikipedia seed pull is skip-if-exists, not unconditional.

build_corpus.py is meant to be re-run repeatedly as the crawl accumulates more data (confirmed in
practice: it has been re-run several times over the course of Phase-1 to refresh the committed
corpus snapshot -- see PHASE1_LOG.md). ingest() has no internal caching of its own; every call
re-streams the full token budget from the Hub from scratch. The primary language's seed pull was
originally unconditional while the contrast-language seed already correctly checked existence
first -- an inconsistency that wasted a real network call on every single re-run.
"""
import gzip
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_corpus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_seed_shard(seed_dir: Path, texts: list[str]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(seed_dir / "downloaded_wikipedia.jsonl.gz", "wt", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def test_skips_ingest_when_seed_already_exists(tmp_path, monkeypatch):
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    other_lang_dir = tmp_path / "nepali"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    # Pre-populate both languages' seeds, as if a previous build_corpus.py run already pulled them.
    _write_seed_shard(lang_dir / "data" / "raw" / "seed", ["एक हिन्दी वाक्य यहाँ है भाषा पहचान हेतु।"] * 5)
    _write_seed_shard(other_lang_dir / "data" / "raw" / "seed", ["यो नेपाली वाक्य हो भाषा पहिचानको लागि।"] * 5)

    calls = []
    monkeypatch.setattr(module, "ingest", lambda *a, **k: calls.append(a))

    module._train_models("hindi", lang_dir)

    assert calls == [], f"ingest() should not have been called; existing seed should be reused, got {calls}"


def test_pulls_ingest_when_seed_missing(tmp_path, monkeypatch):
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    calls = []

    def fake_ingest(language, out_dir, token_budget_per_source, sources=None):
        calls.append(language)
        _write_seed_shard(out_dir, [f"sample {language} text यहाँ हो"] * 5)

    monkeypatch.setattr(module, "ingest", fake_ingest)

    module._train_models("hindi", lang_dir)

    assert "hindi" in calls  # primary language pulled since no seed existed
    assert "nepali" in calls  # contrast language also pulled since it didn't exist either


def test_empty_seed_dir_is_treated_as_missing(tmp_path, monkeypatch):
    # An empty seed/ directory (e.g. created but never populated, or all shards deleted) must
    # still trigger a real pull, not be mistaken for "already has data".
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    other_lang_dir = tmp_path / "nepali"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    (lang_dir / "data" / "raw" / "seed").mkdir(parents=True)  # exists but empty
    _write_seed_shard(other_lang_dir / "data" / "raw" / "seed", ["यो नेपाली वाक्य हो।"] * 5)

    calls = []

    def fake_ingest(language, out_dir, token_budget_per_source, sources=None):
        calls.append(language)
        _write_seed_shard(out_dir, [f"sample {language} text"] * 5)

    monkeypatch.setattr(module, "ingest", fake_ingest)

    module._train_models("hindi", lang_dir)

    assert "hindi" in calls
