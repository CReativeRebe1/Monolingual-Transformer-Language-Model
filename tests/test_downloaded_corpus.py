"""Tests for lma.crawl.downloaded_corpus -- offline, via a monkeypatched `datasets.load_dataset`.

Live end-to-end behaviour was verified manually against the real HuggingFace Hub during
development: wikimedia/wikipedia (hi) streams and writes real documents successfully, while
oscar-corpus/OSCAR-2301 and uonlp/CulturaX are confirmed gated (require `huggingface-cli login`
with an account that has accepted their terms -- documented in the top-level README's
reproduction steps) and ai4bharat/IndicCorpV2 required correcting the source config from an
assumed `config=<lang>` scheme to its actual `config="indiccorp_v2", split="hin_Deva"/"npi_Deva"`
scheme. These tests lock in the resulting behaviour (budget stopping, filtering, error handling,
record schema) without depending on network access or Hub availability at test time.
"""
import gzip
import json

import pytest

from lma.crawl import downloaded_corpus as dc


class _FakeStreamingDataset:
    """Mimics an HF streaming dataset: iterable of dicts, one field per example."""

    def __init__(self, examples):
        self._examples = examples

    def __iter__(self):
        return iter(self._examples)


def test_ingest_source_stops_at_token_budget(tmp_path, monkeypatch):
    examples = [{"text": " ".join(["word"] * 100), "title": f"Doc {i}"} for i in range(20)]

    def fake_load_dataset(repo_id, config, split, streaming):
        return _FakeStreamingDataset(examples)

    monkeypatch.setattr(dc, "load_dataset", fake_load_dataset, raising=False)
    # patch the lazy import target inside the datasets module namespace used by ingest_source
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    cfg = {"name": "fake", "repo_id": "fake/repo", "config": "hi", "text_field": "text", "split": "train"}
    report = dc.ingest_source(cfg, tmp_path, token_budget=250)

    assert report.status == "ok"
    # 100 tokens/doc: needs 3 docs to reach/exceed a 250-token budget.
    assert report.documents_written == 3
    assert report.approx_tokens_written >= 250


def test_ingest_source_filters_short_documents(tmp_path, monkeypatch):
    examples = [
        {"text": "short", "title": "too short"},
        {"text": " ".join(["word"] * 100), "title": "long enough"},
    ]

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: _FakeStreamingDataset(examples))

    cfg = {"name": "fake", "repo_id": "fake/repo", "config": "hi", "text_field": "text", "split": "train"}
    report = dc.ingest_source(cfg, tmp_path, token_budget=10_000, min_chars=200)

    assert report.documents_written == 1  # the short doc was filtered out


def test_ingest_source_written_records_have_expected_schema(tmp_path, monkeypatch):
    examples = [{"text": " ".join(["शब्द"] * 60), "title": "नमूना लेख"}]
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: _FakeStreamingDataset(examples))

    cfg = {"name": "wikipedia", "repo_id": "wikimedia/wikipedia", "config": "20231101.hi", "text_field": "text", "split": "train"}
    dc.ingest_source(cfg, tmp_path, token_budget=10_000)

    shard_path = tmp_path / "downloaded_wikipedia.jsonl.gz"
    assert shard_path.exists()
    with gzip.open(shard_path, "rt", encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["source_type"] == "downloaded"
    assert record["title"] == "नमूना लेख"
    assert record["host_source_name"] == "wikipedia"
    assert "content_sha256" in record and len(record["content_sha256"]) == 64


def test_ingest_source_handles_gated_or_missing_dataset_gracefully(tmp_path, monkeypatch):
    import datasets

    def raise_gated(*args, **kwargs):
        raise Exception("Dataset 'x' is a gated dataset on the Hub.")

    monkeypatch.setattr(datasets, "load_dataset", raise_gated)

    cfg = {"name": "oscar", "repo_id": "oscar-corpus/OSCAR-2301", "config": "hi", "text_field": "text", "split": "train"}
    report = dc.ingest_source(cfg, tmp_path, token_budget=1000)

    assert report.status == "gated_or_error"
    assert report.documents_written == 0
    assert "gated" in report.error.lower()


def test_ingest_unknown_language_raises(tmp_path):
    with pytest.raises(ValueError):
        dc.ingest("french", tmp_path, token_budget_per_source=1000)


def test_ingest_respects_sources_filter(tmp_path, monkeypatch):
    calls = []

    import datasets

    def fake_load_dataset(repo_id, config, split, streaming):
        calls.append(repo_id)
        return _FakeStreamingDataset([{"text": " ".join(["w"] * 60), "title": "t"}])

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    dc.ingest("hindi", tmp_path, token_budget_per_source=1000, sources=["wikipedia"])
    assert calls == ["wikimedia/wikipedia"]
