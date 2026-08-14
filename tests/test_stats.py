"""Tests for lma.text.stats -- pure computation, no plotting."""
import pytest

from lma.text.stats import (
    CorpusStats,
    DocumentRecord,
    PipelineFunnel,
    compute_corpus_stats,
    document_length_distribution,
)


def _doc(text, source_type="manual", host="example.com", n_tokens=None):
    return DocumentRecord(text=text, source_type=source_type, host=host, n_tokens=n_tokens)


def test_compute_corpus_stats_basic_counts_without_tokens():
    docs = [_doc("एक दो तीन"), _doc("चार पांच")]
    stats = compute_corpus_stats(docs)
    assert stats.n_documents == 2
    assert stats.n_words == 5
    assert stats.n_chars == sum(len(d.text) for d in docs)
    assert stats.n_tokens is None  # no n_tokens supplied anywhere
    assert stats.manual_fraction is None


def test_compute_corpus_stats_with_full_token_counts():
    docs = [
        _doc("एक दो तीन", source_type="manual", host="news.com", n_tokens=5),
        _doc("चार पांच", source_type="downloaded", host="wikipedia", n_tokens=3),
    ]
    stats = compute_corpus_stats(docs)
    assert stats.n_tokens == 8
    assert stats.manual_tokens == 5
    assert stats.downloaded_tokens == 3
    assert stats.manual_fraction == pytest.approx(5 / 8)


def test_compute_corpus_stats_docs_per_source():
    docs = [_doc("a", host="site1"), _doc("b", host="site1"), _doc("c", host="site2")]
    stats = compute_corpus_stats(docs)
    assert stats.docs_per_source == {"site1": 2, "site2": 1}


def test_compute_corpus_stats_partial_token_counts_degrades_to_none():
    # If even one document lacks n_tokens, token-derived aggregates must not silently
    # under-count -- they degrade to None rather than reporting a partial total as if complete.
    docs = [_doc("a", n_tokens=5), _doc("b", n_tokens=None)]
    stats = compute_corpus_stats(docs)
    assert stats.n_tokens is None
    assert stats.manual_fraction is None


def test_compute_corpus_stats_empty_input():
    stats = compute_corpus_stats([])
    assert stats.n_documents == 0
    assert stats.n_words == 0
    assert stats.n_tokens is None


def test_document_length_distribution_words():
    docs = [_doc("एक दो तीन"), _doc("चार")]
    lengths = document_length_distribution(docs, unit="words")
    assert lengths == [3, 1]


def test_document_length_distribution_chars():
    docs = [_doc("abc"), _doc("de")]
    lengths = document_length_distribution(docs, unit="chars")
    assert lengths == [3, 2]


def test_document_length_distribution_invalid_unit_raises():
    with pytest.raises(ValueError):
        document_length_distribution([_doc("a")], unit="bytes")


# --- PipelineFunnel ---

def test_pipeline_funnel_survival_rate():
    funnel = PipelineFunnel(
        stage_names=["collected", "lang_filtered", "heuristic_filtered", "deduped"],
        counts=[1000, 800, 600, 500],
    )
    assert funnel.survival_rate("collected") == 1.0
    assert funnel.survival_rate("deduped") == pytest.approx(0.5)


def test_pipeline_funnel_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        PipelineFunnel(stage_names=["a", "b"], counts=[1])


def test_pipeline_funnel_zero_initial_count_survival_rate_is_zero():
    funnel = PipelineFunnel(stage_names=["a", "b"], counts=[0, 0])
    assert funnel.survival_rate("b") == 0.0
