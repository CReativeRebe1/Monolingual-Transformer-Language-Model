"""
Dataset statistics: pure computation over a list of cleaned documents, no plotting here (that is
lma/viz/plots.py's job -- guideline #4 requires visualization and computation to live in
separate functions). Produces exactly the numbers the Phase-1 report's dataset-statistics
section and figures are built from: per-source token/doc counts, manual-vs-downloaded split,
length distributions, domain distribution, and cleaning-stage survival counts for the funnel
chart.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DocumentRecord:
    """The minimal fields every pipeline stage's output shares, regardless of whether the
    document came from the crawler, OCR, or a downloaded dataset."""

    text: str
    source_type: str  # "manual" | "downloaded"
    host: str  # domain (manual) or dataset repo id (downloaded), for the domain-distribution stat
    n_tokens: int | None = None  # filled in once a tokenizer exists; None during earlier stages


@dataclass
class CorpusStats:
    n_documents: int
    n_words: int
    n_chars: int
    n_bytes: int
    n_tokens: int | None  # None if no tokenizer was supplied
    manual_tokens: int | None
    downloaded_tokens: int | None
    manual_fraction: float | None
    docs_per_source: dict[str, int] = field(default_factory=dict)
    tokens_per_source: dict[str, int] = field(default_factory=dict)


def compute_corpus_stats(documents: list[DocumentRecord]) -> CorpusStats:
    """Aggregates the headline dataset-statistics numbers for the Phase-1 report.

    Token counts are only meaningful once documents carry n_tokens (post-tokenizer); if any
    document lacks it, the token-derived fields degrade to None rather than silently
    under-counting, so a caller cannot accidentally report a manual-fraction computed from a
    partially-tokenized corpus.
    """
    n_documents = len(documents)
    n_words = sum(len(d.text.split()) for d in documents)
    n_chars = sum(len(d.text) for d in documents)
    n_bytes = sum(len(d.text.encode("utf-8")) for d in documents)

    docs_per_source: Counter[str] = Counter()
    tokens_per_source: Counter[str] = Counter()
    manual_tokens = 0
    downloaded_tokens = 0
    have_all_token_counts = all(d.n_tokens is not None for d in documents) and n_documents > 0

    for d in documents:
        docs_per_source[d.host] += 1
        if have_all_token_counts:
            tokens_per_source[d.host] += d.n_tokens  # type: ignore[arg-type]
            if d.source_type == "manual":
                manual_tokens += d.n_tokens  # type: ignore[arg-type]
            elif d.source_type == "downloaded":
                downloaded_tokens += d.n_tokens  # type: ignore[arg-type]

    n_tokens = (manual_tokens + downloaded_tokens) if have_all_token_counts else None
    manual_fraction = (
        manual_tokens / n_tokens if have_all_token_counts and n_tokens else None
    )

    return CorpusStats(
        n_documents=n_documents,
        n_words=n_words,
        n_chars=n_chars,
        n_bytes=n_bytes,
        n_tokens=n_tokens,
        manual_tokens=manual_tokens if have_all_token_counts else None,
        downloaded_tokens=downloaded_tokens if have_all_token_counts else None,
        manual_fraction=manual_fraction,
        docs_per_source=dict(docs_per_source),
        tokens_per_source=dict(tokens_per_source) if have_all_token_counts else {},
    )


def document_length_distribution(documents: list[DocumentRecord], unit: str = "words") -> list[int]:
    """Per-document lengths (word or char count), for the length-histogram figure."""
    if unit == "words":
        return [len(d.text.split()) for d in documents]
    if unit == "chars":
        return [len(d.text) for d in documents]
    raise ValueError(f"unit must be 'words' or 'chars', got {unit!r}")


@dataclass
class PipelineFunnel:
    """Document counts surviving each named cleaning stage, in order -- the funnel chart."""

    stage_names: list[str]
    counts: list[int]

    def __post_init__(self):
        if len(self.stage_names) != len(self.counts):
            raise ValueError("stage_names and counts must be the same length")

    def survival_rate(self, stage_name: str) -> float:
        """Fraction of the ORIGINAL (first-stage) count still present at `stage_name`."""
        if self.counts[0] == 0:
            return 0.0
        idx = self.stage_names.index(stage_name)
        return self.counts[idx] / self.counts[0]
