"""
Learned quality filter -- stage 4 of the cleaning cascade (the stage after heuristic rules,
before dedup). Heuristic rules (lma/text/filters.py) catch structurally malformed documents;
this stage catches documents that are structurally fine but *read badly* -- boilerplate-heavy,
keyword-stuffed, or machine-generated text that passes every heuristic check yet is clearly not
naturally-written prose to a native reader.

Implementation: a character-level n-gram language model trained on a small trusted seed corpus
(Wikipedia text for the target language, whose text is known-clean by construction) scores every
candidate document by its mean per-character negative log-likelihood under that model -- i.e. its
model perplexity. Documents whose perplexity is far outside the distribution observed on more
trusted text are dropped. This mirrors the "learned filter" stage of the Yi pipeline (Lecture 3),
implemented here as a from-scratch char-n-gram LM rather than a downloaded classifier, consistent
with the project's "train everything yourself" constraint on tokenizers/models.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


def _char_ngrams(text: str, n: int) -> list[str]:
    padded = f"{'~' * (n - 1)}{text}"  # pad so the first real characters have full context
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


@dataclass
class QualityScorer:
    """A char-n-gram LM trained on trusted seed text; scores documents by mean NLL per character.

    Lower perplexity = more predictable text given the trusted-text statistics = more likely to
    be naturally-written prose in the target language/domain. Machine-generated spam, keyword
    dumps, and heavily corrupted OCR output tend to have anomalously HIGH perplexity under a
    model trained on clean prose (they don't follow the target language's normal character
    statistics), which is the signal this filter acts on.
    """

    n: int = 4
    smoothing_k: float = 0.1
    _ngram_counts: Counter = None  # type: ignore[assignment]
    _context_counts: Counter = None  # type: ignore[assignment]
    _vocab: set[str] = None  # type: ignore[assignment]
    _fitted: bool = False

    def __post_init__(self):
        self._ngram_counts = Counter()
        self._context_counts = Counter()
        self._vocab = set()

    def fit(self, trusted_texts: list[str]) -> None:
        for text in trusted_texts:
            grams = _char_ngrams(text, self.n)
            for g in grams:
                context, char = g[:-1], g[-1]
                self._ngram_counts[g] += 1
                self._context_counts[context] += 1
                self._vocab.add(char)
        self._fitted = True

    def perplexity(self, text: str) -> float:
        """Mean-per-character perplexity of `text` under the fitted model. Lower = more expected."""
        if not self._fitted:
            raise RuntimeError("QualityScorer.fit() must be called before perplexity()")
        if not text:
            return float("inf")
        grams = _char_ngrams(text, self.n)
        vocab_size = max(len(self._vocab), 1)
        log_prob_sum = 0.0
        for g in grams:
            context, char = g[:-1], g[-1]
            count = self._ngram_counts.get(g, 0)
            context_total = self._context_counts.get(context, 0)
            prob = (count + self.smoothing_k) / (context_total + self.smoothing_k * vocab_size)
            log_prob_sum += -math.log(prob)
        mean_nll = log_prob_sum / len(grams)
        return math.exp(mean_nll)

    def fit_ppl_dist(self, texts: list[str]) -> list[float]:
        """Perplexities of a batch of (typically held-out, trusted-domain) texts -- used to pick
        a keep/drop threshold empirically (e.g. the 95th/99th percentile) rather than guessing
        one, and to draw the perplexity-distribution figure in the Phase-1 report."""
        return [self.perplexity(t) for t in texts]


def choose_threshold(reference_perplexities: list[float], percentile: float = 99.0) -> float:
    """A percentile of a trusted reference distribution, used as the keep/drop cutoff.

    Setting the threshold from the trusted-text distribution itself (rather than an arbitrary
    constant) means the cutoff adapts to whatever the n-gram model actually learned as "normal"
    for this language/domain, and the chosen percentile is reported alongside the resulting
    keep-rate in the Phase-1 dataset statistics.
    """
    if not reference_perplexities:
        raise ValueError("reference_perplexities must be non-empty")
    sorted_vals = sorted(reference_perplexities)
    idx = min(int(len(sorted_vals) * percentile / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]
