"""
Stage 3c -- perplexity, document quality, and coherence scoring for Hindi.

Three scorers, in increasing subtlety:

* `ppl(text)` -- perplexity under a char-4-gram model fitted on trusted Hindi text (Wikipedia).
  Reuses `lma.text.quality.QualityScorer`; the threshold comes from a percentile of the trusted
  distribution rather than a constant, so it adapts to what the model actually learned.
* `quality(text)` -- Yi's Document Quality Filtering. A logistic classifier separating
  Wikipedia-like prose from raw crawl, trained on cheap structural features rather than vocabulary,
  so it generalises to crawl domains it never saw. Returns P(wikipedia-like).
* `coherent(text)` -- Yi's Coherence Scorer. Mean pairwise TF-IDF cosine between adjacent
  paragraphs, which catches documents stitched together from unrelated fragments (a common
  extraction failure: three article teasers concatenated into one "document").

**Coherence bypasses short and thin documents by design.** With fewer than
`coherence_min_paragraphs` paragraphs or `coherence_min_words` words, the cosine is
undefined-to-noisy: a genuinely fine three-sentence Devanagari news blurb scores near zero purely
for lack of vocabulary overlap. Such documents **pass** -- on a low-resource corpus an unmeasurable
score must never become a rejection. Note the binding constraint is paragraph count, not length:
`min_words = 40` already gates upstream, so a 50-word bypass would exempt only the narrow 40-50
band while leaving the real problem in place. You need >=3 paragraphs to tell "stitched together"
from "two paragraphs making related points", and ~150 words before TF-IDF vectors stop being mostly
zeros.

The classifier and the perplexity model are *fitted per language* on that language's own trusted
text -- fitting once on Hindi and reusing it for Nepali is exactly the "threshold tuned on another
language" mistake Lecture 3 warns about.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hindi.clean.config import DEFAULT, CleanConfig  # noqa: E402
from hindi.clean.rules import paras, words  # noqa: E402
from hindi.clean.words import STOPWORDS  # noqa: E402
from lma.text.quality import QualityScorer, choose_threshold  # noqa: E402

__all__ = ["ppl", "fit_ppl", "quality", "QualityModel", "coherent", "coherence_score",
           "calibrate_coherence", "features"]


# ------------------------------------------------------------------ perplexity

def fit_ppl(trusted_texts: list[str], *, config: CleanConfig | None = None):
    """Fits the char-n-gram model on trusted Hindi text; returns (scorer, threshold)."""
    cfg = config or DEFAULT
    scorer = QualityScorer()
    scorer.fit(trusted_texts)
    threshold = choose_threshold(scorer.fit_ppl_dist(trusted_texts), percentile=cfg.ppl_percentile)
    return scorer, threshold


def ppl(text: str, scorer: QualityScorer) -> float:
    return scorer.perplexity(text)


# ------------------------------------------------------------------ document quality

def features(text: str) -> list[float]:
    """Structural features for the quality classifier -- deliberately vocabulary-free.

    Using structure rather than words is what lets a model fitted on Wikipedia-vs-crawl transfer to
    crawl domains it never saw: it learns "well-formed prose has sentences of this length, this
    punctuation density, this stopword rate", not "well-formed prose mentions these topics".
    """
    ws = words(text)
    n_words = max(len(ws), 1)
    n_chars = max(len(text), 1)
    sentences = [s for s in text.replace("।", ".").split(".") if s.strip()]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return [
        math.log1p(n_words),
        sum(len(w) for w in ws) / n_words,                       # mean word length
        n_words / max(len(sentences), 1),                        # mean sentence length in words
        len(set(ws) & STOPWORDS) / min(n_words, 50),             # stopword density (capped)
        sum(1 for c in text if c in ".,;:!?()\"'-।") / n_chars,   # punctuation density
        sum(1 for c in text if c.isdigit()) / n_chars,            # digit density
        sum(1 for c in text if c.isupper()) / n_chars,            # latin caps: markup/nav debris
        n_words / max(len(lines), 1),                             # words per line: nav = tiny
        len(set(ws)) / n_words,                                   # type-token ratio
    ]


@dataclass
class QualityModel:
    """Logistic regression on `features`, trained with plain gradient descent.

    Hand-rolled rather than scikit-learn's LogisticRegression because the assignment asks for the
    pipeline to be built from scratch where it is the object of study, and because nine features
    and a few thousand examples need nothing more. Features are standardised at fit time --
    log-word-count and punctuation density differ by orders of magnitude, and unscaled gradient
    descent would be dominated by whichever feature happens to be largest.
    """

    lr: float = 0.5
    epochs: int = 400
    l2: float = 1e-4
    w: list[float] = field(default_factory=list)
    b: float = 0.0
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    def _standardise(self, x: list[float]) -> list[float]:
        return [(xi - m) / s for xi, m, s in zip(x, self.mean, self.std)]

    def fit(self, good_texts: list[str], bad_texts: list[str]) -> None:
        X = [features(t) for t in good_texts] + [features(t) for t in bad_texts]
        y = [1.0] * len(good_texts) + [0.0] * len(bad_texts)
        if not X:
            raise ValueError("no training text given")

        d = len(X[0])
        self.mean = [sum(row[j] for row in X) / len(X) for j in range(d)]
        self.std = [
            max(math.sqrt(sum((row[j] - self.mean[j]) ** 2 for row in X) / len(X)), 1e-6)
            for j in range(d)
        ]
        Xs = [self._standardise(row) for row in X]
        self.w = [0.0] * d
        self.b = 0.0

        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for row, target in zip(Xs, y):
                z = self.b + sum(wi * xi for wi, xi in zip(self.w, row))
                err = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0))) - target
                for j in range(d):
                    gw[j] += err * row[j]
                gb += err
            n = len(Xs)
            self.w = [wi - self.lr * (gw[j] / n + self.l2 * wi) for j, wi in enumerate(self.w)]
            self.b -= self.lr * gb / n

    def score(self, text: str) -> float:
        """P(wikipedia-like) in [0, 1]. 0.5 if the model was never fitted."""
        if not self.w:
            return 0.5
        row = self._standardise(features(text))
        z = self.b + sum(wi * xi for wi, xi in zip(self.w, row))
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def quality(text: str, model: QualityModel, *, config: CleanConfig | None = None) -> tuple[bool, float]:
    cfg = config or DEFAULT
    s = model.score(text)
    return s >= cfg.quality_min_score, s


# ------------------------------------------------------------------ coherence

def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0.0) + 1.0
    return counts


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


COHERENCE_NGRAM = 4


def _char_grams(text: str, n: int) -> list[str]:
    s = re.sub(r"\s+", " ", text.strip())
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def coherence_score(text: str) -> float:
    """Mean TF-IDF cosine between consecutive paragraphs. Higher = more on one topic.

    **Character 4-grams, not words.** Measured on real Hindi paragraphs, word-level cosine barely
    separates a coherent article from a stitched-together one (0.031 vs 0.029) because Hindi is
    inflected: consecutive paragraphs of the same article share stems, not exact surface forms, so
    exact word matching misses the very overlap being measured. Character 4-grams match the shared
    stem and separated the same pair 2x better (0.052 vs 0.024). This is a deliberate divergence
    from Yi's word-level formulation, made on measurement.

    **Smoothed IDF, log((N+1)/(1+df)) + 1, not log(N/(1+df)).** The textbook form is exactly zero
    for any term appearing in 2 of 3 paragraphs -- which is to say it zeroes out precisely the
    shared terms the metric exists to detect, and returned 0.0000 for every document, coherent or
    not. Found by running it on real text; it would have silently rejected the whole corpus.

    IDF is computed *within* the document, treating its paragraphs as the collection, so the scorer
    needs no corpus-wide state and can run in a worker process one document at a time.
    """
    ps = paras(text)
    if len(ps) < 2:
        return 1.0
    tokenised = [_char_grams(p, COHERENCE_NGRAM) for p in ps]
    df: dict[str, int] = {}
    for toks in tokenised:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = len(tokenised)
    vectors = []
    for toks in tokenised:
        vectors.append(
            {
                t: (1 + math.log(v)) * (math.log((n + 1) / (1 + df[t])) + 1)
                for t, v in _tf(toks).items()
            }
        )
    sims = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    return sum(sims) / len(sims) if sims else 1.0


def calibrate_coherence(trusted_texts: list[str], *, config: CleanConfig | None = None) -> float:
    """Threshold = a low percentile of trusted (Wikipedia) documents' coherence scores.

    Setting it from the corpus rather than hardcoding a constant is the point: the absolute cosine
    values depend on paragraph length and script, so a number that looks reasonable in a docstring
    is a number tuned on nothing. Documents scoring below almost every Wikipedia article are the
    ones being targeted. Returns the config default if too few trusted documents are measurable.
    """
    cfg = config or DEFAULT
    scores = sorted(
        coherence_score(t)
        for t in trusted_texts
        if len(paras(t)) >= cfg.coherence_min_paragraphs and len(words(t)) >= cfg.coherence_min_words
    )
    if len(scores) < 100:
        return cfg.coherence_min_cosine
    idx = min(int(len(scores) * cfg.coherence_percentile / 100.0), len(scores) - 1)
    return scores[idx]


def coherent(text: str, *, config: CleanConfig | None = None) -> tuple[bool, float | None]:
    """Returns (is_coherent, score). Score is None when the document bypassed the check.

    Bypassed documents pass. See the module docstring for why that direction is deliberate.
    """
    cfg = config or DEFAULT
    if len(paras(text)) < cfg.coherence_min_paragraphs or len(words(text)) < cfg.coherence_min_words:
        return True, None
    s = coherence_score(text)
    return s >= cfg.coherence_min_cosine, s
