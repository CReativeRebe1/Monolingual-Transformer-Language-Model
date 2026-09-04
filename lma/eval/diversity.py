"""
Diversity and degeneration diagnostics for generated text.

This is where a 24M-parameter model's real failure mode shows up. Perplexity can look entirely
reasonable while greedy decoding collapses into "और और और और" or loops a four-word phrase forever --
the model has learned a sound distribution over next tokens and still has no mechanism preventing it
from walking into a fixed point. Perplexity cannot see that, because it never samples; it only ever
scores the true continuation. These metrics sample and look at what comes out.

* **distinct-n** (Li et al. 2016): unique n-grams / total n-grams, over the pooled generations. Near
  1.0 means every n-gram is new; collapsing toward 0 means the model is repeating itself. Pooled
  rather than per-sample and averaged, so that *cross*-sample repetition -- every prompt producing
  the same paragraph, which greedy decoding does -- is visible too.
* **repetition rate**: the fraction of tokens inside an n-gram that has already occurred earlier in
  the same sample. This is the within-sample loop detector.
* **self-BLEU** (Zhu et al. 2018): each generation scored against the others as references. High
  self-BLEU means the model produces the same text regardless of prompt. It is the metric that
  exposes a model that looks fluent because it has memorised one fluent paragraph.

Reported over a temperature/top-p grid, because every one of these trades off against fluency and a
single operating point says nothing: greedy decoding maximises fluency and minimises diversity, and
the interesting question is where the knee is.
"""
from __future__ import annotations

from collections import Counter

from lma.eval.textmetrics import bleu


def distinct_n(texts: list[str], n: int) -> float:
    """Unique n-grams / total n-grams across the pooled texts. 0 when there are none."""
    total = Counter()
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            total[tuple(toks[i:i + n])] += 1
    n_total = sum(total.values())
    return len(total) / n_total if n_total else 0.0


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of a sample's n-grams that repeat an n-gram seen earlier in the SAME sample.

    Within-sample, deliberately: this is the loop detector. A model stuck emitting the same phrase
    scores near 1.0 here while distinct-n over a pooled corpus might still look acceptable if the
    other samples are varied.
    """
    toks = text.split()
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    seen, repeats = set(), 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams)


def self_bleu(texts: list[str], max_samples: int = 50) -> float:
    """Mean BLEU of each generation against one other generation. 0-100; higher = less diverse.

    Each hypothesis is scored against a single other sample rather than all of them, which is a
    cheaper approximation of the usual definition: the full version is O(n^2) BLEU computations and
    the ranking it produces over decoding settings is the same. Capped at `max_samples` because that
    quadratic bites quickly.
    """
    texts = [t for t in texts if t.strip()][:max_samples]
    if len(texts) < 2:
        return 0.0
    hyps = texts
    refs = texts[1:] + texts[:1]        # each sample paired with its neighbour, cyclically
    return bleu(hyps, refs)


def mean_length(texts: list[str]) -> float:
    return sum(len(t.split()) for t in texts) / max(len(texts), 1)


def diversity_report(texts: list[str]) -> dict[str, float]:
    """Every diagnostic for one set of generations, i.e. one decoding configuration."""
    return {
        "n_samples": len(texts),
        "mean_length_words": round(mean_length(texts), 2),
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, 2), 4),
        "distinct_3": round(distinct_n(texts, 3), 4),
        "repetition_rate_4gram": round(
            sum(repetition_rate(t, 4) for t in texts) / max(len(texts), 1), 4
        ),
        "self_bleu": round(self_bleu(texts), 4),
    }
