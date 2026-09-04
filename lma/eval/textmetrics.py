"""
BLEU, chrF and ROUGE-L, implemented from first principles and cross-checked against `sacrebleu`.

Same pattern Phase 1 used for MinHash (validated against `datasketch`) and the tokenizer (validated
against HuggingFace `tokenizers`): implement it, then prove it agrees with an independent
implementation. A metric that is quietly wrong produces plausible-looking numbers and there is no
internal consistency check that would reveal it.

**A caveat that belongs in the report, not buried here.** These metrics were designed for machine
translation, where a reference translation is genuinely the target. Open-ended language modelling
has no single correct continuation: a fluent, sensible Hindi paragraph that differs from the held-out
one scores near zero, and so does gibberish. The assignment names them, so they are computed and
reported -- but perplexity, bits-per-byte and the diversity diagnostics are the numbers that
actually say something about these models, and the report says so rather than presenting a BLEU of
1.2 as if it were a finding.

chrF is the least bad of the three here. It scores character n-grams, so it survives Devanagari's
rich inflectional morphology -- where a correct word form differing in one suffix costs word-BLEU an
entire n-gram match but costs chrF only a fraction of one.
"""
from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(hypotheses: list[str], references: list[str], max_n: int = 4) -> float:
    """Corpus-level BLEU with the standard brevity penalty, on whitespace tokens. Returns 0-100.

    Corpus-level, not the mean of sentence-level scores: BLEU's precision is a ratio of summed
    counts, and averaging per-sentence scores gives a different (and conventionally wrong) number,
    which is the most common way BLEU is misreported.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")

    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        h, r = hyp.split(), ref.split()
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            h_grams, r_grams = _ngrams(h, n), _ngrams(r, n)
            # "Clipped" counts: an n-gram repeated 5 times in the hypothesis but once in the
            # reference counts once. Without the clip, emitting one correct word 100 times scores a
            # perfect unigram precision, which is the degenerate output BLEU exists to punish.
            clipped[n - 1] += sum(min(c, r_grams[g]) for g, c in h_grams.items())
            totals[n - 1] += max(sum(h_grams.values()), 0)

    if any(t == 0 for t in totals) or all(c == 0 for c in clipped):
        return 0.0
    # Geometric mean in log space. A single zero-count order sends BLEU to 0, which is correct and
    # is why short hypotheses so often score exactly 0 on 4-gram BLEU.
    log_precisions = []
    for c, t in zip(clipped, totals):
        if c == 0:
            return 0.0
        log_precisions.append(math.log(c / t))
    geo_mean = math.exp(sum(log_precisions) / max_n)
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * geo_mean * brevity


def chrf(hypotheses: list[str], references: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Corpus-level chrF: F-score over character n-grams up to `max_n`, recall weighted `beta` times
    precision. Returns 0-100.

    beta=2 is sacrebleu's default and the value the chrF paper recommends: recall matters more than
    precision for generation, because omitting content is worse than adding it.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")

    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        match = hyp_total = ref_total = 0
        for hyp, ref in zip(hypotheses, references):
            # Whitespace removed: chrF scores character sequences, and leaving spaces in makes the
            # metric partly a word-boundary metric, which is not what it measures.
            h = _ngrams(list(hyp.replace(" ", "")), n)
            r = _ngrams(list(ref.replace(" ", "")), n)
            match += sum(min(c, r[g]) for g, c in h.items())
            hyp_total += sum(h.values())
            ref_total += sum(r.values())
        precisions.append(match / hyp_total if hyp_total else 0.0)
        recalls.append(match / ref_total if ref_total else 0.0)

    avg_p = sum(precisions) / max_n
    avg_r = sum(recalls) / max_n
    if avg_p == 0 and avg_r == 0:
        return 0.0
    b2 = beta ** 2
    return 100.0 * (1 + b2) * avg_p * avg_r / (b2 * avg_p + avg_r)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, computed over one rolling row.

    The full table is O(len(a) * len(b)) in memory; at 128-token continuations that is trivial, but
    the rolling row is the same number of lines and keeps the function usable on whole documents.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b):
            curr.append(prev[j] + 1 if x == y else max(prev[j + 1], curr[j]))
        prev = curr
    return prev[-1]


def rouge_l(hypotheses: list[str], references: list[str], beta: float = 1.2) -> float:
    """Mean sentence-level ROUGE-L F-score over whitespace tokens. Returns 0-100.

    ROUGE-L is a subsequence metric, not an n-gram one: it rewards getting the right words in the
    right ORDER without requiring them to be adjacent. That makes it the most forgiving of the three
    for a small model whose continuations are topically right and locally clumsy -- which is exactly
    the failure mode a 24M-parameter model has.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    scores = []
    for hyp, ref in zip(hypotheses, references):
        h, r = hyp.split(), ref.split()
        lcs = _lcs_length(h, r)
        if lcs == 0:
            scores.append(0.0)
            continue
        p, rec = lcs / len(h), lcs / len(r)
        b2 = beta ** 2
        scores.append((1 + b2) * p * rec / (rec + b2 * p))
    return 100.0 * sum(scores) / max(len(scores), 1)


def all_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    return {
        "bleu": round(bleu(hypotheses, references), 4),
        "chrf": round(chrf(hypotheses, references), 4),
        "rouge_l": round(rouge_l(hypotheses, references), 4),
    }
