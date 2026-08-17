"""
Stage 3a -- text metric filtering and repetitive document removal, for Hindi.

Two entry points:

* `check(text)` -- the heuristic cascade from Gopher (Rae et al., 2021) and C4 (Raffel et al.,
  2020), with Devanagari-aware thresholds. Every applicable rule runs; the verdict lists *all*
  rules a document violated, because the Phase-1 funnel needs per-rule counts and a
  short-circuiting cascade would attribute every rejection to whichever rule happens to be first.
* `repetitive(text)` -- Yi's separate "Repetitive Document Removal" step, which the earlier
  pipeline only partially covered (it caught repeated *lines* and nothing else). Repetition is
  measured four ways, because a spam page can be repetitive at any granularity: one dominant word,
  a dominant n-gram, a large share of text sitting inside repeated n-grams, or duplicated
  paragraphs *within* one document. Cross-document duplication is a different stage (dedup.py).

Both are Hindi copies rather than shared code: the thresholds are the thing that must be free to
diverge per language, and a shared implementation would make divergence a special case instead of
the default.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hindi.clean.config import DEFAULT, CleanConfig  # noqa: E402
from hindi.clean.words import BOILERPLATE_PHRASES, STOPWORDS  # noqa: E402

_BULLET_OR_ELLIPSIS_LINE_RE = re.compile(r"^\s*([•\-\*·▪]|\.{3,}|…)\s*")
_DIGIT_RE = re.compile(r"[0-9०-९]")
_WORD_SPLIT_RE = re.compile(r"\s+")


@dataclass
class Verdict:
    passed: bool
    failed: list[str] = field(default_factory=list)


def words(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT_RE.split(text.strip()) if w]


def paras(text: str) -> list[str]:
    """Paragraphs = blank-line-separated blocks, falling back to lines.

    The fallback matters for crawled text: trafilatura sometimes emits one line per paragraph with
    no blank line between them, and without the fallback such a document would look like a single
    giant paragraph and silently bypass every paragraph-level check.
    """
    blocks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(blocks) > 1:
        return blocks
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def _symbol_ratio(text: str, ws: list[str]) -> float:
    if not ws:
        return 1.0
    symbols = sum(1 for c in text if not (c.isalnum() or c.isspace() or "ऀ" <= c <= "ॿ"))
    return symbols / len(ws)


def _digit_ratio(text: str) -> float:
    return len(_DIGIT_RE.findall(text)) / len(text) if text else 0.0


def _bullet_frac(lines: list[str]) -> float:
    return sum(1 for ln in lines if _BULLET_OR_ELLIPSIS_LINE_RE.match(ln)) / len(lines) if lines else 0.0


def _repeated_line_frac(lines: list[str]) -> float:
    non_blank = [ln for ln in lines if ln.strip()]
    if not non_blank:
        return 0.0
    counts = Counter(non_blank)
    return sum(c for c in counts.values() if c > 1) / len(non_blank)


def _ngrams(seq: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def top_fraction(seq: list[str], n: int) -> float:
    """Share of all n-grams taken by the single most common n-gram (Gopher's repetition metric)."""
    grams = _ngrams(seq, n) if n > 1 else [(w,) for w in seq]
    if not grams:
        return 0.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


def dup_ngram_fraction(ws: list[str], n: int) -> float:
    """Share of the document's words that sit inside an n-gram occurring more than once.

    Gopher measures this over characters; words are used here because Devanagari character counts
    are distorted by matras (one phonological unit spans several code points), which would make a
    character-based threshold mean something different than it does for English.
    """
    grams = _ngrams(ws, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    covered = set()
    for i, g in enumerate(grams):
        if counts[g] > 1:
            covered.update(range(i, i + n))
    return len(covered) / len(ws)


def dup_para_fraction(text: str) -> float:
    """Share of paragraphs that are exact duplicates of another paragraph in the SAME document."""
    ps = paras(text)
    if not ps:
        return 0.0
    counts = Counter(ps)
    return sum(c for c in counts.values() if c > 1) / len(ps)


def repetitive(text: str, *, config: CleanConfig | None = None) -> Verdict:
    """Yi's Repetitive Document Removal step. Returns a Verdict listing every metric exceeded."""
    cfg = config or DEFAULT
    failed: list[str] = []
    ws = words(text)

    if top_fraction(ws, 1) > cfg.max_top_word_fraction:
        failed.append("top_word_fraction")
    if top_fraction(ws, 2) > cfg.max_top_2gram_fraction:
        failed.append("top_2gram_fraction")
    if top_fraction(ws, 3) > cfg.max_top_3gram_fraction:
        failed.append("top_3gram_fraction")
    if dup_ngram_fraction(ws, cfg.dup_ngram_n) > cfg.max_dup_ngram_fraction:
        failed.append("dup_ngram_fraction")
    if dup_para_fraction(text) > cfg.max_dup_para_fraction:
        failed.append("dup_paragraph_fraction")

    return Verdict(passed=not failed, failed=failed)


def check(text: str, *, config: CleanConfig | None = None) -> Verdict:
    """Text metric filtering. Includes `repetitive()`'s findings so callers get one verdict."""
    cfg = config or DEFAULT
    failed: list[str] = []

    n_chars = len(text)
    if n_chars < cfg.min_chars:
        failed.append("too_short")
    if n_chars > cfg.max_chars:
        failed.append("too_long")

    ws = words(text)
    if len(ws) < cfg.min_words:
        failed.append("too_few_words")
    if ws:
        mean_len = sum(len(w) for w in ws) / len(ws)
        if mean_len > cfg.max_mean_word_chars:
            failed.append("mean_word_length_too_high")
        if mean_len < cfg.min_mean_word_chars:
            failed.append("mean_word_length_too_low")

    if _symbol_ratio(text, ws) > cfg.max_symbol_to_word_ratio:
        failed.append("symbol_to_word_ratio_too_high")
    if _digit_ratio(text) > cfg.max_digit_char_ratio:
        failed.append("digit_char_ratio_too_high")

    lines = text.split("\n")
    if _bullet_frac(lines) > cfg.max_bullet_or_ellipsis_line_fraction:
        failed.append("bullet_line_fraction_too_high")
    if _repeated_line_frac(lines) > cfg.max_repeated_line_fraction:
        failed.append("repeated_line_fraction_too_high")

    if any(p in text for p in BOILERPLATE_PHRASES):
        failed.append("boilerplate_phrase_present")
    if text.lstrip().startswith("{"):
        # Devanagari prose essentially never begins with an ASCII '{'. This is a specific
        # trafilatura failure on amarujala.com's premium template, which leaks the page's JSON
        # metadata as literal text -- it matched 25% of raw Hindi records when first measured.
        failed.append("starts_with_json_metadata")

    if cfg.require_stopword_hit and len(set(ws) & STOPWORDS) < cfg.min_stopword_hits:
        failed.append("insufficient_stopword_evidence")

    failed.extend(repetitive(text, config=cfg).failed)
    return Verdict(passed=not failed, failed=failed)
