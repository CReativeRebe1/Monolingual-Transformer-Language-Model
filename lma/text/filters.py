"""
Heuristic quality filters -- stage 3 of the Yi-style 5-stage cleaning cascade (Lecture 3).

Each filter is a small, independent, well-understood rule from the Gopher (Rae et al., 2021)
and C4 (Raffel et al., 2020) cleaning literature, adapted with Devanagari-aware thresholds
where the original English-tuned constants would misfire (e.g. "mean word length" measured in
characters means something different for an abugida script than for Latin script). Every rule
reports which specific check rejected a document, not just pass/fail, because the Phase-1 report
requires per-rule rejection counts -- the funnel chart showing what stage 3 actually removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A small, hand-curated stopword list is enough here: the filter only checks for the *presence*
# of at least a few closed-class function words, as a cheap "is this actually connected prose in
# the target language, not a keyword list / product SKU dump" signal -- not a full lexicon.
HINDI_STOPWORDS = frozenset(
    "और के की का को है हैं में से पर एक यह वह हो था थी थे कि लिए भी नहीं ने तो जो अगर लेकिन".split()
)
NEPALI_STOPWORDS = frozenset(
    "र को की का छ छन थियो थिए भएको लागि तर यदि पनि होइन ले मा हो यो त्यो अनि जुन".split()
)

_BULLET_OR_ELLIPSIS_LINE_RE = re.compile(r"^\s*([•\-\*·▪]|\.{3,}|…)\s*")
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-\s]?)?\d{10}(?!\d)")
_DEVANAGARI_DIGIT_OR_ASCII_DIGIT_RE = re.compile(r"[0-9०-९]")
_WORD_SPLIT_RE = re.compile(r"\s+")

# Boilerplate phrases that survive trafilatura's extraction because they are embedded inline in
# an article's markup rather than in an obviously-separate nav/footer element. Checked as
# substrings (case-sensitive; ASCII-only phrases are inherently case-relevant, Devanagari has no
# case). Kept short and specific deliberately, to avoid false-positives on legitimate prose that
# happens to mention e.g. "subscribe" as a verb.
BOILERPLATE_PHRASES = (
    "cookie policy",
    "javascript को सक्षम करें",  # "enable javascript" (Hindi)
    "जावास्क्रिप्ट सक्षम गर्नुहोस्",  # "enable javascript" (Nepali)
    "all rights reserved",
    "subscribe to our newsletter",
    "यह वेबसाइट कुकीज़ का उपयोग करती है",  # "this website uses cookies" (Hindi)
    # Observed directly in the crawled Hindi corpus (amarujala.com's cookie/data-collection
    # notice, verbatim on every section/category "hub" page that has no real article body --
    # e.g. /news-archives, /events, /kavya/*): 745 of ~950 heuristic-surviving documents in an
    # early corpus-build run were exact duplicates of this one disclaimer. Dedup correctly
    # collapsed them to a single copy, but the right fix is catching it here so it never
    # displaces real crawl budget in the final kept corpus at all.
    "हम डाटा संग्रह टूल्स",  # "we collect data via tools [like cookies]" (Hindi, amarujala.com)
    "अमर उजाला एप इंस्टॉल कर रजिस्टर करें",  # "install the Amar Ujala app and register" (app-install nag)
)


@dataclass
class FilterConfig:
    min_chars: int = 200
    max_chars: int = 200_000
    min_words: int = 40
    max_mean_word_chars: float = 25.0  # catches URL/hash/SKU-dump "words"
    min_mean_word_chars: float = 1.5   # catches single-char-token spam / broken segmentation
    max_symbol_to_word_ratio: float = 0.15
    max_digit_char_ratio: float = 0.30
    max_bullet_or_ellipsis_line_fraction: float = 0.30
    max_repeated_line_fraction: float = 0.20
    require_stopword_hit: bool = True
    min_stopword_hits: int = 2


@dataclass
class FilterVerdict:
    passed: bool
    failed_rules: list[str] = field(default_factory=list)


def _words(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT_RE.split(text.strip()) if w]


def _symbol_to_word_ratio(text: str, words: list[str]) -> float:
    if not words:
        return 1.0
    # "symbol" here = any char that is not a Devanagari letter/mark, ASCII letter, digit, or
    # whitespace -- i.e. punctuation-and-noise density relative to word count.
    symbol_chars = sum(
        1 for c in text if not (c.isalnum() or c.isspace() or "ऀ" <= c <= "ॿ")
    )
    return symbol_chars / len(words)


def _digit_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = len(_DEVANAGARI_DIGIT_OR_ASCII_DIGIT_RE.findall(text))
    return digits / len(text)


def _bullet_or_ellipsis_line_fraction(lines: list[str]) -> float:
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _BULLET_OR_ELLIPSIS_LINE_RE.match(ln))
    return hits / len(lines)


def _repeated_line_fraction(lines: list[str]) -> float:
    """Fraction of non-blank lines that are exact duplicates of another line in the document.

    Catches template/navigation debris that survived extraction as repeated short lines (e.g. a
    list of section names repeated in a sidebar block), which whole-document exact-dedup (a
    later pipeline stage) would not catch since it operates across documents, not within one.
    """
    non_blank = [ln for ln in lines if ln.strip()]
    if not non_blank:
        return 0.0
    counts: dict[str, int] = {}
    for ln in non_blank:
        counts[ln] = counts.get(ln, 0) + 1
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(non_blank)


def _has_boilerplate_phrase(text: str) -> bool:
    return any(phrase in text for phrase in BOILERPLATE_PHRASES)


def _stopword_hits(text: str, stopwords: frozenset[str]) -> int:
    tokens = set(_words(text))
    return len(tokens & stopwords)


def evaluate(text: str, *, language: str, config: FilterConfig | None = None) -> FilterVerdict:
    """Run the full heuristic cascade. language: 'hindi' or 'nepali' (selects the stopword list).

    All applicable rules are checked (not short-circuited on first failure), so a rejected
    document's failed_rules list is a complete diagnostic, and the Phase-1 report's per-rule
    rejection counts reflect every rule a document actually violated, not just the first one hit.
    """
    config = config or FilterConfig()
    failed: list[str] = []

    n_chars = len(text)
    if n_chars < config.min_chars:
        failed.append("too_short")
    if n_chars > config.max_chars:
        failed.append("too_long")

    words = _words(text)
    if len(words) < config.min_words:
        failed.append("too_few_words")

    if words:
        mean_word_chars = sum(len(w) for w in words) / len(words)
        if mean_word_chars > config.max_mean_word_chars:
            failed.append("mean_word_length_too_high")
        if mean_word_chars < config.min_mean_word_chars:
            failed.append("mean_word_length_too_low")

    if _symbol_to_word_ratio(text, words) > config.max_symbol_to_word_ratio:
        failed.append("symbol_to_word_ratio_too_high")

    if _digit_char_ratio(text) > config.max_digit_char_ratio:
        failed.append("digit_char_ratio_too_high")

    lines = text.split("\n")
    if _bullet_or_ellipsis_line_fraction(lines) > config.max_bullet_or_ellipsis_line_fraction:
        failed.append("bullet_or_ellipsis_line_fraction_too_high")
    if _repeated_line_fraction(lines) > config.max_repeated_line_fraction:
        failed.append("repeated_line_fraction_too_high")

    if _has_boilerplate_phrase(text):
        failed.append("boilerplate_phrase_present")

    if config.require_stopword_hit:
        stopwords = HINDI_STOPWORDS if language == "hindi" else NEPALI_STOPWORDS
        if _stopword_hits(text, stopwords) < config.min_stopword_hits:
            failed.append("insufficient_stopword_evidence")

    return FilterVerdict(passed=len(failed) == 0, failed_rules=failed)


def strip_light_pii(text: str) -> str:
    """Redact emails and 10-digit phone-shaped numbers. Not a comprehensive PII scrubber --
    a best-effort pass appropriate for a course-project corpus, applied after filtering so it
    does not distort the length/digit-ratio statistics the heuristic rules above depend on."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text
