"""
Stage 2 -- language filtering for Nepali.

Three pieces of evidence, cheapest first:

1. **Devanagari character ratio.** A pure script test. Catches the common crawl failure -- an
   English page, a page of transliterated Nepali, or a page that is mostly navigation markup --
   without loading any model at all.
2. **fastText lid.176.** The real classifier -- but a hesitant one here. It labels 96% of real
   Nepali correctly and still reports a median confidence of only 0.69-0.76, because Nepali is
   low-resource in that model. `lid_min_confidence` is therefore 0.50, not Hindi's 0.65; see
   config.py for the measured rejection rates behind that number.
3. **Nepali-vs-Hindi marker vote.** fastText genuinely confuses the two -- same script, shared
   Sanskrit vocabulary, and 98 of 2,500 Nepali Wikipedia documents were labelled 'hi' outright.
   This project's whole point is two *independent* corpora, so a Hindi page leaking into the
   Nepali corpus is worse than an ordinary quality failure. When the marker counts clearly
   favour Hindi, the document is rejected regardless of what fastText said.

The fastText model is optional: if it is not available, `check()` degrades to the script ratio plus
the marker vote rather than failing. A missing model must never silently accept everything.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nepali.clean.config import DEFAULT, FASTTEXT_CODE, CleanConfig  # noqa: E402
from nepali.clean.words import MARKERS, OTHER_LANG_MARKERS  # noqa: E402

DEVANAGARI_START, DEVANAGARI_END = "ऀ", "ॿ"


def dev_ratio(text: str) -> float:
    """Devanagari share of the letters in `text` (digits, punctuation and space ignored).

    Measured over letters only, not all characters: a legitimate Nepali article carrying a long
    numeric table or heavy punctuation would otherwise look like a script failure.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if DEVANAGARI_START <= c <= DEVANAGARI_END) / len(letters)


def marker_vote(text: str) -> tuple[int, int]:
    """(nepali_marker_hits, hindi_marker_hits) as distinct word types present."""
    types = set(text.split())
    return len(types & MARKERS), len(types & OTHER_LANG_MARKERS)


def check(
    text: str,
    *,
    config: CleanConfig | None = None,
    lid=None,
) -> tuple[bool, str | None]:
    """Returns (is_nepali, reason_if_not). `lid` is an optional object with .predict(text).

    Reasons are returned rather than raised so run.py can attribute every rejection to a specific
    rule in the funnel -- the Phase-1 report needs per-rule counts, not a single stage total.
    """
    cfg = config or DEFAULT

    if dev_ratio(text) < cfg.lid_min_devanagari_ratio:
        return False, "low_devanagari_ratio"

    ne, hi = marker_vote(text)
    # A clear Hindi lean, not a narrow one: the two languages share function words, so this only
    # fires when Hindi markers outnumber Nepali ones by a real margin AND Nepali evidence is thin.
    if hi >= 3 and hi > ne * 2:
        return False, "hindi_markers_dominant"

    if lid is not None:
        label, conf = lid.predict(text)
        if label != FASTTEXT_CODE:
            return False, f"lid_said_{label}"
        if conf < cfg.lid_min_confidence:
            return False, "lid_low_confidence"

    return True, None
