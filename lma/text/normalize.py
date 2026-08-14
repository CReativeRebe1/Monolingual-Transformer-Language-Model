"""
Unicode normalization for Devanagari-script text (Hindi and Nepali).

Both Model H (Hindi) and Model L (Nepali) corpora pass through this module before any
downstream filtering, deduplication, or tokenizer training. The rules here are intentionally
conservative: correct real encoding inconsistencies (nukta composition, stray joiners,
inconsistent sentence-final punctuation) without altering meaning.

Every individual rule is idempotent, and `normalize()` as a whole is asserted idempotent in
tests/test_normalize.py — running it twice must equal running it once, since corpus documents
may pass through the pipeline more than once (e.g. re-normalized after dedup) and drift would
otherwise silently corrupt text on the second pass.
"""
from __future__ import annotations

import re
import unicodedata

# --- Devanagari digit <-> ASCII digit mapping (U+0966-U+096F <-> '0'-'9') ---
_DEVANAGARI_DIGITS = "०१२३४५६७८९"
_ASCII_DIGITS = "0123456789"
_DEV_TO_ASCII_DIGITS = str.maketrans(_DEVANAGARI_DIGITS, _ASCII_DIGITS)
_ASCII_TO_DEV_DIGITS = str.maketrans(_ASCII_DIGITS, _DEVANAGARI_DIGITS)

# Zero-width joiner / non-joiner: legitimate inside conjuncts, but never legitimate at a
# string boundary or repeated back-to-back — those patterns are near-certainly copy/typing noise.
_ZW_CHARS = "‌‍"  # ZWNJ, ZWJ
_ZW_REPEATED_RE = re.compile(f"[{_ZW_CHARS}]{{2,}}")
_ZW_AT_BOUNDARY_RE = re.compile(rf"^[{_ZW_CHARS}]+|[{_ZW_CHARS}]+$")

# Devanagari danda (sentence/verse terminator). Text typed without a Devanagari input method
# (very common in scraped forum/blog content and OCR output) frequently substitutes the ASCII
# pipe. A pipe adjacent to Devanagari script on at least one side is not legitimate ASCII
# punctuation in this corpus, so we normalize it to the correct danda.
_DEVANAGARI_BLOCK = r"ऀ-ॿ"
_DOUBLE_PIPE_RE = re.compile(r"\|\s*\|")
_SINGLE_PIPE_NEAR_DEV_RE = re.compile(
    rf"(?<=[{_DEVANAGARI_BLOCK}])\s*\|(?!\|)|\|(?!\|)\s*(?=[{_DEVANAGARI_BLOCK}])"
)

# Control characters (Unicode category Cc) other than the whitespace we handle separately.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Run of any Unicode whitespace, and specifically runs of the space character.
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t ​]+(?=\n)")
_MULTI_SPACE_RE = re.compile(r"[ \t ]{2,}")

# Anusvara / chandrabindu are phonemically distinct (U+0902 vs U+0901) and are NOT merged —
# doing so would change meaning (e.g. हँस "to laugh" vs हंस "swan"). We only assert, in tests,
# that we never touch them, so a future edit doesn't accidentally introduce a merge.


def normalize_unicode_form(text: str) -> str:
    """Apply canonical composition (NFC).

    NFC already canonically composes the standard Devanagari nukta letters (U+0958-095F have
    canonical decompositions to base-consonant + U+093C), so a decomposed nukta sequence typed
    by an input method that emits the two-codepoint form becomes identical to text that already
    used the single-codepoint form. Without this step, otherwise-identical documents would be
    treated as different by exact-dedup hashing and would tokenize inconsistently.
    """
    return unicodedata.normalize("NFC", text)


def strip_control_characters(text: str) -> str:
    """Remove non-printable control characters, keeping \n and \t (handled elsewhere)."""
    return _CONTROL_RE.sub("", text)


def normalize_zero_width_joiners(text: str) -> str:
    """Collapse repeated ZWJ/ZWNJ runs to a single occurrence and drop them at line boundaries.

    ZWJ/ZWNJ are meaningful *between* consonants inside a conjunct (e.g. controlling whether a
    half-form ligates). They are never meaningful repeated, or at the start/end of a line, so
    those occurrences are stripped rather than collapsed-to-one.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = _ZW_REPEATED_RE.sub(lambda m: m.group(0)[0], line)
        line = _ZW_AT_BOUNDARY_RE.sub("", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def normalize_danda(text: str) -> str:
    """Normalize ASCII-pipe stand-ins for the Devanagari danda / double danda.

    '||' anywhere -> '॥' (double danda). A single '|' directly adjacent to Devanagari script
    -> '।' (danda). A lone '|' with no Devanagari neighbour (e.g. inside an embedded table or
    code fragment) is left untouched, since it is plausibly not Devanagari punctuation at all.
    """
    text = _DOUBLE_PIPE_RE.sub("॥", text)
    text = _SINGLE_PIPE_NEAR_DEV_RE.sub("।", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse horizontal whitespace runs, strip trailing blanks, cap blank-line runs at one.

    Deliberately paragraph-preserving: a single blank line (two consecutive '\n') is kept as a
    paragraph break, since a downstream sentence/paragraph segmenter for generation examples
    relies on that structure. Runs of 3+ newlines collapse to exactly one blank line.
    """
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def convert_digits(text: str, target: str = "ascii") -> str:
    """Convert all decimal digits to a single script, one direction only.

    Devanagari text mixes Devanagari numerals (०-९) and ASCII numerals (0-9) depending on
    source (news sites lean ASCII; literary/government sources lean Devanagari). This is left
    OFF by default in `normalize()` — the corpus statistics report both digit-script counts
    before any decision is made, and the choice actually applied is logged in the dataset
    stats/config so the tradeoff (ASCII digits tokenize more efficiently under a shared
    BPE/Unigram vocab; Devanagari digits keep the script visually and orthographically pure)
    is explicit and reproducible rather than silently baked in.

    Args:
        text: input string.
        target: "ascii" or "devanagari".
    """
    if target == "ascii":
        return text.translate(_DEV_TO_ASCII_DIGITS)
    if target == "devanagari":
        return text.translate(_ASCII_TO_DEV_DIGITS)
    raise ValueError(f"target must be 'ascii' or 'devanagari', got {target!r}")


def normalize(text: str, *, digit_target: str | None = None) -> str:
    """Full normalization pipeline, applied in a fixed, idempotent order.

    Order matters: Unicode form composition must run first (so every later regex operates on a
    single canonical representation), control-character stripping and ZWJ/ZWNJ cleanup happen
    before whitespace collapsing (so a stripped character doesn't leave whitespace that then
    needs a second collapse pass), and danda normalization runs before the final whitespace pass
    so any spacing it introduces around punctuation gets cleaned up too.

    Args:
        text: raw document text.
        digit_target: None (leave digits as-is, the default), "ascii", or "devanagari".

    Returns:
        Normalized text. Calling normalize() on an already-normalized string is a no-op.
    """
    text = normalize_unicode_form(text)
    text = strip_control_characters(text)
    text = normalize_zero_width_joiners(text)
    text = normalize_danda(text)
    text = normalize_whitespace(text)
    if digit_target is not None:
        text = convert_digits(text, digit_target)
    return text
