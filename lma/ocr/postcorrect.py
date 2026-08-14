"""
OCR post-correction: cheap, high-precision fixes for Tesseract's most common Devanagari errors.

Not a general OCR-error-correction model (that would be its own significant project) -- a small
set of targeted regex fixes for failure modes that are extremely common and safe to fix
mechanically: broken matra (vowel sign) spacing introduced by imperfect line segmentation, and
stray single-character "words" left over from noise/artifacts that a line-level confidence
filter (lma/ocr/pdf_ocr.py's min_confidence) does not catch at the word level. Reused text-level
cleanup (Unicode normalization, whitespace collapsing) is delegated to lma.text.normalize rather
than reimplemented here.
"""
from __future__ import annotations

import re

from lma.text.normalize import normalize

# Devanagari dependent vowel signs (matras) and other combining marks: U+093A-093C, 093E-094D,
# 0951-0957, 0962-0963. A space accidentally inserted between a consonant and its following matra
# (a common OCR line-segmentation artifact) changes या -> य ा, which normalize()'s NFC pass alone
# cannot fix since the two pieces are separated by a space, not just decomposed. This regex
# removes exactly that inserted space, when the character after the space is a combining mark.
_MATRA_RE = r"[ऺ-़ा-्॑-ॗॢ-ॣ]"
_SPACE_BEFORE_MATRA_RE = re.compile(rf"(\S) +({_MATRA_RE})")

# A single Devanagari consonant/vowel character surrounded by spaces, with no combining marks
# attached, is very often OCR noise (a broken conjunct split into pieces, or a scan artifact)
# rather than a genuine one-character word (Devanagari has very few genuine single-character
# words; ओ "oh", ए "hey" being the rare exceptions this pattern would also strip -- an accepted
# false-positive rate given how much noise it removes on real scanned book pages).
_ISOLATED_SINGLE_CHAR_RE = re.compile(r"(?<=\s)[ऀ-ॿ](?=\s)")


def fix_matra_spacing(text: str) -> str:
    """Removes a stray space between a consonant and an immediately following combining mark."""
    prev = None
    while prev != text:
        prev = text
        text = _SPACE_BEFORE_MATRA_RE.sub(r"\1\2", text)
    return text


def strip_isolated_noise_characters(text: str) -> str:
    """Drops single free-standing Devanagari characters (see module docstring for rationale)."""
    return _ISOLATED_SINGLE_CHAR_RE.sub("", text)


def postcorrect_ocr_text(text: str) -> str:
    """Full OCR cleanup pipeline: matra-spacing fix -> noise-character strip -> normalize.

    Order matters: matra spacing must be fixed before whitespace collapsing (lma.text.normalize)
    would otherwise treat the stray space as ordinary inter-word whitespace and leave it in place
    since it does not, by itself, look like a multi-space run.
    """
    text = fix_matra_spacing(text)
    text = strip_isolated_noise_characters(text)
    text = normalize(text)
    return text
