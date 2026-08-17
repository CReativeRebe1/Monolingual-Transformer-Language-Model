"""
Nepali text normalisation -- Yi's stage-1 "cleaning" applied to Devanagari.

Shares the script-level work with Hindi (lma.text.normalize: Unicode form, control characters,
zero-width joiners, danda, whitespace, digits) and diverges where the orthographies diverge.

**The important divergence: chandrabindu (U+0901) is NOT folded to anusvara.** hindi/clean/norm.py
folds it, because modern Hindi typesetting treats the two as interchangeable for nasalisation. In
Nepali the chandrabindu is part of standard spelling and carries the nasal vowel in extremely common
words -- गाउँ (village), ठाउँ (place), हुँदा (when being), भाँडा (vessel), बाँकी (remaining). Folding it
would rewrite गाउँ as गाउं, which is simply misspelt Nepali: it would fragment the tokenizer's
vocabulary against every correctly-spelled occurrence in the rest of the corpus and put wrong
spellings into a model meant to generate the language.

This is the concrete case Lecture 3 (slide 30) is describing: a normalisation justified on the
higher-resource language quietly damages the lower-resource one, and the only way to notice is to
decide it per language.

Everything else matches the Hindi pass: repeated matras are collapsed (गाउँँ -> गाउँ) and
word-initial matras dropped, both being encoding corruption rather than spelling -- a dependent
vowel sign cannot legally begin a word in either language.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lma.text.normalize import normalize as _shared_normalize  # noqa: E402

CHANDRABINDU = "ँ"
ANUSVARA = "ं"
VIRAMA = "्"

# Dependent vowel signs and other marks that cannot legally begin a word. Note the chandrabindu is
# absent from this class as well: it is a legitimate mark mid-word and must survive both passes.
_MATRAS = "ा-ौॎॏॕ-ॗॢॣ"
_REPEATED_MATRA_RE = re.compile(f"([{_MATRAS}{CHANDRABINDU}])\\1+")
_WORD_INITIAL_MATRA_RE = re.compile(f"(?<![ऀ-ॿ])[{_MATRAS}{VIRAMA}]+")
# A virama with nothing after it: a dangling half-form left behind by broken HTML extraction.
_DANGLING_VIRAMA_RE = re.compile(f"{VIRAMA}(?=[ \t\r\n]|$)")


def fix_matras(text: str) -> str:
    """Repairs the encoding corruptions that actually appear in crawled Devanagari."""
    text = _REPEATED_MATRA_RE.sub(r"\1", text)
    text = _WORD_INITIAL_MATRA_RE.sub("", text)
    text = _DANGLING_VIRAMA_RE.sub("", text)
    return text


def norm(text: str, *, digit_target: str | None = None) -> str:
    """The full Nepali normalisation pass. Idempotent: norm(norm(x)) == norm(x).

    Deliberately has no `fold_nasal` step -- see the module docstring.
    """
    text = _shared_normalize(text, digit_target=digit_target)
    text = fix_matras(text)
    # Re-run the shared whitespace pass: dropping matras can leave doubled spaces behind.
    return _shared_normalize(text, digit_target=digit_target)
