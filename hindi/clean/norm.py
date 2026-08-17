"""
Hindi text normalisation -- Yi's stage-1 "cleaning" applied to Devanagari.

Builds on lma.text.normalize (Unicode form, control chars, zero-width joiners, danda, whitespace,
digits), all of which is script-level and language-neutral, and adds the Hindi-specific folds on
top. Keeping the shared part shared and the language part here is the whole point of the split:
`norm()` below is what each language is free to disagree about.

Hindi-specific decisions taken here, with reasons:

* **Chandrabindu (U+0901) is folded to anusvara (U+0902).** Standard Hindi orthography uses the
  two almost interchangeably for nasalisation and modern typesetting overwhelmingly prefers
  anusvara; folding cuts vocabulary fragmentation for the tokenizer at no semantic cost.
  **Nepali does NOT fold this** -- see nepali/clean/norm.py for why.
* **Nukta is preserved.** क़/ख़/ग़/ज़/फ़ distinguish Perso-Arabic loanwords that are genuinely
  frequent in Hindi. NFC composition of decomposed nukta sequences is still applied, since that is
  an encoding difference rather than an orthographic one.
* **Repeated matras are collapsed** (कीी -> की) and **word-initial matras are dropped**, both of
  which are encoding corruption rather than spelling -- a dependent vowel sign cannot legally
  begin a word.
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

# Dependent vowel signs and other marks that cannot legally begin a word.
_MATRAS = "ा-ौॎॏॕ-ॗॢॣ"
_REPEATED_MATRA_RE = re.compile(f"([{_MATRAS}])\\1+")
_WORD_INITIAL_MATRA_RE = re.compile(f"(?<![ऀ-ॿ])[{_MATRAS}{VIRAMA}]+")
# A virama with nothing after it: a dangling half-form left behind by broken HTML extraction.
_DANGLING_VIRAMA_RE = re.compile(f"{VIRAMA}(?=[ \t\r\n]|$)")


def fold_nasal(text: str) -> str:
    """Chandrabindu -> anusvara. Hindi-only; see module docstring."""
    return text.replace(CHANDRABINDU, ANUSVARA)


def fix_matras(text: str) -> str:
    """Repairs the encoding corruptions that actually appear in crawled Devanagari."""
    text = _REPEATED_MATRA_RE.sub(r"\1", text)
    text = _WORD_INITIAL_MATRA_RE.sub("", text)
    text = _DANGLING_VIRAMA_RE.sub("", text)
    return text


def norm(text: str, *, digit_target: str | None = None) -> str:
    """The full Hindi normalisation pass. Idempotent: norm(norm(x)) == norm(x)."""
    text = _shared_normalize(text, digit_target=digit_target)
    text = fold_nasal(text)
    text = fix_matras(text)
    # Re-run the shared whitespace pass: dropping matras can leave doubled spaces behind.
    return _shared_normalize(text, digit_target=digit_target)
