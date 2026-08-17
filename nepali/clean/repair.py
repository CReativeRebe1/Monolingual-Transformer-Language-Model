"""
Stage 3b -- Rule-Based Correction (Yi Figure 1), for Nepali.

The stage the earlier pipeline skipped entirely, and the reason it matters is ordering: it runs
**before** the filters decide anything. A news article carrying three "विज्ञापन" furniture lines and
a mojibake apostrophe is a perfectly good training document with removable debris on it. Filter
first and it is discarded; repair first and it is kept. On a low-resource corpus that difference is
the whole ballgame (Lecture 3, slide 30).

What is repaired, in order:

1. **Mojibake and HTML entities** -- from `words.MOJIBAKE`, which lists the specific corruptions
   found in this corpus rather than a generic table.
2. **Leaked JSON metadata** -- article templates that emit a metadata object as literal text,
   sometimes before the body and sometimes after the headline. Removing the object leaves the
   real prose behind; rejecting the document throws the prose away with it. (Diagnosed in
   detail on the Hindi corpus, where amarujala.com's premium template does this; the rule
   itself is generic and the same templates appear in the Nepali crawl.)
3. **Boilerplate lines** -- whole lines equal to a known furniture string (`BOILERPLATE_LINES`),
   plus lines *containing* a known boilerplate phrase (`BOILERPLATE_PHRASES`).
4. **Repeated punctuation and inter-line blank runs** -- cosmetic, but they distort the
   symbol-to-word ratio that stage 3a then measures.

**The safety valve:** if repair would remove more than `max_repair_char_drop_fraction` of a
document's characters, the original is returned untouched. Past that point the document is boilerplate
with some prose attached rather than prose with some boilerplate attached, and it should be allowed
to fail the filters honestly so the funnel attributes it to the right stage instead of showing a
suspiciously well-repaired document sailing through.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nepali.clean.config import DEFAULT, CleanConfig  # noqa: E402
from nepali.clean.words import BOILERPLATE_LINES, BOILERPLATE_PHRASES, MOJIBAKE  # noqa: E402

_REPEATED_PUNCT_RE = re.compile(r"([!?.,;:|])\1{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# A quoted key followed by a colon: the signature of JSON rather than prose in braces.
_JSON_KEY_RE = re.compile(r'"[^"{}]+"\s*:')


@dataclass
class RepairResult:
    text: str
    actions: list[str]
    lines_dropped: int


def fix_mojibake(text: str) -> str:
    for broken, correct in MOJIBAKE:
        text = text.replace(broken, correct)
    return text.replace("�", "")


def strip_json_prefix(text: str) -> str:
    """Removes leaked JSON metadata objects, wherever they appear in the document.

    Originally this only handled a JSON object at the very START of the text, on the assumption that
    the amarujala.com template always leaks its metadata first. Inspecting the documents rejected
    for a high symbol-to-word ratio showed otherwise: the blob frequently sits *after* a headline
    line ("Live\\nChamoli Tunnel Landslide Live: ...\\n{"_id":"6a7e...","slug":...}"), so the prefix
    rule missed it, the JSON's quotes, colons, hyphens and underscores dominated the symbol count,
    and the whole document -- headline, article and all -- was rejected for punctuation that was
    never prose in the first place.

    Only brace groups that actually look like JSON are removed: they must contain a quoted key
    followed by a colon. That keeps the rule from eating legitimate prose in braces.

    Brace matching is done by scanning for *balanced* groups rather than with a regex. The regex
    version could not match nested objects -- and these blobs are nested in practice
    ({"_id":"...","meta":{...}}), so 1,285 documents were still being rejected for a leading JSON
    object that the repair step believed it had already removed.
    """
    for start, end in reversed(_brace_spans(text)):
        blob = text[start:end]
        if _JSON_KEY_RE.search(blob):
            text = text[:start] + text[end:]
    return text


def _brace_spans(text: str) -> list[tuple[int, int]]:
    """Spans of every top-level balanced {...} group, in order."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def strip_boilerplate_phrases(text: str) -> tuple[str, int]:
    """Deletes whole lines containing a known boilerplate phrase. Returns (text, n_dropped).

    Complements `drop_boilerplate_lines`, which only matches a line equal to a furniture string.
    These phrases are embedded inline -- a cookie notice or app-install nag inside an otherwise
    real article -- so excising the line rescues documents that would otherwise be rejected
    whole. Measured on the Hindi corpus it rescued 16% of the boilerplate-rejected documents
    with a median 96% of their text intact; the rest failed other rules too, which is the
    correct outcome.
    """
    kept, dropped = [], 0
    for line in text.split("\n"):
        if any(p in line for p in BOILERPLATE_PHRASES):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def drop_boilerplate_lines(text: str) -> tuple[str, int]:
    """Deletes whole lines that are known furniture. Returns (text, n_dropped)."""
    kept, dropped = [], 0
    for line in text.split("\n"):
        probe = line.strip().strip("।.:-—|").strip().casefold()
        if probe and probe in {b.casefold() for b in BOILERPLATE_LINES}:
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def repair(text: str, *, config: CleanConfig | None = None) -> RepairResult:
    """Applies every correction, or none of them if the document is mostly boilerplate."""
    cfg = config or DEFAULT
    actions: list[str] = []
    original = text
    n_chars_before = max(len(text.strip()), 1)

    fixed = fix_mojibake(text)
    if fixed != text:
        actions.append("mojibake")
    text = fixed

    fixed = strip_json_prefix(text)
    if fixed != text:
        actions.append("json_blob")
    text = fixed

    text, dropped = drop_boilerplate_lines(text)
    if dropped:
        actions.append("boilerplate_lines")

    text, dropped_phrases = strip_boilerplate_phrases(text)
    if dropped_phrases:
        actions.append("boilerplate_phrase_lines")
    dropped += dropped_phrases

    if (n_chars_before - len(text.strip())) / n_chars_before > cfg.max_repair_char_drop_fraction:
        # Too much of the document was furniture -- see the module docstring's safety valve.
        return RepairResult(text=original, actions=["skipped_mostly_boilerplate"], lines_dropped=0)

    fixed = _REPEATED_PUNCT_RE.sub(r"\1", text)
    if fixed != text:
        actions.append("repeated_punct")
    text = _MULTI_BLANK_RE.sub("\n\n", fixed).strip()

    return RepairResult(text=text, actions=actions, lines_dropped=dropped)
