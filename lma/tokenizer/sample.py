"""
Builds the sentence-level sample that SentencePiece is actually trained on.

Two problems this solves, both of which silently corrupted the earlier sweep.

**SentencePiece drops long lines.** `SentencePieceTrainer` defaults `max_sentence_length` to 4192
BYTES, and anything longer is skipped without warning. Devanagari costs 3 bytes per character in
UTF-8, so 4192 bytes is only ~1,400 characters -- roughly 230 Hindi words. Our splits are written one
DOCUMENT per line and average ~1.6 KB, with a long tail well past the limit, so a large share of the
corpus never reached the trainer at all. Splitting on danda (।) turns the input into real sentences,
which is the unit SentencePiece expects and which sits comfortably under any length cap.

**The corpus is far too large to train on ten times.** Hindi's train split is 5.1 GB and Nepali's is
7.5 GB, and the sweep trains 10 models per language. Sampling once into a reusable file means the
multi-gigabyte split is read once rather than ten times, and keeps each training run's input at a
size that fits in memory on a 7.4 GB machine. Sampling is by stride rather than from the head, so the
sample spans the whole corpus -- the splits are ordered manual-then-downloaded, and taking a prefix
would train the tokenizer on manual text only.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("lma.tokenizer.sample")

# Danda and double danda end Devanagari sentences; ASCII .!? appear in transliterated and
# numeric contexts. Newlines are already gone (splits are one line per document).
_SENTENCE_END_RE = re.compile(r"(?<=[।॥.!?])\s+")
MAX_SENTENCE_CHARS = 1_000          # hard cap; a "sentence" longer than this is a parse failure
MIN_SENTENCE_CHARS = 10


@dataclass
class SampleInfo:
    path: Path
    n_documents_read: int
    n_sentences: int
    n_chars: int


def split_sentences(text: str) -> list[str]:
    """Splits one document into sentences, hard-wrapping anything pathologically long.

    The wrap matters: a document with no danda at all (a table, a list, an extraction failure)
    would otherwise come through as one enormous line and be dropped by the trainer -- exactly the
    silent loss this module exists to prevent.
    """
    out: list[str] = []
    for part in _SENTENCE_END_RE.split(text.strip()):
        part = part.strip()
        if len(part) < MIN_SENTENCE_CHARS:
            continue
        while len(part) > MAX_SENTENCE_CHARS:
            cut = part.rfind(" ", 0, MAX_SENTENCE_CHARS)
            if cut <= 0:
                cut = MAX_SENTENCE_CHARS
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        if len(part) >= MIN_SENTENCE_CHARS:
            out.append(part)
    return out


def build_training_sample(
    split_path: Path,
    out_path: Path,
    max_sentences: int = 5_000_000,
    stride: int = 1,
) -> SampleInfo:
    """Streams `split_path` (one document per line) into `out_path` (one sentence per line).

    `stride` takes every Nth document so the sample spans the corpus rather than its first N
    documents. Returns counts for the report -- how much text the tokenizer actually saw is a
    number worth stating, since the honest answer for the previous sweep was "unknown".
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_sentences = n_chars = 0
    with open(split_path, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if stride > 1 and i % stride:
                continue
            line = line.strip()
            if not line:
                continue
            n_docs += 1
            for sentence in split_sentences(line):
                dst.write(sentence + "\n")
                n_sentences += 1
                n_chars += len(sentence)
                if n_sentences >= max_sentences:
                    logger.info("sample cap reached: %d sentences from %d documents",
                                n_sentences, n_docs)
                    return SampleInfo(out_path, n_docs, n_sentences, n_chars)
    logger.info("sample: %d sentences (%d chars) from %d documents", n_sentences, n_chars, n_docs)
    return SampleInfo(out_path, n_docs, n_sentences, n_chars)


def count_lines(path: Path) -> int:
    """Line count without loading the file -- the splits are multi-gigabyte."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n
