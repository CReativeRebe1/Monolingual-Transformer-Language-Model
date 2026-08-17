"""
N-gram decontamination: remove training documents that leak held-out evaluation text.

Two separate applications in this project, both using the same mechanism: (1) within the
pretraining corpus, decontaminate train against val/test so intrinsic LM metrics (perplexity,
BPB) are not inflated by memorized held-out text; (2) decontaminate the *pretraining* corpus
against the Phase-3 synthetic reasoning test set, so a model cannot have seen reasoning-test
sentences verbatim during pretraining before ever being finetuned on the reasoning task.

13-word n-gram overlap (rather than exact document match, which exact.py already handles, or
whole-document near-duplicate match, which lma/dedup/minhash.py handles) is the standard choice
for contamination detection: it catches a held-out passage embedded inside a much longer
training document, which whole-document similarity would miss entirely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"\S+")


def word_ngrams(text: str, n: int = 13) -> set[str]:
    words = _WORD_RE.findall(text)
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclass
class ContaminationIndex:
    """Indexes n-grams from one or more reference sets (val/test/reasoning-test), then flags
    any candidate training document that shares at least one n-gram with a reference set."""

    n: int = 13
    _reference_ngrams: set[str] = field(default_factory=set)

    def add_reference_text(self, text: str) -> None:
        self._reference_ngrams.update(word_ngrams(text, n=self.n))

    def add_reference_texts(self, texts: list[str]) -> None:
        for t in texts:
            self.add_reference_text(t)

    def is_contaminated(self, candidate_text: str) -> tuple[bool, set[str]]:
        """Returns (is_contaminated, the overlapping n-grams found) for a training-side document."""
        candidate_ngrams = word_ngrams(candidate_text, n=self.n)
        overlap = candidate_ngrams & self._reference_ngrams
        return (len(overlap) > 0, overlap)

    def __len__(self) -> int:
        return len(self._reference_ngrams)


def filter_contaminated(
    candidate_texts: list[str], index: ContaminationIndex
) -> tuple[list[str], list[int]]:
    """Returns (clean_texts, dropped_indices) -- the indices into candidate_texts that were removed."""
    clean = []
    dropped = []
    for i, text in enumerate(candidate_texts):
        contaminated, _ = index.is_contaminated(text)
        if contaminated:
            dropped.append(i)
        else:
            clean.append(text)
    return clean, dropped


def decontaminate(
    train_split: list[dict], val_split: list[dict], test_split: list[dict], n: int = 13
) -> tuple[list[dict], int, ContaminationIndex]:
    """Drops any train document sharing an n-word n-gram with val or test.

    Without this, a near-duplicate of a held-out document embedded inside a longer training document
    -- which whole-document dedup cannot catch, since MinHash measures whole-document similarity, not
    substring containment -- leaks into training and inflates Phase-2's perplexity and BPB on the
    held-out splits.

    Lives here rather than in a per-language module because it is pure n-gram set arithmetic with no
    language content, and both languages' run.py call it. It works on record dicts (not bare strings)
    so callers keep their metadata; the index is returned so a caller can report its size.

    Returns (clean_train_split, n_dropped, the_index_used).
    """
    index = ContaminationIndex(n=n)
    index.add_reference_texts([r["text"] for r in val_split])
    index.add_reference_texts([r["text"] for r in test_split])
    _, contaminated = filter_contaminated([r["text"] for r in train_split], index)
    bad = set(contaminated)
    clean = [r for i, r in enumerate(train_split) if i not in bad]
    return clean, len(train_split) - len(clean), index
