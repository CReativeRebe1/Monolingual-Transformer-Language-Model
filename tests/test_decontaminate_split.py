"""Tests for lma.dedup.decontaminate.decontaminate -- the split-level leak check.

Originally a real gap: the pipeline's docstring described itself as "dedup -> decontaminate ->
split" and lma/dedup/decontaminate.py was fully built and unit-tested, but nothing actually called
it -- the 98/1/1 split was a plain shuffle, so a near-duplicate of a held-out document embedded
inside a longer training document (which whole-document dedup cannot catch) leaked from train into
val/test and would inflate Phase-2's perplexity/BPB.

The function now lives in lma/dedup/ rather than in a script, because both hindi/clean/run.py and
nepali/clean/run.py call it and it contains no language-specific content -- it is n-gram set
arithmetic. These tests use tiny synthetic documents and need no network.
"""
from lma.dedup.decontaminate import decontaminate


def _doc(text):
    return {"text": text}


LEAKED_PASSAGE = " ".join(f"word{i}" for i in range(20))  # 20 words, well over the 13-gram window


def test_train_document_sharing_13grams_with_val_is_dropped():
    train = [
        _doc(f"some unrelated preamble text here that is different. {LEAKED_PASSAGE} and some more unrelated trailing text follows after this point"),
        _doc("a completely clean training document with no overlap whatsoever with anything else"),
    ]
    val = [_doc(LEAKED_PASSAGE)]
    test = []

    clean_train, n_dropped, index = decontaminate(train, val, test)

    assert n_dropped == 1
    assert len(clean_train) == 1
    assert "completely clean" in clean_train[0]["text"]


def test_train_document_sharing_13grams_with_test_is_dropped():
    train = [_doc(f"prefix words go here first. {LEAKED_PASSAGE} suffix words go here after")]
    val = []
    test = [_doc(LEAKED_PASSAGE)]

    clean_train, n_dropped, _ = decontaminate(train, val, test)

    assert n_dropped == 1
    assert clean_train == []


def test_clean_train_documents_are_never_dropped():
    train = [_doc(f"document number {i} with entirely its own unique unrelated content and nothing shared") for i in range(10)]
    val = [_doc("a held out validation document with its own completely separate wording and phrasing")]
    test = [_doc("a held out test document with yet another completely separate set of words entirely")]

    clean_train, n_dropped, _ = decontaminate(train, val, test)

    assert n_dropped == 0
    assert len(clean_train) == len(train)


def test_short_documents_below_ngram_window_are_never_flagged():
    # word_ngrams() returns an empty set for documents shorter than n=13 words -- such documents
    # can never register an n-gram overlap and must never be spuriously dropped.
    train = [_doc("short document")]
    val = [_doc("short document")]  # identical, but both far under 13 words
    test = []

    clean_train, n_dropped, _ = decontaminate(train, val, test)

    assert n_dropped == 0
    assert len(clean_train) == 1


def test_empty_val_and_test_drops_nothing():
    train = [_doc(LEAKED_PASSAGE), _doc("something else entirely unrelated and different in every way possible")]

    clean_train, n_dropped, index = decontaminate(train, [], [])

    assert n_dropped == 0
    assert len(clean_train) == 2
    assert len(index) == 0  # no reference n-grams indexed at all


def test_returns_the_contamination_index_used():
    val = [_doc(LEAKED_PASSAGE)]
    _, _, index = decontaminate([], val, [])
    assert len(index) > 0  # the 13-gram(s) from the val document were indexed
