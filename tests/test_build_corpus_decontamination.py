"""Tests for scripts/build_corpus.py's decontaminate -- a real gap found by
reviewing (not running) the pipeline: build_corpus.py's own module docstring described the
pipeline as "dedup -> decontaminate -> split", and lma/dedup/decontaminate.py was fully built and
unit-tested in isolation, but nothing in build_corpus.py actually called it. The document-level
98/1/1 split was a plain random shuffle with no check that a near-duplicate of a held-out
document (embedded inside a longer training document, which whole-document dedup would not catch)
could leak from train into val/test, which would inflate Phase-2's perplexity/BPB metrics.

Pulled decontaminate() out as its own pure function (no network dependency, unlike
the rest of build_corpus()) specifically so this could be tested with tiny synthetic documents.
"""
from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_corpus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doc(text):
    return {"text": text}


LEAKED_PASSAGE = " ".join(f"word{i}" for i in range(20))  # 20 words, well over the 13-gram window


def test_train_document_sharing_13grams_with_val_is_dropped():
    module = _load_module()
    train = [
        _doc(f"some unrelated preamble text here that is different. {LEAKED_PASSAGE} and some more unrelated trailing text follows after this point"),
        _doc("a completely clean training document with no overlap whatsoever with anything else"),
    ]
    val = [_doc(LEAKED_PASSAGE)]
    test = []

    clean_train, n_dropped, index = module.decontaminate(train, val, test)

    assert n_dropped == 1
    assert len(clean_train) == 1
    assert "completely clean" in clean_train[0]["text"]


def test_train_document_sharing_13grams_with_test_is_dropped():
    module = _load_module()
    train = [_doc(f"prefix words go here first. {LEAKED_PASSAGE} suffix words go here after")]
    val = []
    test = [_doc(LEAKED_PASSAGE)]

    clean_train, n_dropped, _ = module.decontaminate(train, val, test)

    assert n_dropped == 1
    assert clean_train == []


def test_clean_train_documents_are_never_dropped():
    module = _load_module()
    train = [_doc(f"document number {i} with entirely its own unique unrelated content and nothing shared") for i in range(10)]
    val = [_doc("a held out validation document with its own completely separate wording and phrasing")]
    test = [_doc("a held out test document with yet another completely separate set of words entirely")]

    clean_train, n_dropped, _ = module.decontaminate(train, val, test)

    assert n_dropped == 0
    assert len(clean_train) == len(train)


def test_short_documents_below_ngram_window_are_never_flagged():
    # word_ngrams() returns an empty set for documents shorter than n=13 words -- such documents
    # can never register an n-gram overlap and must never be spuriously dropped.
    module = _load_module()
    train = [_doc("short document")]
    val = [_doc("short document")]  # identical, but both far under 13 words
    test = []

    clean_train, n_dropped, _ = module.decontaminate(train, val, test)

    assert n_dropped == 0
    assert len(clean_train) == 1


def test_empty_val_and_test_drops_nothing():
    module = _load_module()
    train = [_doc(LEAKED_PASSAGE), _doc("something else entirely unrelated and different in every way possible")]

    clean_train, n_dropped, index = module.decontaminate(train, [], [])

    assert n_dropped == 0
    assert len(clean_train) == 2
    assert len(index) == 0  # no reference n-grams indexed at all


def test_returns_the_contamination_index_used():
    module = _load_module()
    val = [_doc(LEAKED_PASSAGE)]
    _, _, index = module.decontaminate([], val, [])
    assert len(index) > 0  # the 13-gram(s) from the val document were indexed
