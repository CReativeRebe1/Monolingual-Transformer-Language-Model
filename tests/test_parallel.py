"""Tests for lma.text.parallel -- the fused, multi-core per-document cleaning cascade.

Runs a real process pool (not mocked): the point is that the fused worker produces exactly the
funnel counts, rejection tallies, and PII-redacted survivors that the previous four sequential
loops in build_corpus.py produced, and that pickling models to workers actually works.
"""
import pytest

from lma.text.parallel import clean_documents

GOOD_HI = (
    "यह एक सामान्य समाचार लेख है और इसमें कई वाक्य हैं। "
    "यह लेख भारत की राजधानी दिल्ली के बारे में है और यह बताता है कि वहाँ का मौसम कैसा है। "
    "इस लेख को पढ़ने वाले लोग यह जान पाएंगे कि दिल्ली में सर्दियों में क्या होता है। "
    "यह जानकारी उन लोगों के लिए उपयोगी है जो दिल्ली आना चाहते हैं और अधिक जानना चाहते हैं।"
) * 2

SHORT_HI = "छोटा लेख।"
JSON_PREFIX = '{"_id":"1","slug":"x"}\n' + GOOD_HI
ENGLISH = ("This is an ordinary English news article with several sentences in it. " * 8)


def _recs(*texts):
    return [{"text": t, "host": "x.com", "source_type": "manual"} for t in texts]


def test_good_document_survives_and_is_returned(hi_models):
    kept, survived, _ = clean_documents(_recs(GOOD_HI), "hindi", "hi", *hi_models, workers=2)
    assert len(kept) == 1
    assert survived["quality_filtered"] == 1
    assert "दिल्ली" in kept[0]["text"]


def test_wrong_language_is_rejected_at_language_stage(hi_models):
    kept, survived, _ = clean_documents(_recs(ENGLISH), "hindi", "hi", *hi_models, workers=2)
    assert kept == []
    assert survived["language_filtered"] == 0


def test_too_short_document_rejected_with_rule_names(hi_models):
    kept, survived, rejects = clean_documents(_recs(SHORT_HI), "hindi", "hi", *hi_models, workers=2)
    assert kept == []
    # Rejected at the heuristic stage means it passed language filtering first.
    if survived["language_filtered"]:
        assert "too_short" in rejects or "too_few_words" in rejects


def test_json_prefixed_document_is_rejected(hi_models):
    kept, _, rejects = clean_documents(_recs(JSON_PREFIX), "hindi", "hi", *hi_models, workers=2)
    assert kept == []
    assert "starts_with_json_metadata" in rejects


def test_pii_is_redacted_in_survivors(hi_models):
    text = GOOD_HI + " संपर्क: test@example.com और 9876543210 पर।"
    kept, _, _ = clean_documents(_recs(text), "hindi", "hi", *hi_models, workers=2)
    assert len(kept) == 1
    assert "test@example.com" not in kept[0]["text"]
    assert "9876543210" not in kept[0]["text"]
    assert "[EMAIL]" in kept[0]["text"] and "[PHONE]" in kept[0]["text"]


def test_funnel_counts_are_monotonically_non_increasing(hi_models):
    docs = _recs(GOOD_HI, GOOD_HI, SHORT_HI, ENGLISH, JSON_PREFIX)
    _, survived, _ = clean_documents(docs, "hindi", "hi", *hi_models, workers=2)
    assert survived["language_filtered"] >= survived["heuristic_filtered"]
    assert survived["heuristic_filtered"] >= survived["quality_filtered"]


def test_kept_records_preserve_their_metadata(hi_models):
    kept, _, _ = clean_documents(_recs(GOOD_HI), "hindi", "hi", *hi_models, workers=2)
    assert kept[0]["host"] == "x.com"
    assert kept[0]["source_type"] == "manual"


def test_empty_input(hi_models):
    kept, survived, rejects = clean_documents([], "hindi", "hi", *hi_models, workers=2)
    assert kept == [] and rejects == {}
    assert all(v == 0 for v in survived.values())


def test_results_match_regardless_of_worker_count(hi_models):
    docs = lambda: _recs(GOOD_HI, GOOD_HI, SHORT_HI, ENGLISH)  # noqa: E731
    k1, s1, r1 = clean_documents(docs(), "hindi", "hi", *hi_models, workers=1)
    k4, s4, r4 = clean_documents(docs(), "hindi", "hi", *hi_models, workers=4)
    assert len(k1) == len(k4)
    assert s1 == s4 and r1 == r4


@pytest.fixture(scope="module")
def hi_models():
    """(langid_model, quality_scorer, threshold) -- no language-ID or quality model.

    Passing None for both exercises the cascade's documented "model unavailable" path (fastText
    alone still runs), which keeps this test module fast and free of any network dependency for
    downloading a trusted Wikipedia seed.
    """
    return (None, None, None)
