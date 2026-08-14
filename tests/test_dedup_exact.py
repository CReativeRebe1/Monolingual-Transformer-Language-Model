"""Tests for lma.dedup.exact and lma.dedup.decontaminate."""
from lma.dedup.decontaminate import ContaminationIndex, filter_contaminated, word_ngrams
from lma.dedup.exact import ExactDeduplicator, content_hash


def test_content_hash_is_deterministic():
    assert content_hash("नमस्ते") == content_hash("नमस्ते")


def test_content_hash_differs_for_different_text():
    assert content_hash("नमस्ते") != content_hash("नमस्कार")


def test_exact_deduplicator_first_occurrence_is_not_duplicate():
    dedup = ExactDeduplicator()
    assert dedup.is_duplicate("दस्तावेज़ एक") is False


def test_exact_deduplicator_second_occurrence_is_duplicate():
    dedup = ExactDeduplicator()
    dedup.is_duplicate("दस्तावेज़ एक")
    assert dedup.is_duplicate("दस्तावेज़ एक") is True


def test_exact_deduplicator_distinct_texts_both_pass():
    dedup = ExactDeduplicator()
    assert dedup.is_duplicate("दस्तावेज़ एक") is False
    assert dedup.is_duplicate("दस्तावेज़ दो") is False
    assert len(dedup) == 2


# --- Decontamination ---

def test_word_ngrams_basic():
    grams = word_ngrams("a b c d e", n=3)
    assert grams == {"a b c", "b c d", "c d e"}


def test_word_ngrams_below_n_words_returns_empty():
    assert word_ngrams("a b", n=3) == set()


def test_contamination_index_flags_exact_ngram_overlap():
    index = ContaminationIndex(n=5)
    index.add_reference_text("भारत की राजधानी नई दिल्ली है और यह बहुत बड़ा शहर है")
    contaminated_doc = (
        "किसी लेख में यह पंक्ति छिपी है कि भारत की राजधानी नई दिल्ली है और यह जानकारी सही है"
    )
    is_contaminated, overlap = index.is_contaminated(contaminated_doc)
    assert is_contaminated
    assert len(overlap) > 0


def test_contamination_index_does_not_flag_unrelated_text():
    index = ContaminationIndex(n=13)
    index.add_reference_text(
        "भारत की राजधानी नई दिल्ली है और यह बहुत बड़ा और ऐतिहासिक शहर है जहाँ करोड़ों लोग रहते हैं"
    )
    unrelated_doc = "क्रिकेट मैच में आज भारतीय टीम ने शानदार जीत दर्ज की और सभी खुश हुए"
    is_contaminated, overlap = index.is_contaminated(unrelated_doc)
    assert not is_contaminated
    assert overlap == set()


def test_filter_contaminated_drops_only_matching_documents():
    index = ContaminationIndex(n=5)
    index.add_reference_text("यह एक विशिष्ट परीक्षण वाक्य है जो केवल परीक्षण सेट में मौजूद है")
    candidates = [
        "यह एक विशिष्ट परीक्षण वाक्य है जो केवल परीक्षण सेट में लीक हो गया दुर्भाग्य से",
        "यह पूरी तरह से अलग और असंबंधित एक साफ प्रशिक्षण दस्तावेज़ है",
    ]
    clean, dropped = filter_contaminated(candidates, index)
    assert dropped == [0]
    assert len(clean) == 1
    assert clean[0] == candidates[1]


def test_add_reference_texts_accumulates_across_multiple_documents():
    index = ContaminationIndex(n=4)
    index.add_reference_texts(["one two three four five", "alpha beta gamma delta epsilon"])
    assert len(index) > 0
    is_contam, _ = index.is_contaminated("one two three four five six seven")
    assert is_contam
