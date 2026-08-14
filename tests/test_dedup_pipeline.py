"""Tests for lma.dedup.pipeline.deduplicate_documents -- the integrated exact+LSH dedup pass."""
from lma.dedup.pipeline import deduplicate_documents

EXACT_DUP_TEXT = "यह एक बिल्कुल समान दस्तावेज़ है जो दो बार दिखाई देता है इस उदाहरण में"

NEWS_A = "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में विस्तार से बताता है और सर्दियों की बात करता है"
NEWS_B = "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में विस्तार से बताता है लेकिन गर्मियों की बात करता है"
NEWS_C = "यह एक समाचार लेख है जो मुंबई के मौसम के बारे में विस्तार से बताता है लेकिन गर्मियों की बात करता है"

UNRELATED = "क्रिकेट मैच में भारत ने आज शानदार प्रदर्शन करते हुए जीत दर्ज की और सभी दर्शक बहुत खुश हुए"


def test_exact_duplicates_are_removed():
    doc_ids = ["d1", "d2", "d3"]
    texts = [EXACT_DUP_TEXT, EXACT_DUP_TEXT, "एक पूरी तरह से अलग और असंबंधित दस्तावेज़ है यह वाला"]
    kept, stats = deduplicate_documents(doc_ids, texts)
    assert stats.exact_duplicates_removed == 1
    assert len(kept) == 2
    assert "d1" in kept  # first-seen representative kept
    assert "d2" not in kept


def test_unrelated_documents_are_all_kept():
    doc_ids = ["a", "b"]
    texts = [NEWS_A, UNRELATED]
    kept, stats = deduplicate_documents(doc_ids, texts, lsh_b=32, lsh_r=4, similarity_threshold=0.5)
    assert kept == {"a", "b"}
    assert stats.near_duplicates_removed == 0


def test_near_duplicates_are_clustered_and_one_representative_kept():
    doc_ids = ["a", "b", "unrelated"]
    texts = [NEWS_A, NEWS_B, UNRELATED]
    # Loose banding (see test_lsh.py for why b=32,r=4 catches ~0.67 similarity reliably).
    kept, stats = deduplicate_documents(
        doc_ids, texts, lsh_b=32, lsh_r=4, similarity_threshold=0.5
    )
    assert "unrelated" in kept
    # Exactly one of {a, b} should survive as the cluster representative.
    assert len(kept & {"a", "b"}) == 1
    assert "a" in kept  # first-seen (lowest original index) is the deterministic representative
    assert stats.near_duplicate_clusters == 1
    assert stats.near_duplicates_removed == 1


def test_transitive_clustering_merges_a_chain_into_one_cluster():
    # Engineered so A-B ~= 0.667 and B-C ~= 0.739 (both above threshold=0.55), but A-C ~= 0.481
    # (below 0.55, verified via exact_jaccard) -- there is NO direct A-C edge. The union-find
    # must still merge all three into one cluster transitively via the shared B connection.
    doc_ids = ["a", "b", "c"]
    texts = [NEWS_A, NEWS_B, NEWS_C]
    kept, stats = deduplicate_documents(
        doc_ids, texts, lsh_b=32, lsh_r=4, similarity_threshold=0.55
    )
    assert stats.kept_count == 1
    assert kept == {"a"}  # first-seen representative of the merged 3-way cluster


def test_deterministic_representative_is_first_seen_by_input_order():
    doc_ids = ["z", "a"]  # doc_id "z" is listed FIRST in the input, despite alphabetical order
    texts = [NEWS_A, NEWS_B]
    kept, _ = deduplicate_documents(doc_ids, texts, lsh_b=32, lsh_r=4, similarity_threshold=0.5)
    assert kept == {"z"}  # kept by input POSITION, not by doc_id sort order


def test_stats_input_count_matches_length():
    doc_ids = ["a", "b", "c"]
    texts = [NEWS_A, UNRELATED, EXACT_DUP_TEXT]
    _, stats = deduplicate_documents(doc_ids, texts)
    assert stats.input_count == 3


def test_empty_input_is_handled():
    kept, stats = deduplicate_documents([], [])
    assert kept == set()
    assert stats.input_count == 0
    assert stats.kept_count == 0
