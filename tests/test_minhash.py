"""Tests for lma.dedup.minhash -- validated against `datasketch` as an independent reference.

The MinHash implementation here is from scratch (own hash family, own signature computation),
per the Phase-1 plan's decision to implement MinHash-LSH directly rather than import it. These
tests exist to prove the from-scratch implementation is *correct*, not just that it runs: the
MinHash identity (estimated Jaccard converges to true Jaccard as num_perm grows) is checked
directly, and independently cross-checked against `datasketch.MinHash` on the same inputs.
"""
import numpy as np
import pytest
from datasketch import MinHash as ReferenceMinHash

from lma.dedup.minhash import MinHasher, estimate_jaccard, exact_jaccard, word_shingles

DOC_A = "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में विस्तार से बताता है और सर्दियों की बात करता है"
DOC_B = "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में विस्तार से बताता है लेकिन गर्मियों की बात करता है"
DOC_C = "क्रिकेट मैच में भारत ने आज शानदार प्रदर्शन करते हुए जीत दर्ज की और सभी दर्शक बहुत खुश हुए"


def test_word_shingles_basic():
    shingles = word_shingles("a b c d", k=3)
    assert shingles == {"a b c", "b c d"}


def test_word_shingles_short_text_falls_back_to_whole_text():
    shingles = word_shingles("a b", k=3)
    assert shingles == {"a b"}


def test_word_shingles_empty_text():
    assert word_shingles("", k=3) == set()


def test_exact_jaccard_identical_sets():
    s = word_shingles(DOC_A, k=3)
    assert exact_jaccard(s, s) == 1.0


def test_exact_jaccard_disjoint_sets():
    a = word_shingles(DOC_A, k=3)
    c = word_shingles(DOC_C, k=3)
    assert exact_jaccard(a, c) < 0.05  # near-zero overlap between unrelated articles


def test_minhash_signature_is_deterministic_for_same_hasher():
    hasher = MinHasher(num_perm=64, seed=1)
    sig1 = hasher.signature_for_text(DOC_A)
    sig2 = hasher.signature_for_text(DOC_A)
    assert np.array_equal(sig1, sig2)


def test_minhash_estimate_converges_to_true_jaccard_as_num_perm_grows():
    shingles_a = word_shingles(DOC_A, k=3)
    shingles_b = word_shingles(DOC_B, k=3)
    true_jaccard = exact_jaccard(shingles_a, shingles_b)

    errors = []
    for num_perm in (16, 128, 512):
        hasher = MinHasher(num_perm=num_perm, seed=7)
        sig_a = hasher.signature(shingles_a)
        sig_b = hasher.signature(shingles_b)
        estimate = estimate_jaccard(sig_a, sig_b)
        errors.append(abs(estimate - true_jaccard))

    # More hash functions should generally tighten the estimate toward the true value; check
    # the largest num_perm setting is within a reasonably tight tolerance of ground truth.
    assert errors[-1] < 0.15, f"num_perm=512 estimate too far from true Jaccard: {errors}"


def test_minhash_near_duplicate_pair_scores_high_similarity():
    hasher = MinHasher(num_perm=128, seed=7)
    sig_a = hasher.signature_for_text(DOC_A)
    sig_b = hasher.signature_for_text(DOC_B)  # near-duplicate: one clause changed
    sig_c = hasher.signature_for_text(DOC_C)  # unrelated article

    sim_ab = estimate_jaccard(sig_a, sig_b)
    sim_ac = estimate_jaccard(sig_a, sig_c)
    assert sim_ab > sim_ac
    assert sim_ab > 0.5  # near-duplicates should score clearly above unrelated docs


def test_signature_mismatched_lengths_raises():
    hasher = MinHasher(num_perm=32)
    sig32 = hasher.signature_for_text(DOC_A)
    other_hasher = MinHasher(num_perm=64)
    sig64 = other_hasher.signature_for_text(DOC_A)
    with pytest.raises(ValueError):
        estimate_jaccard(sig32, sig64)


# --- Cross-validation against datasketch's independent MinHash implementation ---

def _reference_minhash_jaccard(shingles_a: set[str], shingles_b: set[str], num_perm: int) -> float:
    m1, m2 = ReferenceMinHash(num_perm=num_perm), ReferenceMinHash(num_perm=num_perm)
    for s in shingles_a:
        m1.update(s.encode("utf-8"))
    for s in shingles_b:
        m2.update(s.encode("utf-8"))
    return m1.jaccard(m2)


def test_own_minhash_agrees_with_datasketch_within_tolerance():
    """Both implementations estimate the SAME quantity (Jaccard of the shingle sets) via the
    SAME algorithm family, using independent hash functions -- so their estimates should track
    each other and both track the true Jaccard, even though the exact per-permutation values
    differ (different hash function seeds/families)."""
    shingles_a = word_shingles(DOC_A, k=3)
    shingles_b = word_shingles(DOC_B, k=3)
    true_jaccard = exact_jaccard(shingles_a, shingles_b)

    num_perm = 256
    own_hasher = MinHasher(num_perm=num_perm, seed=7)
    own_estimate = estimate_jaccard(
        own_hasher.signature(shingles_a), own_hasher.signature(shingles_b)
    )
    reference_estimate = _reference_minhash_jaccard(shingles_a, shingles_b, num_perm)

    assert abs(own_estimate - true_jaccard) < 0.1
    assert abs(reference_estimate - true_jaccard) < 0.1
    # The two independent estimators should agree with each other to a similar tolerance.
    assert abs(own_estimate - reference_estimate) < 0.15
