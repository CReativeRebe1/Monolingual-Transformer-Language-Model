"""Tests for lma.dedup.lsh -- LSH banding over MinHash signatures."""
import inspect

import numpy as np
import pytest

from lma.dedup.lsh import LSHIndex, find_threshold, s_curve_probability
from lma.dedup.minhash import MinHasher, word_shingles

DOC_A = "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में विस्तार से बताता है और सर्दियों की बात करता है"
DOC_B = "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में विस्तार से बताता है लेकिन गर्मियों की बात करता है"
DOC_C = "क्रिकेट मैच में भारत ने आज शानदार प्रदर्शन करते हुए जीत दर्ज की और सभी दर्शक बहुत खुश हुए"
DOC_D = "शेयर बाजार में आज भारी गिरावट देखी गई और निवेशकों को भारी नुकसान उठाना पड़ा है"


def test_bad_banding_raises():
    with pytest.raises(ValueError):
        LSHIndex(num_perm=128, b=10, r=16)  # 10*16 = 160 != 128


def test_valid_banding_constructs():
    index = LSHIndex(num_perm=128, b=8, r=16)
    assert index.b * index.r == index.num_perm


def test_duplicate_doc_id_raises():
    hasher = MinHasher(num_perm=32, seed=1)
    index = LSHIndex(num_perm=32, b=8, r=4)
    sig = hasher.signature_for_text(DOC_A)
    index.insert("doc1", sig)
    with pytest.raises(ValueError):
        index.insert("doc1", sig)


def test_near_duplicates_become_candidate_pair():
    # DOC_A vs DOC_B has true Jaccard ~0.667 (verified separately via exact_jaccard); a strict
    # b=8,r=16 banding (threshold ~0.86, see test_find_threshold_b8_r16_is_near_0_8) is tuned for
    # much higher similarity than that and would legitimately miss this pair. A looser banding
    # (b=32, r=4 -> threshold ~0.38) is the appropriate choice for catching this level of
    # near-duplication, which is realistic for "same article, one clause reworded".
    hasher = MinHasher(num_perm=128, seed=7)
    index = LSHIndex(num_perm=128, b=32, r=4)
    index.insert("a", hasher.signature_for_text(DOC_A))
    index.insert("b", hasher.signature_for_text(DOC_B))
    index.insert("c", hasher.signature_for_text(DOC_C))
    index.insert("d", hasher.signature_for_text(DOC_D))

    pairs = index.candidate_pairs()
    assert ("a", "b") in pairs
    # Unrelated documents (different topics entirely, true Jaccard ~0) should not collide.
    assert ("a", "c") not in pairs
    assert ("c", "d") not in pairs


def test_identical_documents_always_become_candidates():
    hasher = MinHasher(num_perm=128, seed=7)
    index = LSHIndex(num_perm=128, b=8, r=16)
    sig = hasher.signature_for_text(DOC_A)
    index.insert("a1", sig)
    index.insert("a2", sig)  # identical signature (e.g. exact-duplicate text)
    assert ("a1", "a2") in index.candidate_pairs()


def test_candidate_pairs_are_symmetric_and_unique():
    hasher = MinHasher(num_perm=64, seed=3)
    index = LSHIndex(num_perm=64, b=8, r=8)
    sig = hasher.signature_for_text(DOC_A)
    index.insert("x", sig)
    index.insert("y", sig)
    index.insert("z", sig)
    pairs = index.candidate_pairs()
    # 3 identical documents -> exactly 3 unique unordered pairs, each stored once (a < b).
    assert pairs == {("x", "y"), ("x", "z"), ("y", "z")}


# --- S-curve tuning ---

def test_s_curve_probability_at_similarity_one_is_one():
    assert s_curve_probability(1.0, b=8, r=16) == pytest.approx(1.0)


def test_s_curve_probability_at_similarity_zero_is_near_zero():
    assert s_curve_probability(0.0, b=8, r=16) == pytest.approx(0.0, abs=1e-9)


def test_s_curve_is_monotonically_increasing_in_similarity():
    probs = [s_curve_probability(s, b=8, r=16) for s in np.linspace(0, 1, 11)]
    assert all(a <= b + 1e-12 for a, b in zip(probs, probs[1:]))


def test_find_threshold_b8_r16_is_near_0_8():
    # The b=8, r=16 (num_perm=128) configuration cited in the Phase-1 plan targets ~0.8.
    threshold = find_threshold(b=8, r=16)
    assert 0.7 < threshold < 0.9


def test_find_threshold_matches_s_curve_crossing():
    b, r = 8, 16
    threshold = find_threshold(b, r)
    assert s_curve_probability(threshold, b, r) == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------- bucket cap / lazy enumeration

def _identical_bucket(n: int) -> LSHIndex:
    """An index where all n documents share every band -- the boilerplate pathology."""
    lsh = LSHIndex(num_perm=128, b=8, r=16)
    sig = np.arange(128, dtype=np.uint64)
    for i in range(n):
        lsh.insert(f"doc{i:05d}", sig)
    return lsh


def test_iter_pairs_is_lazy():
    """The point of the generator: nothing is enumerated until it is consumed."""
    pairs = _identical_bucket(5_000).iter_pairs()
    assert inspect.isgenerator(pairs)
    first = next(pairs)
    assert first[0] != first[1]


def test_bucket_cap_prevents_pair_explosion():
    """5,000 identical documents would be ~12.5M pairs per band uncapped; starred it is ~5k.

    This is the OOM that would otherwise hit at corpus scale -- one crawl produced 745
    byte-identical pages out of ~950, so oversized buckets are real here, not hypothetical.
    """
    lsh = _identical_bucket(5_000)
    uncapped_per_band = 5_000 * 4_999 // 2               # ~12.5M
    n_capped = sum(1 for _ in lsh.iter_pairs(max_bucket=1000))
    # 8 bands x (5,000 - 1) star edges. The generator does not dedupe across bands -- it does not
    # need to, since the caller merges with union-find and a repeated pair is a no-op there, and
    # deduping would mean holding every pair in a set, which is the allocation being avoided.
    assert n_capped == 8 * 4_999
    assert n_capped < uncapped_per_band // 100           # ~312x fewer, and it is per band


def test_starred_bucket_still_reaches_every_member():
    """Capping must not silently drop documents: union-find still collapses the whole bucket."""
    lsh = _identical_bucket(2_000)
    seen = set()
    for a, b in lsh.iter_pairs(max_bucket=100):
        seen.add(a)
        seen.add(b)
    assert len(seen) == 2_000


def test_candidate_pairs_is_uncapped_by_default():
    """The old batch API keeps its exact previous behaviour for small inputs."""
    lsh = _identical_bucket(50)
    assert len(lsh.candidate_pairs()) == 50 * 49 // 2
