"""
MinHash signatures over word-shingles, implemented from scratch.

Stage 5 of the cleaning cascade (Lecture 3 covered MinHash-LSH in depth, so this is implemented
directly rather than imported -- see lma/dedup/lsh.py for the banding/bucketing half, and
tests/test_minhash.py for validation against the `datasketch` reference implementation).

Approach: a document is represented as its set of k-word shingles (overlapping windows of k
consecutive words). For h independent hash functions, the MinHash signature stores, for each
hash function, the minimum hashed shingle value. Two documents' estimated Jaccard similarity is
the fraction of signature positions where their minimums agree -- the standard MinHash identity,
P[min_i(h(A)) == min_i(h(B))] == |A n B| / |A u B| for a hash function drawn uniformly at random
from a suitable family.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# A 61-bit Mersenne prime, standard choice for the universal hash family a*x + b mod p mod 2^32.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH_32 = (1 << 32) - 1

_WORD_RE = re.compile(r"\S+")


def word_shingles(text: str, k: int = 3) -> set[str]:
    """k-word (default trigram) shingles over whitespace-tokenized text.

    Word shingles (rather than character shingles) are the right granularity for near-duplicate
    *document* detection in morphologically rich, space-delimited Devanagari text: they are
    robust to the same article being re-published with a different headline or a different lede
    paragraph while the shared body text still overlaps almost entirely, which is the common
    real-world near-duplicate case in a news corpus (syndicated wire copy, republished articles).
    """
    words = _WORD_RE.findall(text)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


@dataclass
class MinHasher:
    """Generates MinHash signatures with a fixed, seeded family of hash functions.

    The same MinHasher instance (same num_perm, same seed) must be used to sign every document
    that will be compared -- signatures from two different hash-function families are not
    comparable. This is why num_perm/seed are constructor arguments rather than per-call ones.
    """

    num_perm: int = 128
    seed: int = 42

    def __post_init__(self):
        rng = np.random.RandomState(self.seed)
        # Universal hash family: h_i(x) = (a_i * x + b_i) mod p mod 2^32, a_i drawn from [1, p-1]
        # so no hash function is degenerately constant.
        self._a = rng.randint(1, _MERSENNE_PRIME - 1, size=self.num_perm, dtype=np.int64)
        self._b = rng.randint(0, _MERSENNE_PRIME - 1, size=self.num_perm, dtype=np.int64)

    def signature(self, shingles: set[str]) -> np.ndarray:
        """Returns a (num_perm,) uint64 array: the MinHash signature of the shingle set."""
        if not shingles:
            return np.full(self.num_perm, _MAX_HASH_32, dtype=np.uint64)
        shingle_hashes = np.array(
            [hash(s) & _MAX_HASH_32 for s in shingles], dtype=np.int64
        )
        # Broadcast: (num_perm, 1) hash-fn params against (1, n_shingles) shingle hashes.
        permuted = (
            np.outer(self._a, shingle_hashes) + self._b[:, None]
        ) % _MERSENNE_PRIME % (_MAX_HASH_32 + 1)
        return permuted.min(axis=1).astype(np.uint64)

    def signature_for_text(self, text: str, k: int = 3) -> np.ndarray:
        return self.signature(word_shingles(text, k=k))


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity from two MinHash signatures: fraction of matching positions."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length (same MinHasher)")
    return float(np.mean(sig_a == sig_b))


def exact_jaccard(shingles_a: set[str], shingles_b: set[str]) -> float:
    """Ground-truth Jaccard over the actual shingle sets, used only to validate MinHash's estimate."""
    if not shingles_a and not shingles_b:
        return 1.0
    union = shingles_a | shingles_b
    if not union:
        return 1.0
    return len(shingles_a & shingles_b) / len(union)
