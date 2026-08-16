"""
Locality-Sensitive Hashing (banding) over MinHash signatures, implemented from scratch.

Comparing every document's MinHash signature against every other document's is O(n^2) and
infeasible at corpus scale. LSH banding makes near-duplicate *candidate* generation sub-linear:
split each num_perm-length signature into b bands of r rows each (num_perm = b * r), hash each
band to a bucket, and treat any two documents that land in the same bucket for at least one band
as *candidates* for a near-duplicate pair (their exact/estimated Jaccard is still checked before
actually removing one -- LSH only prunes the search space, it does not decide dedup on its own).

The banding parameters (b, r) determine the S-curve: the probability that a pair with true
Jaccard similarity s is caught by at least one band is 1 - (1 - s^r)^b. Lecture 3 covered tuning
this curve; see lma/dedup/lsh.py's `s_curve_probability` and tests/test_minhash.py for the
concrete b=8, r=16 (num_perm=128) choice used in the Phase-1 report, which targets a similarity
threshold of about 0.8.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


def s_curve_probability(similarity: float, b: int, r: int) -> float:
    """P[a pair with true Jaccard `similarity` shares a bucket in >=1 of b bands of r rows]."""
    return 1.0 - (1.0 - similarity**r) ** b


def find_threshold(b: int, r: int) -> float:
    """The similarity s* at which the S-curve 1 - (1 - s^r)^b crosses probability 0.5 exactly.

    Solved in closed form: 1 - (1 - s^r)^b = 0.5  =>  s* = (1 - 0.5^(1/b))^(1/r).
    (A commonly cited *approximation*, s* ~= (1/b)^(1/r), instead locates where a single band's
    match probability s^r ~= 1/b -- a nearby but distinct point on the curve. The exact form is
    used here since tests assert the literal 0.5 crossing.)
    """
    return (1.0 - 0.5 ** (1.0 / b)) ** (1.0 / r)


@dataclass
class LSHIndex:
    """Bands MinHash signatures into buckets and reports candidate near-duplicate pairs.

    num_perm must equal b * r; this is asserted at construction so a caller cannot silently
    build an index whose banding does not evenly partition the signature.
    """

    num_perm: int
    b: int
    r: int
    _buckets: list[dict[bytes, list[str]]] = field(default_factory=list, init=False)
    _doc_ids_seen: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        if self.b * self.r != self.num_perm:
            raise ValueError(
                f"b * r must equal num_perm: got b={self.b}, r={self.r}, "
                f"b*r={self.b * self.r} != num_perm={self.num_perm}"
            )
        self._buckets = [defaultdict(list) for _ in range(self.b)]

    def insert(self, doc_id: str, signature: np.ndarray) -> None:
        if len(signature) != self.num_perm:
            raise ValueError(f"signature length {len(signature)} != num_perm {self.num_perm}")
        if doc_id in self._doc_ids_seen:
            raise ValueError(f"doc_id {doc_id!r} already inserted")
        self._doc_ids_seen.add(doc_id)
        for band_idx in range(self.b):
            band = signature[band_idx * self.r : (band_idx + 1) * self.r]
            bucket_key = band.tobytes()
            self._buckets[band_idx][bucket_key].append(doc_id)

    def query(self, signature: np.ndarray) -> set[str]:
        """doc_ids already indexed that share >=1 band bucket with `signature`.

        This is the one-against-many lookup, as opposed to candidate_pairs()'s many-against-many.
        It is what streaming dedup needs -- "does this incoming document match anything I have
        seen?" -- and it costs O(b) dict lookups with no dependence on corpus size. It also
        sidesteps candidate_pairs()'s pair-explosion entirely, since nothing is enumerated: a
        bucket holding 20k near-identical boilerplate documents returns 20k ids here, never the
        200M pairs that enumerating them would produce.

        Like candidate_pairs(), these are *candidates* -- verify with estimate_jaccard against a
        real threshold before treating a hit as a duplicate.
        """
        if len(signature) != self.num_perm:
            raise ValueError(f"signature length {len(signature)} != num_perm {self.num_perm}")
        hits: set[str] = set()
        for band_idx in range(self.b):
            band = signature[band_idx * self.r : (band_idx + 1) * self.r]
            hits.update(self._buckets[band_idx].get(band.tobytes(), ()))
        return hits

    def candidate_pairs(self) -> set[tuple[str, str]]:
        """All (doc_id_a, doc_id_b) pairs sharing >=1 band bucket, a<b lexicographically.

        These are *candidates* only -- callers should verify with estimate_jaccard/exact_jaccard
        against a real similarity threshold before treating a pair as an actual duplicate, since
        LSH banding has a nonzero false-positive rate away from the S-curve's threshold region.
        """
        pairs: set[tuple[str, str]] = set()
        for band_buckets in self._buckets:
            for doc_ids in band_buckets.values():
                if len(doc_ids) < 2:
                    continue
                sorted_ids = sorted(doc_ids)
                for i in range(len(sorted_ids)):
                    for j in range(i + 1, len(sorted_ids)):
                        pairs.add((sorted_ids[i], sorted_ids[j]))
        return pairs
