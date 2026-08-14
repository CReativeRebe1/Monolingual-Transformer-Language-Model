"""
Ties exact dedup + MinHash-LSH near-duplicate detection into one "which documents to keep"
decision -- the actual stage-5 pipeline step that lma/dedup/{exact,minhash,lsh}.py are building
blocks for.

Two passes: (1) exact dedup drops byte-identical documents outright (cheap, no false positives);
(2) LSH generates near-duplicate *candidates* among the survivors, each candidate pair is
verified against a real Jaccard threshold (LSH banding has a nonzero false-positive rate near
the S-curve's threshold, so nothing is dropped on LSH's say-so alone), and verified pairs are
merged into clusters via union-find -- transitively, since "A near-dupes B" and "B near-dupes C"
should keep only one of {A, B, C} even if A and C individually fall just under threshold. One
representative (first-seen, i.e. lowest original index) is kept per cluster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lma.dedup.exact import ExactDeduplicator
from lma.dedup.lsh import LSHIndex
from lma.dedup.minhash import MinHasher, estimate_jaccard, word_shingles


class _UnionFind:
    """Minimal union-find (disjoint-set) with path compression, for clustering duplicate pairs."""

    def __init__(self, items):
        self._parent = {item: item for item in items}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


@dataclass
class DedupStats:
    input_count: int
    exact_duplicates_removed: int
    near_duplicate_clusters: int
    near_duplicates_removed: int
    kept_count: int


def deduplicate_documents(
    doc_ids: list[str],
    texts: list[str],
    *,
    shingle_k: int = 3,
    num_perm: int = 128,
    lsh_b: int = 8,
    lsh_r: int = 16,
    similarity_threshold: float = 0.8,
    seed: int = 42,
) -> tuple[set[str], DedupStats]:
    """Returns (doc_ids to keep, stats). doc_ids and texts must be the same length and order.

    Exact-duplicate documents are dropped first; among the remainder, near-duplicate CANDIDATE
    pairs from LSH banding are individually verified against `similarity_threshold` using the
    estimated Jaccard from their full MinHash signatures before being merged into a cluster, so
    the removal decision is never made on LSH bucket membership alone.
    """
    assert len(doc_ids) == len(texts), "doc_ids and texts must be parallel lists"

    # Pass 1: exact dedup.
    exact_dedup = ExactDeduplicator()
    survivors: list[tuple[str, str]] = []
    exact_removed = 0
    for doc_id, text in zip(doc_ids, texts):
        if exact_dedup.is_duplicate(text):
            exact_removed += 1
        else:
            survivors.append((doc_id, text))

    # Pass 2: MinHash-LSH near-duplicate clustering among the exact-dedup survivors.
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    lsh = LSHIndex(num_perm=num_perm, b=lsh_b, r=lsh_r)
    signatures: dict[str, np.ndarray] = {}
    for doc_id, text in survivors:
        sig = hasher.signature_for_text(text, k=shingle_k)
        signatures[doc_id] = sig
        lsh.insert(doc_id, sig)

    uf = _UnionFind([doc_id for doc_id, _ in survivors])
    verified_pairs = 0
    for a, b in lsh.candidate_pairs():
        sim = estimate_jaccard(signatures[a], signatures[b])
        if sim >= similarity_threshold:
            uf.union(a, b)
            verified_pairs += 1

    # One representative per cluster: the first-seen doc_id (lowest original index) in that
    # cluster, so "keep" is deterministic given a fixed input order.
    order = {doc_id: i for i, doc_id in enumerate(doc_id for doc_id, _ in survivors)}
    cluster_representative: dict[str, str] = {}
    cluster_size: dict[str, int] = {}
    for doc_id, _ in survivors:
        root = uf.find(doc_id)
        cluster_size[root] = cluster_size.get(root, 0) + 1
        current = cluster_representative.get(root)
        if current is None or order[doc_id] < order[current]:
            cluster_representative[root] = doc_id

    kept = set(cluster_representative.values())
    # Only clusters with an actual near-dup merge (size > 1) count as "near-duplicate clusters";
    # a singleton (a document with no near-dup partner) is not a dedup finding.
    near_dup_clusters = sum(1 for root, size in cluster_size.items() if size > 1)
    near_dup_removed = len(survivors) - len(kept)

    stats = DedupStats(
        input_count=len(doc_ids),
        exact_duplicates_removed=exact_removed,
        near_duplicate_clusters=near_dup_clusters,
        near_duplicates_removed=near_dup_removed,
        kept_count=len(kept),
    )
    return kept, stats
