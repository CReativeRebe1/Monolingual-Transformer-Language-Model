"""
Union-find (disjoint-set), the clustering primitive behind transitive near-duplicate grouping.

Lives here rather than inside lma/dedup/pipeline.py because two callers now need it: the batch
`deduplicate_documents` pipeline and each language's `clean/dedup.py`. It is pure structure with
no language content and no text handling at all, which is why it belongs in the shared lma/ tree
rather than being copied into hindi/ and nepali/.

Why transitive clustering at all: "A near-dupes B" and "B near-dupes C" should keep only one of
{A, B, C} even when A and C individually fall just below the similarity threshold. Deciding
pairwise instead would keep both ends of a chain of near-identical documents.
"""
from __future__ import annotations


class UnionFind:
    """Minimal union-find with path compression, for clustering duplicate pairs."""

    def __init__(self, items=()):
        self._parent = {item: item for item in items}

    def add(self, x) -> None:
        """Registers an item not known at construction time (idempotent)."""
        self._parent.setdefault(x, x)

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra
