"""
Exact deduplication: the cheap first pass before MinHash-LSH near-duplicate detection.

Runs on sha256(normalized_text) so it catches byte-identical documents regardless of source
(the same article syndicated to two different sections of one site, or a crawl re-fetching a
page whose URL changed but content did not) without paying MinHash-LSH's cost for pairs that
would match at similarity 1.0 anyway.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ExactDeduplicator:
    """Streaming exact-dedup: call `is_duplicate` once per document, in any order.

    Streaming (rather than "collect everything, then dedup") because the corpus is built from
    JSONL.gz shards that may total gigabytes; holding every seen hash (32 bytes each) is cheap
    even at corpus scale, holding every seen document is not.
    """

    _seen_hashes: set[str] = field(default_factory=set)

    def is_duplicate(self, text: str) -> bool:
        """Returns True (and does NOT register the hash) if this exact text was already seen."""
        h = content_hash(text)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def __len__(self) -> int:
        return len(self._seen_hashes)
