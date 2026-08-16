"""
A one-against-many duplicate index: build it from a reference corpus, then ask of each incoming
document "have I seen this already?".

This is the streaming counterpart to lma/dedup/pipeline.py's batch `deduplicate_documents`. The
batch version answers "which documents in this list duplicate each other?" and needs every
document in memory at once; this one answers "does this single new document duplicate anything in
the reference set?" and never materialises the incoming side at all.

Built for cross-source dedup, where the two sides are not equal partners: the reference corpus is
the side we want to KEEP (manually collected text, which the assignment requires to be >=20% of the
final corpus and which we cannot simply fetch more of), and the queried side is the one we discard
from (public downloaded corpora, which are effectively unlimited -- dropping a document there costs
nothing because the next one is free). Because of that asymmetry a false positive here is cheap and
a false negative is not, which is why the near-duplicate threshold is deliberately not set high.

Two checks, cheapest first:
  1. exact  -- SHA-256 over normalised text
  2. near   -- MinHash/LSH candidate lookup, then verified with estimate_jaccard

Normalisation applies to the *hash key only*. Callers keep their original text: two corpora that
differ merely in whitespace or Unicode composition are the same document for dedup purposes, but
that is a matching decision, not a licence to rewrite what gets stored.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from lma.dedup.lsh import LSHIndex
from lma.dedup.minhash import MinHasher, estimate_jaccard, word_shingles
from lma.text.normalize import normalize

logger = logging.getLogger("lma.dupindex")

EXACT = "exact"
NEAR = "near"


def hash_key(text: str) -> str:
    """SHA-256 of normalised text -- the exact-duplicate key.

    Normalising first is what lets a crawled copy and a public-corpus copy of the same article
    match despite differing whitespace or Unicode composition. See the module docstring: this
    affects the key only, never the stored document.
    """
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


@dataclass
class DupIndex:
    """Reference-corpus duplicate index. Build with `add`/`add_many`, query with `seen`."""

    num_perm: int = 128
    lsh_b: int = 8
    lsh_r: int = 16
    shingle_k: int = 3
    threshold: float = 0.8
    seed: int = 42

    _exact: set[str] = field(default_factory=set, init=False)
    _sigs: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._hasher = MinHasher(num_perm=self.num_perm, seed=self.seed)
        self._lsh = LSHIndex(num_perm=self.num_perm, b=self.lsh_b, r=self.lsh_r)

    def __len__(self) -> int:
        return len(self._sigs)

    def add(self, text: str) -> None:
        """Indexes one reference document (both exact key and MinHash signature)."""
        self._exact.add(hash_key(text))
        sig = self._hasher.signature(word_shingles(text, k=self.shingle_k))
        # Row index doubles as the doc id: the caller never needs to know *which* reference
        # document matched, only that one did, so carrying real ids would waste memory at
        # hundreds of thousands of documents.
        self._lsh.insert(str(len(self._sigs)), sig)
        self._sigs.append(sig)

    def add_many(self, texts) -> int:
        n = 0
        for t in texts:
            self.add(t)
            n += 1
        return n

    def add_precomputed(self, key: str, sig: np.ndarray) -> None:
        """Indexes a document whose key and signature were computed elsewhere (e.g. in a worker
        process). Safe only because the shingle hash is process-stable -- see
        lma/dedup/minhash.py's `_base_hash`, which exists specifically to make this valid."""
        self._exact.add(key)
        self._lsh.insert(str(len(self._sigs)), sig)
        self._sigs.append(sig)

    def seen(self, text: str) -> tuple[bool, str | None]:
        """Returns (is_duplicate, kind) where kind is "exact", "near", or None."""
        if hash_key(text) in self._exact:
            return True, EXACT
        if not self._sigs:
            return False, None
        sig = self._hasher.signature(word_shingles(text, k=self.shingle_k))
        for cand_id in self._lsh.query(sig):
            if estimate_jaccard(sig, self._sigs[int(cand_id)]) >= self.threshold:
                return True, NEAR
        return False, None
