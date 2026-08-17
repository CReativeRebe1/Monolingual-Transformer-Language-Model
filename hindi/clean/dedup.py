"""
Stage 4 -- deduplication for Hindi: paragraphs, then exact, then near.

Yi's Figure-1 order is followed deliberately, and paragraph-dedup coming *first* is not cosmetic:
removing a shared footer before MinHash runs stops two otherwise-unrelated articles from matching
each other on boilerplate they both carry. Run it the other way around and near-dedup deletes real
articles for agreeing about a cookie notice.

1. `dedup_paras(docs)` -- cross-document paragraph dedup. A paragraph appearing in at least
   `para_dup_min_docs` different documents is site furniture, not content, and is excised from
   every document. Short paragraphs are exempt (`para_dup_min_chars`): a one-line date stamp or a
   two-word section heading legitimately repeats, and deleting those would shred normal articles.
2. `dedup_exact(docs)` -- SHA-256 over normalised text. Cheap, no false positives.
3. `dedup_near(docs)` -- MinHash + LSH banding, verified against a real Jaccard threshold, clustered
   transitively with union-find. Uses `iter_pairs()` rather than `candidate_pairs()`: this corpus
   has buckets large enough that materialising every pair would exhaust memory.

The MinHash/LSH/union-find machinery is imported from `lma/dedup/` -- it is pure math with no
language content, so it is shared. What is per-language here is the *thresholds* and the paragraph
rules, which is why this file exists rather than a shared pipeline call.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hindi.clean.config import DEFAULT, CleanConfig  # noqa: E402
from hindi.clean.norm import norm  # noqa: E402
from hindi.clean.rules import paras  # noqa: E402
from lma.dedup.lsh import LSHIndex  # noqa: E402
from lma.dedup.minhash import MinHasher, estimate_jaccard, word_shingles  # noqa: E402
from lma.dedup.union import UnionFind  # noqa: E402

logger = logging.getLogger("hindi.clean.dedup")


@dataclass
class DedupStats:
    n_input: int = 0
    paras_removed: int = 0
    docs_touched_by_para_dedup: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    near_clusters: int = 0
    n_kept: int = 0
    extra: dict = field(default_factory=dict)


def key(text: str) -> str:
    """Exact-duplicate key: SHA-256 of normalised text.

    Normalisation applies to the key only -- two copies differing in whitespace or Unicode
    composition are the same document for dedup purposes, but that is a matching decision, not a
    licence to rewrite what gets stored.
    """
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def dedup_paras(
    texts: list[str], *, config: CleanConfig | None = None
) -> tuple[list[str], int, int]:
    """Removes paragraphs that recur across documents. Returns (texts, n_removed, n_docs_touched).

    Two passes over the corpus: count paragraph occurrences, then excise the ones over threshold.
    Counting hashes rather than the paragraphs themselves keeps memory proportional to the number
    of distinct paragraphs times 32 bytes, not times their length.
    """
    cfg = config or DEFAULT
    counts: Counter[str] = Counter()
    doc_paras: list[list[str]] = []

    for text in texts:
        ps = paras(text)
        doc_paras.append(ps)
        # Count each paragraph once per document, so a footer repeated 10x inside one page does not
        # look like a footer occurring in 10 documents.
        for h in {hashlib.sha256(norm(p).encode("utf-8")).hexdigest()
                  for p in ps if len(p) >= cfg.para_dup_min_chars}:
            counts[h] += 1

    boilerplate = {h for h, c in counts.items() if c >= cfg.para_dup_min_docs}
    out: list[str] = []
    removed = touched = 0
    for ps in doc_paras:
        kept = []
        dropped_here = 0
        for p in ps:
            if len(p) >= cfg.para_dup_min_chars:
                h = hashlib.sha256(norm(p).encode("utf-8")).hexdigest()
                if h in boilerplate:
                    dropped_here += 1
                    continue
            kept.append(p)
        removed += dropped_here
        touched += 1 if dropped_here else 0
        out.append("\n\n".join(kept))
    return out, removed, touched


def dedup_exact(doc_ids: list[str], texts: list[str]) -> tuple[list[int], int]:
    """Returns (indices to keep, n_removed). First occurrence wins, so output order is stable."""
    seen: set[str] = set()
    keep: list[int] = []
    removed = 0
    for i, text in enumerate(texts):
        k = key(text)
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        keep.append(i)
    return keep, removed


def dedup_near(
    doc_ids: list[str], texts: list[str], *, config: CleanConfig | None = None
) -> tuple[list[int], int, int]:
    """MinHash-LSH near-duplicate clustering. Returns (indices to keep, n_removed, n_clusters).

    One representative per cluster: the lowest input index, so the result is deterministic given a
    fixed input order. LSH only proposes candidates -- each is verified against
    `near_dup_threshold` before anything is merged.
    """
    cfg = config or DEFAULT
    hasher = MinHasher(num_perm=cfg.num_perm, seed=cfg.seed)
    lsh = LSHIndex(num_perm=cfg.num_perm, b=cfg.lsh_b, r=cfg.lsh_r)
    sigs: list = []

    for i, text in enumerate(texts):
        sig = hasher.signature(word_shingles(text, k=cfg.shingle_k))
        sigs.append(sig)
        lsh.insert(str(i), sig)

    uf = UnionFind(range(len(texts)))
    for a, b in lsh.iter_pairs():
        ia, ib = int(a), int(b)
        if estimate_jaccard(sigs[ia], sigs[ib]) >= cfg.near_dup_threshold:
            uf.union(ia, ib)

    reps: dict[int, int] = {}
    sizes: Counter[int] = Counter()
    for i in range(len(texts)):
        root = uf.find(i)
        sizes[root] += 1
        if root not in reps or i < reps[root]:
            reps[root] = i

    keep = sorted(reps.values())
    clusters = sum(1 for s in sizes.values() if s > 1)
    return keep, len(texts) - len(keep), clusters
