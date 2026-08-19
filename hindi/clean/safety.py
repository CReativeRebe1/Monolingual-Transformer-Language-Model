"""
Stage 5 -- semantic filtering, topic filtering, and safety filtering for Hindi.

The stage the earlier pipeline skipped entirely. Three functions, each Yi Figure-1 step:

* `cluster(texts)` -- Semantic Filtering. TF-IDF over character n-grams, then MiniBatchKMeans.
  Character n-grams rather than words for the same reason coherence uses them: Hindi inflection
  means surface forms differ where stems agree. Yi uses neural embeddings; this is the lightweight
  substitute, chosen because Phase 1 has no GPU budget and clustering here only needs to be good
  enough to *group* documents, not to represent their meaning.
* `topics(...)` -- Topic Filtering. Labels each cluster by its advertising-keyword rate and
  **downsamples** ad-heavy clusters rather than deleting them (Yi 2.1). The reasoning is that an
  ad-heavy cluster still contains fluent Hindi -- deleting it costs real language data, while
  keeping all of it lets commercial boilerplate dominate the topic mixture.
* `check(text)` -- Safety Filtering. A keyword blocklist plus the existing light PII redaction.

**Deliberately conservative for single tokens, strict for instructions.** `safety_max_hits`
tolerates incidental single-token hits so a news report is not discarded for one word, while a
single instructional phrase is enough to reject; a court report on a sexual assault case is legitimate Hindi
prose and losing it costs a low-resource corpus more than it protects it. This is a keyword filter
appropriate to a course project, not a content-moderation system, and it is documented as such.

Clustering is fitted on a `cluster_sample` subsample, then used to assign every document. Fitting
on 6M documents would be slow and would not move the cluster centres meaningfully.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hindi.clean.config import DEFAULT, CleanConfig  # noqa: E402
from hindi.clean.rules import words  # noqa: E402
from hindi.clean.words import AD_KEYWORDS, SAFETY_BLOCKLIST  # noqa: E402
from lma.text.filters import strip_light_pii  # noqa: E402

logger = logging.getLogger("hindi.clean.safety")


# ------------------------------------------------------------------ safety

def hits(text: str) -> tuple[int, int]:
    """Returns (token_hits, phrase_hits) as counts of DISTINCT blocklist entries present.

    Distinct, not total: a single slur repeated twenty times is one signal about the document, not
    twenty.

    The two classes are counted separately because they carry different evidence. A single-token slur
    can appear incidentally in legitimate prose -- a court report quoting testimony, an article about
    online abuse -- so a lone hit must not condemn the document. A multi-word instructional phrase
    ("आत्महत्या कैसे करें") has no such innocent reading; it is the thing itself.
    """
    lowered = text.casefold()
    ws = set(words(lowered))
    tokens = phrases = 0
    for term in SAFETY_BLOCKLIST:
        if " " in term:
            phrases += 1 if term in lowered else 0
        else:
            tokens += 1 if term in ws else 0
    return tokens, phrases


def check(text: str, *, config: CleanConfig | None = None) -> tuple[bool, str, int]:
    """Returns (is_safe, redacted_text, n_hits) where n_hits is the total across both classes.

    A document is unsafe if it contains ANY instructional phrase, or more than
    `safety_max_hits` distinct single-token blocklist terms. Counting both against one budget was
    wrong in a way a test caught: with `safety_max_hits = 2`, a document containing two explicit
    self-harm instructions was reported safe.

    PII redaction runs regardless of the verdict, and only after the filters have measured the
    document -- substituting "[EMAIL]" earlier would distort the length and symbol-ratio statistics
    that stage 3a depends on.
    """
    cfg = config or DEFAULT
    tokens, phrases = hits(text)
    safe = phrases == 0 and tokens <= cfg.safety_max_hits
    return safe, strip_light_pii(text), tokens + phrases


# ------------------------------------------------------------------ semantic + topic

def ad_rate(text: str) -> float:
    """Share of a document's word types that are advertising/commerce vocabulary."""
    ws = set(words(text))
    return len(ws & AD_KEYWORDS) / max(len(ws), 1)


@dataclass
class ClusterResult:
    labels: list[int]
    ad_rates: dict[int, float] = field(default_factory=dict)
    ad_clusters: set[int] = field(default_factory=set)
    n_clusters: int = 0
    fitted_on: int = 0


@dataclass
class ClusterModel:
    """A fitted vectoriser + k-means, separated from the act of labelling documents.

    Split out so the corpus can be labelled in chunks streamed from disk: `cluster()` below needs
    every document in memory at once, which a 7.4 GB machine cannot afford for a corpus of this size.
    `vec is None` means scikit-learn was unavailable and everything lands in one cluster, making
    stage 5 a no-op rather than an error.
    """

    vec: object = None
    km: object = None
    n_clusters: int = 1
    fitted_on: int = 0


def fit_clusters(sample: list[str], *, config: CleanConfig | None = None) -> ClusterModel:
    """Fits TF-IDF + MiniBatchKMeans on a sample of documents."""
    cfg = config or DEFAULT
    if not sample:
        return ClusterModel()
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn unavailable: semantic/topic filtering skipped")
        return ClusterModel()

    k = min(cfg.n_clusters, max(2, len(sample) // 50))
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), max_features=60_000, min_df=3, sublinear_tf=True
    )
    X = vec.fit_transform(sample)
    km = MiniBatchKMeans(n_clusters=k, random_state=cfg.seed, n_init=3, batch_size=4096)
    km.fit(X)
    return ClusterModel(vec=vec, km=km, n_clusters=k, fitted_on=len(sample))


def assign(model: ClusterModel, texts: list[str]) -> list[int]:
    """Cluster ids for one chunk of documents. All zeros if the model is unfitted."""
    if model.vec is None or not texts:
        return [0] * len(texts)
    return [int(c) for c in model.km.predict(model.vec.transform(texts))]


def ad_heavy(rates: dict[int, float], *, config: CleanConfig | None = None) -> set[int]:
    """Which clusters count as ad-heavy, given each cluster's mean ad-keyword rate.

    Relative, not absolute -- see the comment in topics() for the measurements behind both guards.
    """
    cfg = config or DEFAULT
    if len(rates) < 2:
        return set()
    ordered = sorted(rates.values())
    cutoff = ordered[min(int(len(ordered) * cfg.ad_cluster_percentile / 100.0), len(ordered) - 1)]
    median = ordered[len(ordered) // 2]
    return {
        lab for lab, r in rates.items()
        if r >= cutoff and r >= cfg.ad_cluster_min_ratio * max(median, 1e-9)
    }


def cluster(texts: list[str], *, config: CleanConfig | None = None) -> ClusterResult:
    """Assigns each text a cluster id. Returns all-zeros if scikit-learn is unavailable.

    Degrading to a single cluster rather than raising is the safe direction: stage 5 becomes a
    no-op and the corpus is unchanged, whereas failing hard would block the whole cascade over an
    optional refinement.
    """
    cfg = config or DEFAULT
    if not texts:
        return ClusterResult(labels=[], n_clusters=0)
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn unavailable: semantic/topic filtering skipped")
        return ClusterResult(labels=[0] * len(texts), n_clusters=1)

    k = min(cfg.n_clusters, max(2, len(texts) // 50))
    sample = texts[: cfg.cluster_sample]
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), max_features=60_000, min_df=3, sublinear_tf=True
    )
    X = vec.fit_transform(sample)
    km = MiniBatchKMeans(n_clusters=k, random_state=cfg.seed, n_init=3, batch_size=4096)
    km.fit(X)

    labels: list[int] = []
    step = 20_000
    for start in range(0, len(texts), step):
        chunk = texts[start : start + step]
        labels.extend(int(c) for c in km.predict(vec.transform(chunk)))

    return ClusterResult(labels=labels, n_clusters=k, fitted_on=len(sample))


def topics(
    texts: list[str], result: ClusterResult, *, config: CleanConfig | None = None
) -> tuple[list[int], ClusterResult]:
    """Identifies ad-heavy clusters and downsamples them. Returns (indices to keep, result).

    Downsampling is deterministic (every Nth document by position, seeded only by the keep
    fraction), so a rerun on the same corpus keeps the same documents and the funnel is
    reproducible.
    """
    cfg = config or DEFAULT
    if not texts or result.n_clusters <= 1:
        return list(range(len(texts))), result

    by_cluster: dict[int, list[int]] = {}
    for i, lab in enumerate(result.labels):
        by_cluster.setdefault(lab, []).append(i)

    rates: dict[int, float] = {}
    for lab, idxs in by_cluster.items():
        probe = idxs[:500]      # a sample is enough to characterise a cluster's ad-density
        rates[lab] = sum(ad_rate(texts[i]) for i in probe) / len(probe)

    # "Ad-heavy" is defined RELATIVE to this corpus, not by an absolute rate. Measured over 10,000
    # Hindi crawl documents in 60 clusters: cluster ad-rates run 0.0000 to 0.0214 with a median of
    # 0.0016, because ad_rate is a share of a document's word *types* and an article has hundreds
    # of them. The absolute threshold this code first carried (0.35) could never fire. A "3x the
    # median" rule fires far too readily in the other direction -- it flagged 17 clusters holding
    # 36% of the corpus, and downsampling a third of a low-resource corpus to suppress advertising
    # is a bad trade. Taking the top percentile instead flags the 3 genuinely commercial clusters
    # (~3% of documents), and the ratio guard means that if no cluster stands out, nothing is
    # flagged at all.
    ordered = sorted(rates.values())
    cutoff_idx = min(int(len(ordered) * cfg.ad_cluster_percentile / 100.0), len(ordered) - 1)
    cutoff = ordered[cutoff_idx]
    median = ordered[len(ordered) // 2]
    ad_clusters = {
        lab
        for lab, r in rates.items()
        if r >= cutoff and r >= cfg.ad_cluster_min_ratio * max(median, 1e-9)
    }
    keep: list[int] = []
    for lab, idxs in by_cluster.items():
        if lab in ad_clusters:
            stride = max(int(1 / cfg.ad_cluster_keep_fraction), 1)
            keep.extend(idxs[::stride])
        else:
            keep.extend(idxs)

    result.ad_rates = rates
    result.ad_clusters = ad_clusters
    return sorted(keep), result
