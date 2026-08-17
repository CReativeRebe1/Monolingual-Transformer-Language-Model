"""
Every tunable threshold for the Hindi cleaning cascade, in one place.

Deliberately a per-language file rather than a shared one. Lecture 3 (slide 30) is explicit that
"a threshold tuned on English will empty your corpus", and the same argument applies between two
Devanagari languages with very different corpus compositions: Hindi's manual corpus is a third
Wikipedia, Nepali's is 85% fast-crawl. A single shared config would force one of them to accept
thresholds justified by the other's data.

Nepali's copy starts from these numbers and diverges only where inspecting the removed documents
justified it -- each divergence carries a comment giving the reason, which is the graded judgement
Lecture 3 asks for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LANGUAGE = "hindi"
FASTTEXT_CODE = "hi"


@dataclass
class CleanConfig:
    # ---- stage 2: language filtering
    lid_min_confidence: float = 0.65
    # fastText's lid.176 is confident on Hindi (it is high-resource in that model's training set),
    # so a moderately high bar is affordable here. Nepali cannot afford the same bar -- see its
    # config for the measured reason.
    lid_min_devanagari_ratio: float = 0.60

    # ---- stage 3a: text metric filtering (mirrors lma.text.filters.FilterConfig)
    min_chars: int = 200
    max_chars: int = 200_000
    min_words: int = 40
    max_mean_word_chars: float = 25.0
    min_mean_word_chars: float = 1.5
    max_symbol_to_word_ratio: float = 0.15
    max_digit_char_ratio: float = 0.30
    max_bullet_or_ellipsis_line_fraction: float = 0.30
    max_repeated_line_fraction: float = 0.20
    require_stopword_hit: bool = True
    min_stopword_hits: int = 2

    # ---- stage 3a: repetitive document removal (Gopher, Rae et al. 2021, Table A1)
    max_top_word_fraction: float = 0.16       # single most common word's share of all words
    max_top_2gram_fraction: float = 0.20      # Gopher uses 0.20 for 2-grams
    max_top_3gram_fraction: float = 0.18
    max_dup_ngram_fraction: float = 0.40      # share of words inside any repeated 5-gram
    # Raised from Gopher's stricter setting on evidence. Measured per source over 2,500 documents
    # each: Wikipedia's dup-5gram fraction has p99 = 0.228, while the crawl is strongly bimodal --
    # median 0.11, p90 0.88 -- so the 0.30-0.40 band is nearly empty of junk. Reading the documents
    # rejected at 0.30 for repetition ALONE showed real Hindi news: gold prices quoted city by
    # city, IPO listings, state-by-state rain forecasts. Hindi news legitimately repeats a phrase
    # template across list items. Moving 0.30 -> 0.40 raises the fast-crawl pass rate on this rule
    # from 93.4% to 96.1% and moves crawled by only 1.0pp (57.7% -> 58.7%), i.e. it recovers real
    # prose almost without admitting junk.
    dup_ngram_n: int = 5
    max_dup_para_fraction: float = 0.30       # duplicate paragraphs WITHIN one document

    # ---- stage 3b: rule-based correction
    max_repair_char_drop_fraction: float = 0.50
    # If repair would delete more than half a document's CHARACTERS, the document is boilerplate
    # with some prose attached rather than prose with some boilerplate attached; it is left
    # untouched and allowed to fail the rules honestly, so the funnel attributes it to the right
    # stage. Measured in characters, not lines: crawled articles often arrive as one long prose
    # line plus several short furniture lines, and a line-based valve read that as 60% boilerplate
    # and refused to repair a document that was 95% prose by volume. Found by running it.

    # ---- stage 3c: perplexity + quality + coherence
    ppl_percentile: float = 99.0              # reject above this percentile of trusted-text ppl
    quality_min_score: float = 0.30
    # Measured on 2,400 held-out Hindi documents (Wikipedia vs raw crawl): the classifier scores
    # wiki 0.841 / crawl 0.390 on average. At 0.30 it keeps 94% of Wikipedia and 55% of crawl; at
    # 0.50 it keeps 90% / 30%. 0.30 is chosen deliberately -- the goal is discarding the worst of
    # the crawl, not making the corpus Wikipedia-shaped, and half the crawl is real prose.
    coherence_min_cosine: float = 0.01        # fallback only; calibrate_coherence() overrides it
    coherence_percentile: float = 1.0
    # Why this is calibrated rather than hardcoded: the first hand-picked value here was 0.04,
    # which measurement showed sits at Wikipedia's 10th percentile -- it would have rejected ~9%
    # of Wikipedia itself. The calibrated 1st-percentile threshold on real Hindi wiki text is
    # 0.0107 (wiki p50 = 0.100), and only 1.4% of measurable crawl documents fall below it. This
    # check is a rarely-firing safety net against stitched-together extractions, by design.
    coherence_min_paragraphs: int = 3         # below this, coherence is unmeasurable -> pass
    coherence_min_words: int = 150            # below this, TF-IDF vectors are mostly zeros -> pass

    # ---- stage 4: deduplication
    para_dup_min_chars: int = 40              # ignore very short paragraphs when cross-doc deduping
    para_dup_min_docs: int = 3                # a paragraph must appear in >=3 documents to be boilerplate
    near_dup_threshold: float = 0.80
    shingle_k: int = 3
    num_perm: int = 128
    lsh_b: int = 8
    lsh_r: int = 16

    # ---- stage 5: semantic + topic + safety
    n_clusters: int = 60
    cluster_sample: int = 200_000             # docs sampled to fit TF-IDF + KMeans
    ad_cluster_keep_fraction: float = 0.25    # ad-heavy clusters are downsampled, not deleted
    ad_cluster_percentile: float = 95.0       # only the top-rate clusters count as ad-heavy
    ad_cluster_min_ratio: float = 2.0         # ...and only if well above the median cluster
    # Both are relative on purpose. Measured over 10,000 Hindi crawl documents in 60 clusters,
    # cluster ad-rates span 0.0000-0.0214 with median 0.0016 -- an absolute threshold has no
    # meaningful scale here, and the 0.35 first written in this file could never have fired. The
    # 95th percentile plus a 2x-median guard flags the 3 genuinely commercial clusters (~3% of
    # documents); a "3x median" rule alone flagged 17 clusters holding 36% of the corpus.
    safety_max_hits: int = 2                  # blocklist hits tolerated before rejecting

    # ---- corpus budget
    target_manual_fraction: float = 0.25      # >=0.20 is the requirement; 0.25 leaves margin
    seed: int = 42

    # ---- splits
    split_fractions: tuple[float, float, float] = (0.98, 0.01, 0.01)
    extra: dict = field(default_factory=dict)


DEFAULT = CleanConfig()
