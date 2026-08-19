"""
Every tunable threshold for the Nepali cleaning cascade, in one place.

Starts from hindi/clean/config.py's values, as the plan requires, and diverges only where inspecting
Nepali data justified it. Each divergence below carries the measurement that motivated it -- that
per-language judgement is the point of splitting the pipelines, and Lecture 3 (slide 30) is explicit
that a threshold tuned on the higher-resource language will quietly empty the lower-resource corpus.

Nepali is the lower-resource side here in every sense that matters to cleaning: 85% of its manual
corpus is fast-crawl (Hindi's is a third Wikipedia), its Wikipedia is 8.5M words against Hindi's
51.9M, and fastText's lid.176 has seen far less Nepali than Hindi. Where a threshold could go either
way, it is set to keep text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LANGUAGE = "nepali"
FASTTEXT_CODE = "ne"


@dataclass
class CleanConfig:
    # ---- stage 2: language filtering
    lid_min_confidence: float = 0.50
    # DIVERGES from Hindi's 0.65, on measurement. Over 2,500 Nepali Wikipedia and 2,500 crawl
    # documents, lid.176 labels 95.9%/95.2% of them 'ne' correctly but with median confidence only
    # 0.76/0.69 -- it is right and unsure, because Nepali is genuinely low-resource in that model.
    # Applying Hindi's 0.65 bar would reject 17.9% of correctly-labelled Nepali Wikipedia and 22.5%
    # of correctly-labelled crawl. At 0.50 those become 3.0% and 0.7%. Discarding a fifth of a
    # low-resource corpus because the classifier has seen little of the language is exactly the
    # failure Lecture 3 (slide 30) describes.
    lid_min_devanagari_ratio: float = 0.60

    # ---- stage 3a: text metric filtering
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

    # ---- stage 3a: repetitive document removal (Gopher, Rae et al. 2021)
    max_top_word_fraction: float = 0.16
    max_top_2gram_fraction: float = 0.20
    max_top_3gram_fraction: float = 0.18
    max_dup_ngram_fraction: float = 0.40
    # Same value as Hindi, but note it binds far less here: measured per source, Nepali's dup-5gram
    # fraction reaches only p99 = 0.220 (wiki), 0.270 (fast) and 0.405 (crawled), against Hindi
    # crawl's p90 of 0.88. The Nepali crawl is markedly less template-duplicated, so this rule
    # removes little -- which is a fact about the corpora, not a reason to tighten the threshold.
    dup_ngram_n: int = 5
    max_dup_para_fraction: float = 0.30

    # ---- stage 3b: rule-based correction
    max_repair_char_drop_fraction: float = 0.50
    # Measured in characters, not lines: Nepali news portals, like the Hindi ones, emit an article as
    # one long prose line plus several short furniture lines, and a line-based valve reads that as
    # mostly-boilerplate and refuses to repair a document that is 95% prose by volume.

    # ---- stage 3c: perplexity + quality + coherence
    ppl_percentile: float = 99.0
    quality_min_score: float = 0.10
    # DIVERGES from Hindi's 0.30, and by a lot, for a structural reason. The classifier separates
    # Nepali more sharply than Hindi (held-out means: wiki 0.766 vs crawl 0.177; Hindi's were 0.841
    # vs 0.390), because Nepali Wikipedia is small (8.5M words) and encyclopedic while 85% of the
    # Nepali manual corpus is news fast-crawl -- so "Wikipedia-like" is a narrower target here.
    # Measured keep rates (wiki / crawl): 0.30 -> 87.0% / 19.2%, 0.20 -> 92.4% / 26.8%,
    # 0.10 -> 96.8% / 49.4%, 0.05 -> 97.8% / 70.6%. Hindi's 0.30 keeps 94% / 55%, so 0.10 is the
    # Nepali setting that matches that operating point. Keeping Hindi's number would discard 81% of
    # the crawl -- four fifths of the entire Nepali corpus -- on a proxy for quality, not evidence
    # of junk.
    coherence_min_cosine: float = 0.01        # fallback only; calibrate_coherence() overrides it
    coherence_percentile: float = 1.0
    coherence_min_paragraphs: int = 3
    coherence_min_words: int = 150

    # ---- stage 4: deduplication
    para_dup_min_chars: int = 40
    para_dup_min_docs: int = 3
    near_dup_threshold: float = 0.80
    shingle_k: int = 3
    num_perm: int = 128
    lsh_b: int = 8
    lsh_r: int = 16

    # ---- stage 5: semantic + topic + safety
    n_clusters: int = 60
    cluster_sample: int = 50_000
    # Documents sampled to fit TF-IDF + KMeans, then used to label the whole corpus in chunks. 50k
    # rather than 200k because the sample is the one point where stage 5 holds text in memory, and
    # this machine has 7.4 GB total: 200k documents is ~1 GB of str for no benefit, since k-means
    # centres over 60 clusters stop moving long before 50k samples.
    ad_cluster_keep_fraction: float = 0.25
    ad_cluster_percentile: float = 95.0
    ad_cluster_min_ratio: float = 2.0
    safety_max_hits: int = 2
    # Applies to single-token blocklist terms ONLY; multi-word instructional phrases reject on
    # the first hit. Sharing one budget meant two explicit self-harm instructions read as safe.

    # ---- corpus budget
    target_manual_fraction: float = 0.25
    seed: int = 42

    # ---- splits
    split_fractions: tuple[float, float, float] = (0.98, 0.01, 0.01)
    holdout_oversample: float = 2.0
    holdout_manual_fraction: float = 0.25
    # Held-out documents are decontaminated against TRAIN rather than the other way round: candidates
    # that share a 13-gram with a training document are discarded and replaced from the oversampled
    # remainder, so training keeps every document. Measured on the first full Nepali run, the other
    # direction cost 123,357 of 266,189 manual train documents (46.3%) and pushed the manual share to
    # 15.0%, under the 20% floor -- because the Nepali crawl reuses syndicated sentences so heavily
    # that each held-out manual article collided with ~23 training articles. Candidates are drawn
    # mostly from the downloaded side (holdout_manual_fraction matches the corpus's own composition),
    # which keeps the eval set representative and the candidate index small, since downloaded
    # documents are much shorter.
    max_val_docs: int = 5_000
    max_test_docs: int = 5_000
    # The fractions above decide WHICH documents are held out; these caps decide HOW MANY, and the
    # caps bind at corpus scale. Held-out size has two independent costs, and the second one is
    # brutal: every held-out document's 13-grams go into the decontamination index, and every train
    # document sharing one is dropped. Measured on the first full Nepali run, 1% of 5.2M downloaded
    # documents gave 110,102 held-out documents and 6,842,371 index n-grams, which discarded 128,567
    # of 265,623 manual train documents -- 48% of the scarce side of the corpus, to protect a
    # held-out set twenty times larger than any evaluation needs. 5,000 documents each is ample for
    # perplexity and BPB (Phase 2 reports both over a few million tokens) and shrinks the index by
    # roughly 20x. Overflow documents go to train, chosen deterministically by stream position so
    # repeated passes agree.
    extra: dict = field(default_factory=dict)


DEFAULT = CleanConfig()
