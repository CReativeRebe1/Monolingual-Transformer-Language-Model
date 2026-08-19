# Phase 1 Report — Data Collection and Tokenizer Construction

CL3410 *Language Models and Agents*, Monsoon 2026 — Individual Project. Model H = Hindi,
Model L = Nepali.

> **Status note.** This report is a living document, refreshed as the corpus grows toward its
> ~500M-token target; it will be finalized with final numbers before the 19 Aug 2026 deadline.
> Numbers below are real, drawn from the branch's current committed `stats.json` files and
> figures, not placeholders or projections — but they are a snapshot, not the final corpus. See
> [`PHASE1_LOG.md`](../PHASE1_LOG.md) for the full commit-by-commit development history,
> including every bug found by running the pipeline against live data and how it was fixed.

## 1. Language selection

**Model H — Hindi (`hi`).** Chosen as the higher-resource language: the most widely available
non-English Indian language in both public corpora (Wikipedia, CC-100, IndicCorp, OSCAR/CulturaX)
and independently scrapable news sources, giving a realistic path to the ~500M-token target.

**Model L — Nepali (`ne`).** Chosen from the assignment's allowed lower-resource list. Among the
ten options (Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo, Nepali, Sindhi),
Nepali has the largest realistic combination of public corpora and independently scrapable news
sources, giving the best chance of a working (not just illustrative) Model L within the project's
compute and time budget, while still leaving a genuine resource-tier gap against Hindi.

**Deliberate same-script pairing.** Both languages use Devanagari. This is not incidental: it
forces a genuine Hindi/Nepali language-identification problem (Lecture 3 covers this confusion
directly — see §2.2), and it turns the H-vs-L comparison into a *controlled* one — same script,
same cleaning pipeline, same model architecture (Phase 2) — so that differences observed in
Phase 3 can be attributed to data scale and quality rather than confounded by script or alphabet
size.

## 2. Data collection

### 2.1 Sources

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| **Manual — crawled** (own async crawler) | 31 domains: news (amarujala, jagran, navbharattimes, bhaskar, livehindustan, aajtak, abplive, patrika, naidunia, jansatta, tv9hindi, indiatv, …), literary/cultural (hindisamay, kavitakosh, gadyakosh, anubhuti-hindi, bharatdiscovery, hindwi, rekhta), science/government (ignca.gov.in, mygov.in), bbc.com/hindi | 26 domains: news (ekantipur, onlinekhabar, setopati, ratopati, nagariknews, himalkhabar, gorkhapatra, ujyaaloonline, nepalkhabar, thahakhabar, baahrakhari, annapurnapost, nepalnews, …), literary (sahityasangraha), bbc.com/nepali |
| **Downloaded** | Wikipedia via the Wikimedia dump (`scripts/run_wiki.py`, confirmed working), IndicCorp v2, OSCAR/CulturaX (pending HF terms acceptance) | same |

The seed lists were widened substantially (13 -> 31 Hindi, 10 -> 26 Nepali) once it became clear
that per-host politeness makes the *number of live domains*, not concurrency, the binding
throughput constraint. Every domain was probed for reachability before inclusion; roughly a
quarter of an initial candidate list were dead hostnames, CDN bot-blocks, or timeouts and were
dropped. All listed domains are seeded and contributing (see
`PHASE1_LOG.md` §16–17 for the crawler fixes that made this true — earlier snapshots had a single
dominant domain due to two separate bugs, both found and fixed by watching the pipeline run
against live data, not assumed correct).

**Why downloaded sources are not yet folded into the corpus.** The current corpus (below) is
**100% manually collected**, comfortably above the ≥20% manual floor. Pulling a large Wikipedia
or IndicCorp batch now, before the manual total's final size is known, risks overshooting the
downloaded share and having to discard work later to get back under a safe manual ratio. The plan
is to pull downloaded sources deliberately near the deadline, sized precisely against whatever the
manual crawl has reached by then, so the final ratio is calculated, not accidental.

### 2.2 Collection pipeline

Async crawler (`lma/crawl/`): SQLite-backed resumable frontier, `robots.txt` + rate-limit
compliance (≤1–2 req/s/host), sitemap-first discovery with a same-domain link-following fallback
for sitemap-less sites, `trafilatura`-based boilerplate extraction. Full crawl-politeness and
resume-safety details, and three real bugs found and fixed by running this against live traffic
(sitemap budget blocking, domain starvation across restarts, FIFO fetch-order bias), are in
`PHASE1_LOG.md` §3, §16, §17.

### 2.3 Cleaning cascade (Yi's 5-stage pipeline, all 14 steps)

The cascade is implemented **once per language** — `hindi/clean/` and `nepali/clean/` are separate
packages with separate configs — because the thresholds are the thing that must differ. Only
language-neutral machinery is shared (`lma/dedup/` for MinHash/LSH/union-find, `lma/text/` for
normalisation primitives, `lma/viz/` for plots). Each language's `clean/config.py` records the
measurement behind every threshold it sets.

| Yi step | Module | Notes |
|---|---|---|
| Data collection | `run.py` | streamed from raw shards; never fully in memory |
| Language filtering | `lang.py` | Devanagari ratio → fastText `lid.176` → Hindi/Nepali marker vote |
| **Rule-based correction** | `repair.py` | **moved before filtering** — see below |
| Text-metric filtering | `rules.py` | Gopher/C4 rules with Devanagari-aware thresholds |
| Repetitive document removal | `rules.py` | top word / 2-gram / 3-gram share, duplicate 5-gram share, intra-document duplicate paragraphs |
| Perplexity filtering | `score.py` | char-4-gram model fitted on our own Wikipedia text |
| Document quality filtering | `score.py` | logistic classifier, Wikipedia vs crawl, on structural (vocabulary-free) features |
| Coherence scoring | `score.py` | inter-paragraph TF-IDF cosine over character 4-grams |
| Paragraph deduplication | `dedup.py` | cross-document; runs **before** MinHash |
| MinHash deduplication | `dedup.py` | 128 permutations, `b=8, r=16`, Jaccard ≥0.80, verified against `datasketch` |
| Exact deduplication | `dedup.py` | SHA-256 over normalised text |
| Semantic filtering | `safety.py` | TF-IDF char n-grams + MiniBatchKMeans, 60 clusters |
| Topic filtering | `safety.py` | ad-heavy clusters downsampled to 25%, not deleted (Yi §2.1) |
| Safety filtering | `safety.py` | Devanagari blocklist + PII redaction |
| *(beyond Yi — Lecture 3 slide 24)* | `lma/dedup/decontaminate.py` | 13-gram train/held-out separation |

**Three deliberate deviations, each made on measurement:**

1. **Correction runs before filtering.** Yi's figure places rule-based correction after the metric
   filters. In that order a news article carrying furniture lines and a leaked JSON metadata blob is
   *rejected* before anything can repair it. Repairing first and filtering second recovered real
   prose: on Hindi, repair fired on 117,000 documents (JSON blobs 22,867; boilerplate lines 31,671;
   boilerplate phrase lines 25,100; mojibake 607). A safety valve returns the document untouched if
   repair would remove more than half its characters, so documents that are *mostly* boilerplate
   still fail the filters honestly and the funnel attributes them to the right stage.
2. **Coherence uses character 4-grams, not words.** Word-level TF-IDF cosine barely separated a
   coherent Devanagari article from a stitched-together one (0.031 vs 0.029) because both languages
   are inflected — consecutive paragraphs share stems, not surface forms. Character 4-grams
   separated the same pair 2× better (0.052 vs 0.024). The IDF is smoothed, `log((N+1)/(1+df)) + 1`;
   the textbook `log(N/(1+df))` is exactly zero for a term appearing in 2 of 3 paragraphs, which
   zeroes out precisely the shared terms the metric exists to detect and scored every document
   0.0000.
3. **Decontamination cleans the held-out side, not the training side.** The conventional direction
   dropped 123,357 of 266,189 Nepali manual training documents (46.3%) and pushed the manual share
   to 15.0%, under the assignment's floor — because the Nepali crawl reuses syndicated agency
   sentences so heavily that each held-out article collided with ~23 training articles. MinHash
   cannot see this (those documents are ~0.05 Jaccard-similar, not 0.8): they are distinct articles
   sharing sentences. Since we construct both sides and have millions of candidates for a
   5,000-document held-out set, colliding *candidates* are discarded and replaced from an
   oversampled pool. Training keeps every document; val/test remain 13-gram-disjoint from train. The
   documented cost is a mild bias — held-out documents are, by construction, ones that do not share
   sentences with training text.

Devanagari normalisation runs before every stage and **differs per language**: Hindi folds
chandrabindu to anusvara (modern Hindi treats them interchangeably), Nepali does not, because ँ is
standard Nepali spelling in गाउँ, ठाउँ, हुँदा — folding it would produce misspelt Nepali and fragment
the tokenizer's vocabulary against every correct occurrence. Idempotency is enforced by test.

### 2.4 Manual collection is crawl-only (no OCR)


The assignment counts *either* "OCR from books/PDFs" *or* "scraping + cleaning pages you gather
yourself" as manual collection. This project uses only the latter. An OCR pipeline (PyMuPDF
rendering + Tesseract + Devanagari post-correction) was built and unit-tested earlier in Phase 1,
but was removed: reaching the ≥20% manual floor at ~500M-token scale requires tens of millions of
manual tokens, which page-scale OCR cannot realistically supply within the phase deadline, while
the crawler supplies them directly. Keeping a second, unused collection path would have meant
carrying its dependencies (`pytesseract`, `PyMuPDF`) and its tests for no contribution to the
corpus. Manual tokens therefore come entirely from the crawl, which is fully spec-compliant.

### 2.5 Wikipedia via dump, not crawl

Wikipedia is ingested with `scripts/run_wiki.py` from the Wikimedia dump rather than crawled.
The MediaWiki API looks like the faster route but is not: for whole-article plain-text extracts
it silently caps `exlimit` at 1 — verified live against `hi.wikipedia.org`, which returned
*"exlimit was too large for a whole article extracts request, lowered to 1"* — making an API walk
one article per round-trip, no better than crawling the HTML. The dump is already plain text and
streams in bulk.

**Manual classification, confirmed with the course TA directly**: asked explicitly whether a raw
Wikipedia XML dump counts as manual collection given it "isn't a curated file, just a
language-wise organized zip" — the TA confirmed yes, provided it is preprocessed by our own
script. `lma/crawl/wiki_dump.py` does exactly that: it streams the raw `.xml.bz2`, strips
wikitext markup with `mwparserfromhell`, and runs every record through the same cleaning cascade
as the crawl (§2.3). Records are tagged `source_type="manual"` accordingly. This is distinct from
`lma/crawl/downloaded_corpus.py`'s HuggingFace-dataset ingestion path (IndicCorp v2, OSCAR/
CulturaX, and a `datasets`-library Wikipedia stream), which pulls already-extracted, pre-curated
text via a third-party library — that path remains tagged `"downloaded"`, since it is not "raw
data we preprocessed ourselves" in the sense the TA's ruling addressed.

Full dumps were processed, not a capped sample (`--max-docs` defaults to unlimited and was never
set): **116,799 Hindi articles (~68.0M tokens) and 26,898 Nepali articles (~9.7M tokens)**, both
counted toward the manual total.

## 3. Dataset statistics (final corpus)

*(figures: `figures/hindi_funnel.png`, `figures/hindi_length_hist.png`, `figures/hindi_sources.png`,
`figures/hindi_manual_vs_downloaded.png`, the Nepali equivalents, and `figures/langid_confusion.png`)*

### 3.1 Cleaning funnel (manual side)

| Stage | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Collected (raw manual) | 288,469 (100.0%) | 370,801 (100.0%) |
| After language filtering | 269,182 (93.3%) | 344,932 (93.0%) |
| After repair + text-metric rules | 197,165 (68.3%) | 306,549 (82.7%) |
| After perplexity / quality / coherence | 157,285 (54.5%) | 286,181 (77.2%) |
| After paragraph dedup | 156,390 (54.2%) | 284,132 (76.6%) |
| After exact dedup | 156,012 (54.1%) | 282,598 (76.2%) |
| After near dedup (MinHash-LSH) | 154,669 (53.6%) | 275,946 (74.4%) |
| After topic filtering | 152,714 (52.9%) | 271,591 (73.2%) |
| After safety filtering | 152,699 (52.9%) | 271,589 (73.2%) |

Hindi retains **52.9%** of its manually collected documents, Nepali
**73.2%**. The gap is real and is explained in §3.4.

### 3.2 Final corpus size

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Manual words | 86,845,726 | 102,346,029 |
| Downloaded words | 322,185,544 | 312,027,108 |
| **Total words** | **409,031,270** | **414,373,137** |
| **Total tokens** (final tokenizer) | **517,640,653** | **572,506,257** |
| Manual fraction (words) | **21.2%** | **24.7%** |
| Train / val / test documents | 3,577,045 / 5,000 / 5,000 | 5,526,928 / 5,000 / 5,000 |
| Train / val / test tokens | 515,972,223 / 821,762 / 846,668 | 571,227,702 / 634,788 / 643,767 |

Both languages exceed the ~500M-token target and both clear the ≥20% manual floor. The floor is
enforced in code, not by inspection: each pipeline raises at the end of a run if the final manual
share falls below 20% (`MANUAL_FLOOR` in `<lang>/clean/run.py`), after writing `stats.json` so the
funnel is still available for diagnosis. An earlier run produced a 15.0% corpus and reported
success with an ordinary log line; that cannot happen now.

Held-out sets are capped at 5,000 documents each rather than a percentage. At corpus scale a 1%
held-out fraction meant 110,102 documents and 6.8M 13-grams in the contamination index, which is
what caused the 46.3% training loss described in §2.3. 5,000 documents is ~0.8M tokens
(Hindi) — far more than perplexity/BPB needs.

### 3.3 Deduplication and decontamination

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Paragraphs excised (cross-document) | 510,077 across 67,059 documents | 1,836,622 across 189,039 documents |
| Exact duplicate documents | 378 | 1,534 |
| Near-duplicate documents (clusters) | 1,343 (1,077) | 6,652 (6,584) |
| Ad-heavy clusters downsampled | 3 of 60 → 1,955 documents | 3 of 60 → 4,355 documents |
| Training documents dropped to decontamination | **0** | **0** |
| Held-out candidates discarded as contaminated | 3,293 of 20,000 (16.5%) | 8,596 of 20,000 (43.0%) |

The candidate-collision rate is the most interesting cross-language number here: **16.5% for Hindi
against 43.0% for Nepali**. Nepali news portals republish agency copy far more
heavily. Note this runs *opposite* to whole-document duplication, where Hindi's crawl is the more
duplicated of the two (dup-5gram p90 of 0.88 versus Nepali's 0.405) — Hindi has more internally
repetitive template pages, Nepali more distinct articles sharing sentences. Two different
pathologies, which is why both paragraph dedup and 13-gram decontamination earn their place.

### 3.4 Why the two survival rates differ, and top rejection reasons

**Hindi** (top 6): `symbol_to_word_ratio_too_high` 41,426, `quality_too_low` 34,405, `dup_ngram_fraction` 22,435, `dup_paragraph_fraction` 19,079, `low_devanagari_ratio` 18,269, `bullet_line_fraction_too_high` 7,662.

**Nepali** (top 6): `symbol_to_word_ratio_too_high` 35,431, `quality_too_low` 18,124, `low_devanagari_ratio` 9,765, `lid_low_confidence` 6,650, `lid_said_hi` 5,050, `hindi_markers_dominant` 4,377.

Hindi loses far more to `quality_too_low` and to the repetition rules; Nepali's largest
language-specific losses are cross-language leakage (`lid_said_hi` 5,050,
`hindi_markers_dominant` 4,377) — real Hindi text caught inside the Nepali crawl, which is
exactly what the corpora-independence requirement is about.

### 3.5 Per-language thresholds, and why they differ

| Threshold | Hindi | Nepali | Reason for divergence |
|---|---:|---:|---|
| fastText LID minimum confidence | 0.65 | 0.5 | lid.176 labels 96% of real Nepali correctly but with median confidence 0.69–0.76. Hindi's 0.65 bar rejects 17.9% of correctly-labelled Nepali Wikipedia; 0.50 rejects 3.0%. |
| Quality classifier minimum | 0.3 | 0.1 | Nepali Wikipedia is small (8.5M words) and 85% of the Nepali manual corpus is news crawl, so "Wikipedia-like" is a narrower target. At Hindi's 0.30 only 19.2% of Nepali crawl survives; 0.10 keeps 49.4%, matching Hindi's operating point. |
| Perplexity cutoff (99th pct of trusted text) | 16.68 | 18.01 | Calibrated per language on that language's own Wikipedia. |
| Coherence minimum (1st pct of trusted text) | 0.01234 | 0.01061 | Calibrated, not hand-picked: the first hand-picked value (0.04) sat at Wikipedia's 10th percentile and would have rejected ~9% of Wikipedia itself. |
| Chandrabindu folded to anusvara | yes | **no** | ँ is standard Nepali spelling (गाउँ, ठाउँ, हुँदा). |

This is the concrete form of Lecture 3's warning that a threshold tuned on the higher-resource
language will quietly empty the lower-resource corpus.

### 3.6 Language identification, measured out-of-sample

*(figure: `figures/langid_confusion.png`, raw numbers in `figures/langid_confusion.json`,
regenerated by `scripts/langid_report.py`)*

The earlier version of this figure was built **in-sample** — the character-n-gram classifier was
evaluated on the very seed documents it had been fitted on — so its diagonal dominance was
guaranteed and said nothing. It has been rebuilt against labels that come from outside the pipeline
entirely: the Wikipedia dump a document came from (`hi.wikipedia.org` = Hindi,
`ne.wikipedia.org` = Nepali), read from the **raw** shards before any language filter runs. fastText
`lid.176` is a pretrained model this project never trained, so this is genuinely held-out for it.
Measuring on the cleaned splits instead would have been circular in a subtler way than the original
bug: those documents survived a fastText check, so fastText would have been scored against its own
decisions.

| | Documents | fastText correct | Confused as the *other* language | Marker vote correct |
|---|---:|---:|---:|---:|
| Hindi | 3,000 | **99.97%** | 0 | 97.30% |
| Nepali | 3,000 | **95.87%** | **119** | 99.97% |

Two findings, both of which justify design decisions taken earlier:

1. **The confusion is one-directional.** fastText labelled 119 of
   3,000 Nepali Wikipedia articles as Hindi (4.0%), and **zero**
   Hindi articles as Nepali. Nepali is the low-resource side of that model, and its errors fall
   asymmetrically toward the higher-resource neighbour — which is precisely the leakage direction
   that would contaminate the Nepali corpus with Hindi text if fastText were trusted alone.
2. **The marker vote is the better detector on the language that needs it.** On Nepali it agrees
   with the dump label 99.97% of the time against fastText's
   95.87%. The two are combined rather than ranked: fastText is stronger on Hindi
   (99.97% vs the marker vote's 97.30%), the marker vote is
   stronger on Nepali, and `lang.check` requires the script ratio, fastText and the marker vote to
   agree before a document is accepted.

This is also why Nepali runs a lower fastText confidence bar (0.50 vs Hindi's 0.65, §3.5): the model
is both less accurate *and* less confident on Nepali, and a bar tuned on Hindi discards correctly
identified Nepali text.

## 4. Tokenizer construction

Two algorithms × six vocabulary sizes per language = **12 tokenizers per language**, all trained on
a sentence-split sample of that language's own training split and evaluated on its real held-out
val split. Vocabularies are never shared between languages.

**One correction worth recording.** SentencePiece defaults `max_sentence_length` to 4192 **bytes**
and silently discards longer lines. Devanagari costs 3 bytes per character, so that is ~1,400
characters, while our splits are written one document per line averaging ~1.6 KB. A large share of
the corpus was therefore never seen by the trainer, with no error and no warning. Training input is
now split into sentences on danda (`lma/tokenizer/sample.py`), with the limit also raised
explicitly as a second line of defence. Every fertility number below post-dates that fix.

### 4.1 Sweep results — Model H (Hindi)

| Vocab | Algorithm | Fertility (tokens/word) | Chars/token | Vocab utilisation | Embedding cost at 25M |
|---:|---|---:|---:|---:|---:|
| 8,000 | bpe | 1.4197 | 3.661 | 0.967 | 16% |
| 8,000 | unigram | 1.3888 | 3.742 | 0.977 | 16% |
| 12,000 | bpe | 1.3390 | 3.882 | 0.958 | 25% |
| 12,000 | unigram | 1.3140 | 3.955 | 0.978 | 25% |
| 16,000 | bpe | 1.2926 | 4.021 | 0.948 | 33% |
| 16,000 | unigram | 1.2728 | 4.084 | 0.970 | 33% **← selected** |
| 24,000 | bpe | 1.2420 | 4.185 | 0.907 | 49% |
| 24,000 | unigram | 1.2305 | 4.224 | 0.915 | 49% |
| 32,000 | bpe | 1.2155 | 4.276 | 0.846 | 66% |
| 32,000 | unigram | 1.2073 | 4.305 | 0.841 | 66% |
| 48,000 | bpe | 1.1877 | 4.376 | 0.710 | 98% |
| 48,000 | unigram | 1.1841 | 4.389 | 0.695 | 98% |

### 4.2 Sweep results — Model L (Nepali)

| Vocab | Algorithm | Fertility (tokens/word) | Chars/token | Vocab utilisation | Embedding cost at 25M |
|---:|---|---:|---:|---:|---:|
| 8,000 | bpe | 1.5807 | 4.116 | 0.966 | 16% |
| 8,000 | unigram | 1.5411 | 4.222 | 0.972 | 16% |
| 12,000 | bpe | 1.4662 | 4.438 | 0.961 | 25% |
| 12,000 | unigram | 1.4393 | 4.521 | 0.975 | 25% |
| 16,000 | bpe | 1.4018 | 4.641 | 0.958 | 33% |
| 16,000 | unigram | 1.3802 | 4.714 | 0.970 | 33% **← selected** |
| 24,000 | bpe | 1.3271 | 4.903 | 0.936 | 49% |
| 24,000 | unigram | 1.3146 | 4.950 | 0.931 | 49% |
| 32,000 | bpe | 1.2852 | 5.063 | 0.893 | 66% |
| 32,000 | unigram | 1.2766 | 5.097 | 0.877 | 66% |
| 48,000 | bpe | 1.2385 | 5.254 | 0.789 | 98% |
| 48,000 | unigram | 1.2343 | 5.271 | 0.762 | 98% |

**Unigram beats BPE at every matched vocabulary size in both languages** — that settles the
algorithm choice on evidence rather than convention. UNK rate is exactly 0.0% everywhere, by
construction: `byte_fallback=True` means no input is unrepresentable, so selection is driven by
fertility rather than coverage.

### 4.3 Selection

**Selected: Unigram at 16,000 for both languages.** Fertility
1.2728 (Hindi) and 1.3802 (Nepali).

Fertility alone always prefers a larger vocabulary, and at Phase 2's ~25M-parameter budget that
preference is not free. The embedding table is `vocab × d_model`; at `d_model = 512` with tied
weights, and a decoder layer costing `12 × d_model²`, the vocabulary directly buys or spends depth:

| Vocab | Embedding params | Share of 25M | Layers affordable at d=512 |
|---:|---:|---:|---:|
| 8,000 | 4.1M | 16% | 6 |
| 12,000 | 6.1M | 25% | 6 |
| 16,000 | 8.2M | 33% | 5 |
| 24,000 | 12.3M | 49% | 4 |
| 32,000 | 16.4M | 66% | 2 |
| 48,000 | 24.6M | 98% | **0** |

A 48k vocabulary would spend essentially the whole budget on embeddings, leaving room for no
transformer layers at all. 16k keeps 67% of the budget for the model itself while giving up only
~3% fertility against 24k, and its vocabulary utilisation is higher (0.970 Hindi) —
the larger vocabularies are already going partly unused.

The sweep also records what happens if selection is left purely to the fertility elbow: on Nepali it
chose BPE-48k, because BPE's improvement never fell below the 3% threshold and its elbow search ran
off the end of the grid, and the result was then compared against unigram's elbow *at a different
vocabulary size*. The selection code now compares algorithms at a matched size and warns when an
elbow never fires. The final choice here is pinned explicitly and recorded as such in
`sweep_results.json`.

### 4.4 Encoded corpus

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Tokenizer | `hindi\tokenizer\hindi_unigram_16000.model` | `nepali\tokenizer\nepali_unigram_16000.model` |
| Train tokens | 515,972,223 | 571,227,702 |
| Val / test tokens | 821,762 / 846,668 | 634,788 / 643,767 |
| **Total tokens** | **517,640,653** | **572,506,257** |
| `.bin` size | 1.03 GB | 1.14 GB |

Encoded to nanoGPT-style memmap format: `.bin` holds `uint16` token ids, `.idx` holds `uint64`
cumulative document offsets, so Phase 2 can train over the corpus without loading it. Each split is
verified after writing — document count from `.idx` and token count from `.bin` are checked against
what the encoder reported, since an `.idx` disagreeing with its `.bin` would silently feed Phase 2
documents starting mid-sentence.

## 5. Independence verification

`scripts/verify_independence.py` mechanically checks the assignment's hardest constraint — that
Model H and Model L share no data, tokenizer, vocabulary, or weights — rather than asserting it in
prose. Run against the final corpus, **all checks pass**:

| Check | Result |
|---|---|
| Directory separation | no cross-language path references found |
| Document overlap | **0** (Hindi 3,587,040 documents vs Nepali 5,536,919) |
| Tokenizer files | model hashes differ (`0683a15aa93d…` vs `371ebc2d5d05…`) |
| Vocabularies | both 16,000 pieces, 5,695 shared |
| Checkpoints | skipped — Phase 2 not started |

**Cross-tokenizer evidence.** `scripts/tokenizer_comparison.py` tokenizes ten real held-out
documents under *both* tokenizers. **0 of 10 produce identical segmentations**, and the
same text costs materially different amounts under each model (427 vs 668 tokens, 342 vs 563,
1,388 vs 2,333; means 512 vs 562). This is the behavioural counterpart to
the file-hash check above: the two vocabularies are not merely different files, they segment the
same Devanagari differently. The raw dump is regenerable and is not committed.

The 5,695 shared vocabulary pieces are worth stating plainly rather than hiding: the two languages
share a script and a large Sanskrit-derived lexicon, so a substantial overlap in *pieces* is
expected and is not a violation. What the requirement forbids is a shared tokenizer or shared
training data, and both are verified distinct — the models were trained independently on disjoint
corpora, and 10,305 of each vocabulary's pieces occur only in that language.

## 6. Reproducibility

`scripts/reproduce.sh` runs the entire pipeline end to end on a small bundled sample corpus
(`scripts/make_sample_shards.py`), in a couple of minutes, with no GPU and no dependency on the live
crawl. It operates against an isolated demo directory (`--lang-dir`/`--report-dir` overrides) so it
can never read, write, or overwrite the real corpus or the real figures.

Reproducibility of the *real* build rests on three things:

1. **Deterministic hashing.** MinHash shingles are hashed with `xxhash`, not Python's builtin
   `hash()`, which is salted per process by `PYTHONHASHSEED` and made dedup results differ run to
   run. Two independent full Nepali cascades produced byte-identical stage counts (286,181 survivors,
   43,378 recurring paragraphs, 1,836,622 paragraphs removed), which is the practical evidence that
   this holds.
2. **Deterministic splits.** Held-out membership comes from a hash of the document itself, with the
   cap applied by stream position, so repeated passes agree without sharing state and a rerun
   reproduces the same splits.
3. **Resumability.** `--from-interim` redoes only the split/decontamination phase from a previous
   run's interim files, so a change to the held-out size or the budget costs minutes rather than the
   ~1.5-hour cascade. The resumed path and the full path call the same function, so they cannot
   drift apart.

The full test suite is **372 tests**, including cross-process determinism of MinHash signatures,
MinHash validated against `datasketch`, and tokenizer metrics validated against an independently
trained HuggingFace BPE.

## 7. What remains before this report is final

1. ~~Upload the corpus artifacts.~~ **Done.** The corpus is hosted at
   [Google Drive](https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing) and on HuggingFace as
   [`YogendraPatel/LMA_Individual_Project`](https://huggingface.co/datasets/YogendraPatel/LMA_Individual_Project).

What is hosted there rather than in git, and why:

| Path | Size | Why not in git |
|---|---:|---|
| `<lang>/data/raw/` | 3.1 + 3.8 GB | Irreplaceable — re-crawling returns different pages |
| `<lang>/data/splits/` | 5.5 + 7.5 GB | The corpus itself; regenerable from raw + interim in ~40 min per language |
| `<lang>/data/encoded/` | 1.0 + 1.2 GB | Phase-2 training input; regenerable from splits + tokenizer in ~15 min |
| `<lang>/data/interim/` | 1.2 + 1.9 GB | Optional; lets `--from-interim` re-split without repeating the ~1.5 h cascade |

The selected tokenizers (~0.7 MB each) are committed to git and need no upload.

3. Optional: revisit the vocabulary choice. 12k is now measured and is a defensible alternative —
   25% of the parameter budget instead of 33%, the highest vocabulary utilisation of any size tested
   (0.978 Hindi / 0.975 Nepali), at ~3% worse fertility than 16k.
