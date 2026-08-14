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
| **Manual — crawled** (own async crawler) | amarujala.com, jagran.com, navbharattimes.indiatimes.com, bhaskar.com, livehindustan.com, aajtak.in, abplive.com, hi.wikipedia.org, hi.wikisource.org, hi.wiktionary.org, india.gov.in, pib.gov.in, bbc.com/hindi | ekantipur.com, onlinekhabar.com, setopati.com, ratopati.com, nagariknews.nagariknetwork.com, himalkhabar.com, ne.wikipedia.org, ne.wikisource.org, nepal.gov.np, bbc.com/nepali |
| **Manual — OCR** | Public-domain Hindi books/PDFs (pipeline built; see §2.4) | Public-domain Nepali books/PDFs (pipeline built; see §2.4) |
| **Downloaded** (planned) | Hindi Wikipedia (confirmed working), IndicCorp v2, OSCAR/CulturaX (pending HF terms acceptance) | Nepali Wikipedia (confirmed working), IndicCorp v2, OSCAR/CulturaX (pending HF terms acceptance) |

All 13 Hindi domains and all 10 Nepali domains are fully seeded and actively contributing (see
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

### 2.3 Cleaning cascade (mirrors Yi's 5-stage pipeline, Lecture 3)

1. **Collect** — raw JSONL.gz shards with full provenance (URL, fetch time, `sha256`).
2. **Language filter** — fastText `lid.176` **and** a from-scratch character-5-gram Hindi/Nepali
   discriminator must agree; this dual-check exists specifically because fastText alone
   demonstrably mistags real Nepali text as Hindi (confirmed live, `PHASE1_LOG.md` §4) — the
   Devanagari cross-language confusion Lecture 3 calls out by name.
3. **Heuristic filters** — Gopher/C4-style: length bounds (200–200,000 chars, ≥40 words), symbol-
   to-word ratio, mean word length (both directions), repeated-line fraction, bullet/ellipsis line
   fraction, boilerplate-phrase blocklist, per-language stopword-presence check.
4. **Learned quality filter** — a from-scratch character-4-gram perplexity scorer trained on a
   trusted Wikipedia seed; documents whose perplexity falls outside the trusted distribution's
   99th percentile are dropped.
5. **Deduplication** — exact `sha256` on normalized text, then a from-scratch MinHash-LSH
   (word-trigram shingles, 128 permutations, `b=8, r=16` targeting Jaccard ≈0.86), validated
   against `datasketch` in tests.

Devanagari-specific Unicode normalization (NFC composition, nukta consistency, danda
normalization, ZWJ/ZWNJ cleanup) runs before every stage; idempotency is enforced by test.

### 2.4 OCR pipeline (built, not yet run at scale)

PyMuPDF renders PDF pages at 300 DPI; Tesseract (`hin`/`nep` language packs) performs OCR with a
per-page confidence floor; a post-correction pass fixes Tesseract's most common Devanagari failure
mode (a stray space splitting a consonant from its following matra) before the shared normalizer
runs. The `tesseract` binary is not installed on the primary dev machine — this is a known,
documented gap; OCR at real scale runs on the Kaggle/Colab compute this project already targets
for pretraining, where `apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-nep` is a
one-line fix. Every stage before the OCR call itself (PDF rendering, post-correction rules) is
unit-tested independently of the binary.

## 3. Dataset statistics (current snapshot)

*(figures: `figures/hindi_funnel.png`, `figures/hindi_domains.png`,
`figures/hindi_length_hist.png`, `figures/hindi_langid_confusion.png`, and the Nepali equivalents)*

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Raw collected | 3,189 | 2,584 |
| Survived language filter | 3,075 | 2,419 |
| Survived heuristic filter | 780 | 2,275 |
| Survived quality filter | 455 | 2,056 |
| **Final (post-dedup)** | **442 documents** | **2,028 documents** |
| Words | 241,420 | 698,403 |
| Characters | 1,271,805 | 4,801,440 |
| Bytes (UTF-8) | 3,233,823 | 12,947,748 |
| Manual fraction | **100%** (crawled) | **100%** (crawled) |
| Train / val / test docs | 434 / 4 / 4 | 1,988 / 20 / 20 |
| Distinct contributing sources | 8 | 8 |

**Heuristic-cascade rejection breakdown (Hindi)**: `boilerplate_phrase_present` 940,
`symbol_to_word_ratio_too_high` 1,162, `repeated_line_fraction_too_high` 697,
`bullet_or_ellipsis_line_fraction_too_high` 148, `insufficient_stopword_evidence` 15,
`too_few_words` 6, `too_short` 3. The dominant rejection reasons (symbol ratio, repeated lines,
boilerplate phrases) are consistent with news-portal navigation chrome and cookie/app-install
notices — exactly the class of noise the cascade is designed to catch (see `PHASE1_LOG.md` §11
for the specific `amarujala.com` boilerplate-flooding incident this cascade was hardened against).

**Heuristic-cascade rejection breakdown (Nepali)**: `symbol_to_word_ratio_too_high` 107,
`bullet_or_ellipsis_line_fraction_too_high` 53, `insufficient_stopword_evidence` 21,
`too_few_words` 9, `repeated_line_fraction_too_high` 6.

**Deduplication**: Hindi — 6 exact duplicates, 1 near-duplicate cluster (7 documents) removed from
455 quality-filtered documents, 442 kept. Nepali — 12 exact duplicates, 16 near-duplicate clusters
(16 documents) removed from 2,056, 2,028 kept.

**Source diversity**: Hindi spans `amarujala.com` (218), `livehindustan.com` (80),
`navbharattimes.indiatimes.com` (85), `aajtak.in` (20), `bhaskar.com` (14), `hi.wikisource.org`
(15), `hi.wikipedia.org` (7), `hi.wiktionary.org` (3). Nepali spans `bbc.com` (1,779),
`ekantipur.com` (105), `onlinekhabar.com` (70), `setopati.com` (51), `ne.wikipedia.org` (15),
`nagariknews.nagariknetwork.com` (5), `himalkhabar.com` (2), `ne.wikisource.org` (1). `bbc.com`'s
share reflects a multi-hour head start before the round-robin fetch-fairness fix landed
(`PHASE1_LOG.md` §17) — its relative share will keep shrinking as the crawl continues, since every
domain now gets equal claim priority per round.

**Target vs. current**: both languages are far below the ~500M-token target at this snapshot — the
crawl is ongoing (both crawlers are running continuously, `PHASE1_LOG.md` §18) and this table will
be refreshed as it grows. If Nepali cannot reach 500M from available public + scrapable sources by
the deadline, the exact shortfall will be reported here with justification, as the assignment
explicitly permits.

## 4. Tokenizer construction

**Status**: tokenizer training and analysis code is built and tested (`lma/tokenizer/`,
10 passing tests covering exact roundtrip on training text, lossless roundtrip on unseen text via
byte-fallback, 0% UNK rate, and provably distinct Hindi/Nepali vocabularies), and
`scripts/tokenizer_sweep.py` (BPE + Unigram × {8k, 16k, 24k, 32k, 48k} per language, selecting by
a fertility elbow on held-out text) is built and ready. **The sweep has not yet been run for real
selection** — attempting it against the current snapshot's train split correctly and informatively
fails (`"Vocabulary size too high (48000). Please set it to a value <= 1950"`), since a few
hundred documents cannot support a vocabulary in the tens of thousands. This is a data-scale
limitation, not a code defect; the sweep will be re-run and this section populated with the real
selected vocabulary sizes, fertility numbers, and cross-language tokenization examples once the
corpus is large enough — planned for shortly before the Phase-1 deadline, once crawl accumulation
has run its course.

**Design decisions already fixed, independent of corpus size**:
- SentencePiece BPE and Unigram both trained per language; vocabularies never shared.
- `character_coverage=0.9995`, `byte_fallback=True` — chosen specifically so the reported UNK rate
  is exactly 0% *by construction*, not by a lucky vocabulary; this will be stated explicitly
  alongside the final UNK-rate table rather than presented as an unexplained result.
- Selection criterion: fertility (tokens/word) and bytes/token compression on held-out text,
  cross-checked against the embedding-parameter cost implied at the ~25M Phase-2 parameter budget.

## 5. Independence verification

`scripts/verify_independence.py` mechanically checks the assignment's hardest constraint —
that Model H and Model L share no data, tokenizer, vocabulary, or weights — rather than asserting
it in prose. Run against the current corpus: **Hindi 442 documents, Nepali 2,028 documents,
document-hash overlap = 0.** Tokenizer/vocabulary and checkpoint divergence checks will populate
once the tokenizer sweep (§4) and Phase-2 checkpoints exist.

## 6. Reproducibility

`scripts/reproduce.sh` runs the entire pipeline above — normalize → language-ID → heuristic filter
→ quality filter → dedup → split → stats/figures → tokenizer training → corpus encoding — end to
end on a small bundled sample corpus (`scripts/make_sample_shards.py`), in a couple of minutes,
with no GPU and no dependency on the live crawl. It operates against an isolated demo directory
(`--lang-dir`/`--report-dir` overrides to `build_corpus.py`) so it can never read, write, or
overwrite the real corpus or these real figures. Verified: 6 of 7 sample documents per language
survive the cascade (the 7th is an intentional low-content rejection test case); a tokenizer
trains; the corpus encodes to `.bin`/`.idx`.

## 7. What remains before this report is final

1. Let both crawlers (now running continuously outside any single assistant session) accumulate
   toward the ~500M-token targets.
2. Pull downloaded sources (Wikipedia at minimum; IndicCorp v2 and OSCAR/CulturaX pending
   `huggingface-cli login` acceptance of the latter two's terms), sized against the final manual
   total to keep the ≥20% manual floor comfortably satisfied.
3. Run OCR at scale on Kaggle/Colab (where `tesseract` is installable).
4. Re-run `scripts/tokenizer_sweep.py` on the final corpus and populate §4 with real vocab-size
   selection, fertility tables, token-frequency statistics, and cross-language tokenization
   examples.
5. Upload the final corpus and tokenizer files to Google Drive
   (`scripts/upload_to_drive.py`) and add the links to the top-level README.
6. Refresh every number and figure in this report against the final corpus.
