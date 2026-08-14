# Phase 1 Log — Data Collection and Tokenizer Construction

Running development log for Phase 1 of the CL3410 individual project (Hindi = Model H, Nepali =
Model L). This is a **working log, updated as changes happen** — distinct from `report/phase1.md`
(the polished final report with figures/tables), which will be written once the corpus reaches
its final size, close to the 19 Aug 2026 deadline. Every entry below corresponds to one or more
real commits on the `phase-1` branch; commit hashes are included so this log stays traceable
against the actual history.

**Plan reference:** `C:\Users\ASUS\.claude\plans\read-the-assignment-carefully-foamy-kettle.md`

---

## Current status snapshot

*(updated live — see timestamp on each entry below for when a given number was true)*

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Crawl frontier: fetched | 3,136 docs | 2,315 docs |
| Crawl frontier: queued (pending) | 894,987 URLs | 6,742 URLs |
| Crawl frontier: total discovered | 898,908 URLs | 11,273 URLs |
| Domains actively contributing (`done`) | 9 (amarujala, bhaskar, navbharattimes, livehindustan, jagran, hi.wikipedia/wikisource/wiktionary, aajtak) | 9 (bbc, onlinekhabar, ekantipur, wikisource, wikipedia, setopati, himalkhabar, nagariknews, ratopati) |
| Cleaned corpus (latest `build_corpus.py` run, post-fix) | **442 docs, 241,420 words, 8 sources** | **2,028 docs, 698,403 words, 8 sources** |

Test suite: **193 passed, 1 skipped** (the skip is an OCR test that needs the `tesseract` binary,
not installed on this dev machine — see the OCR section below).

**Read this number honestly:** the cleaned corpora above (130 / 797 docs) are nowhere near the
~500M-token target yet. The crawlers have been running continuously since work started and need
to keep running across the remaining days of Phase 1 (deadline 19 Aug) to get there. Every
number in this log is a snapshot, not a final claim.

---

## What "done" means in this log

Each step below states: **what was built**, **why**, **how it was verified**, and **what commit(s)**
it lives in. Nothing is marked done without a passing test or a live-verified run against real
data — several entries below specifically describe a real bug found by running the pipeline
against live crawl data, and how it was fixed.

---

### 1. Repo scaffold and environment (commit `91173f2`)

- Created the shared `lma/` code package (installed editable via `pip install -e .`) and
  `hindi/` / `nepali/` per-language artifact directories, per the repo layout the assignment
  specifies. **Decision:** code is shared, artifacts (data/tokenizer/vocab/weights) are strictly
  separate — see `scripts/verify_independence.py` (step 12) for the mechanical proof this holds.
- `.gitignore` keeps large binaries (corpora, checkpoints, `.bin`/`.idx` encoded corpora) out of
  git, per the "large artifacts go to Drive, not git" rule.
- Installed missing dependencies: `sentencepiece`, `trafilatura`, `httpx[http2]`+`h2`,
  `aiolimiter`, `selectolax`, `sacrebleu`, `pytesseract`, `PyMuPDF`, `datasketch`,
  `fasttext-wheel`, `indic-nlp-library`, `zstandard`, `pytest`.
- **Fixed:** GitHub SSH push was failing (`Permission denied (publickey)`) because Git Bash's
  bundled `ssh.exe` doesn't see the Windows OpenSSH agent service that holds the unlocked key.
  Fix: `git config core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe"` (forward slashes —
  backslashes get mis-parsed). Applies to both Bash and PowerShell pushes now.

### 2. Devanagari Unicode normalizer (commit `bb12a9c`)

`lma/text/normalize.py` — runs before every other pipeline stage on every document, regardless
of source (crawl/OCR/download).

- NFC composition (converges decomposed nukta sequences, e.g. क + ़ , to the precomposed form,
  क़ , so exact-dedup hashing and tokenization are consistent regardless of input method).
- Stray/repeated ZWJ-ZWNJ cleanup, ASCII-pipe-to-danda normalization (`|` → । , `||` → ॥ , only
  when adjacent to Devanagari script), whitespace collapsing (paragraph breaks preserved),
  optional digit-script conversion (off by default, logged if used).
- Anusvara/chandrabindu (हंस vs हँस ) are explicitly **never** touched — merging them would change
  word meaning.
- **Verified:** 26 tests, including a full idempotency sweep (`normalize(normalize(x)) == normalize(x)`)
  since documents pass through this more than once across the pipeline.

### 3. Async crawler (commits `75db44c`, `ce5dd46`, `3ba11b1`, `4f13da7`)

`lma/crawl/` — the manual-collection engine (≥20% of tokens must be self-collected per spec).

- **`frontier.py`**: SQLite-backed URL queue. Kill the process at any point, rerun the same
  command, and it resumes exactly where it left off — in-progress rows from a crash are
  requeued on reopen. This is the data-collection analogue of Phase 2's checkpoint-resume
  requirement. 8 tests specifically verify the resume guarantee.
- **`robots.py`**: per-host robots.txt compliance + rate limiting (`Crawl-delay` honoured, 1.5s
  floor otherwise), honest identifying User-Agent with contact email.
- **`sitemap.py`**: recursive sitemap/sitemap-index resolution. **Bug found and fixed:**
  `bbc.com`'s sitemap is a single global archive index shared across every language edition,
  sharded into 100+ dated sub-sitemaps — confirmed live (75+ sequential sub-sitemap fetches
  before the fix, blocking every other domain from being seeded since seeding is sequential).
  Fixed with a `max_sitemaps_total` budget shared across the whole recursion (default 40) and by
  moving `bbc.com` to the end of both languages' domain lists.
- **`extractors.py`**: trafilatura-based boilerplate stripping (`as_dict=True` needed for
  trafilatura ≥2.0 — its `bare_extraction()` now returns a `Document` object by default, which
  broke the first version of this wrapper; fixed and verified).
- **`linkextract.py`**: same-domain link-following BFS fallback, for domains whose sitemap is
  unusable. **Real finding:** several Nepali news sites (ekantipur, onlinekhabar, setopati,
  ratopati, nagariknews) return 404/502 for every standard sitemap path — the Nepali frontier
  had only 11 total URLs after sitemap-only discovery. Link-following (denylist-filtered against
  nav/search/category/asset URLs, same-domain only, depth-capped at 4 hops) fixed this; Nepali
  frontier grew to 7,600+ URLs.
- **`async_crawler.py`**: ties it together — `ShardWriter` rolls JSONL.gz shards, politeness +
  frontier + sitemap + link-follow all composed in `run_crawl()`.
- **Verified live** against `hi.wikisource.org`: sitemap discovery → robots-respecting fetch →
  extraction → gzip shard write → read-back, real Devanagari text confirmed end to end.
- Seed domain lists: `hindi/data/seed_domains.py` (news: amarujala, jagran, navbharattimes,
  bhaskar, livehindustan, aajtak, abplive; encyclopedic: hi.wikipedia/wikisource/wiktionary;
  government: india.gov.in, pib.gov.in) and `nepali/data/seed_domains.py` (news: ekantipur,
  onlinekhabar, setopati, ratopati, nagariknews, himalkhabar; encyclopedic: ne.wikipedia/
  wikisource; government: nepal.gov.np). `DOMAIN_URL_FILTERS` restricts `bbc.com` to `/hindi/`
  or `/nepali/` paths respectively (it shares one sitemap tree across all its language editions).
- **Crawlers launched and have been running continuously since.**

### 4. Language identification (commit `086d145`)

`lma/text/langid.py` — stage 2 of the cleaning cascade.

- `FastTextLangID`: wraps fastText's `lid.176.ftz` (auto-downloaded, cached).
- `CharNgramLangID`: **our own** generative character-5-gram Naive-Bayes-style classifier,
  trained on trusted seed text, specifically for the Hindi/Nepali boundary fastText is known to
  be weak on.
- **Live-verified the reason this second model exists**: fastText correctly tags a real Hindi
  sentence `hi` (0.99 confidence) but **also mistags a real Nepali sentence as `hi`** (0.83
  confidence) — exactly the Devanagari hi/ne confusion Lecture 3 (course slides) calls out. A
  document is kept only when both signals agree.
- 6 tests on synthetic separable "languages" verify fit/predict/save-load/confusion-matrix logic.

### 5. Heuristic filter cascade + learned quality filter (commits `0865068`, `f5aff23`)

`lma/text/filters.py` (stage 3) and `lma/text/quality.py` (stage 4).

- Stage 3: 9 independent, non-short-circuiting Gopher/C4-style rules — length bounds, mean word
  length (both directions — catches URL/hash dumps AND broken single-char segmentation), symbol-
  to-word ratio, digit-char ratio, bullet/ellipsis line fraction, within-document repeated-line
  fraction, boilerplate-phrase blocklist, per-language stopword-presence check (Hindi and Nepali
  use **separate** stopword lists, verified by test to behave independently).
- Stage 4: from-scratch char-4-gram perplexity scorer trained on a trusted seed; documents whose
  perplexity is far outside the trusted distribution (99th percentile cutoff, chosen from the
  distribution itself, not guessed) are dropped.
- 22 tests (stage 3) + 8 tests (stage 4).

### 6. MinHash-LSH dedup + decontamination (commit `24f6aba`)

`lma/dedup/` — stage 5, implemented from scratch per the plan (Lecture 3 covered MinHash-LSH
banding in depth).

- `minhash.py`: own hash-family MinHash over word-trigram shingles. **Validated two ways**: the
  estimated Jaccard converges to the true (exact-set) Jaccard as `num_perm` grows, and
  independently cross-checked against `datasketch.MinHash` on the same real Hindi text.
- `lsh.py`: LSH banding (b bands × r rows) with a closed-form S-curve threshold solver;
  `b=8, r=16` (`num_perm=128`) targets similarity ≈0.86.
- `exact.py`: streaming sha256 exact-dedup.
- `decontaminate.py`: 13-word n-gram overlap detector — will be used both for train-vs-val/test
  leakage and later for decontaminating pretraining data against the Phase-3 reasoning test set.
- `pipeline.py`: ties exact-dedup + LSH-candidate-generation + threshold-verified union-find
  clustering into one keep/drop decision per document. Transitive clustering (A~B~C kept as one
  cluster even when A-C alone falls under threshold) is explicitly tested.
- 39 tests total across the four modules.

### 7. OCR pipeline (commit `6129f21`)

`lma/ocr/` — the second manual-collection source alongside the crawler.

- `pdf_ocr.py`: PyMuPDF renders PDF pages at 300 DPI, Tesseract (hin/nep language packs) OCRs
  them, per-page mean-word-confidence floor drops blank/damaged/illustration pages.
- `postcorrect.py`: fixes Tesseract's most common Devanagari failure mode (a stray space
  inserted between a consonant and its following matra, which breaks NFC recomposition since
  the pieces are separated by a space, not just decomposed), plus isolated-noise-character
  stripping, then delegates to the shared normalizer.
- **Known gap, deliberately not worked around**: the `tesseract` binary is not installed on this
  dev machine. OCR at corpus scale runs on Kaggle/Colab (the project's designated compute, see
  the plan) where `apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-nep` provides
  it. `is_tesseract_available()` lets both the pipeline and the test suite detect this and
  skip/error cleanly. PDF rendering and all post-correction logic are tested unconditionally (no
  tesseract needed); the one test that needs the real binary is marked `skipif` and currently
  skips here — **this is Phase 1's one open item that needs to run somewhere with tesseract**.

### 8. Downloaded-corpus ingestion (commit `28b24e7`)

`lma/crawl/downloaded_corpus.py` — the "downloaded" half of the manual-vs-downloaded split.

- Streams HuggingFace `datasets` (Wikipedia, OSCAR, CulturaX, IndicCorpV2) so multi-hundred-GB
  source datasets are never fully materialized — only the configured token budget is pulled.
- **Live-verified against the real Hub**: `wikimedia/wikipedia` (hi/ne) streams and writes real
  documents successfully. `oscar-corpus/OSCAR-2301` and `uonlp/CulturaX` are **confirmed gated**
  — need `huggingface-cli login` with an account that has accepted their terms, to be done on
  the Kaggle/Colab runtime that performs the real large-scale pull (not this dev machine).
  `ai4bharat/IndicCorpV2`'s actual config scheme (`config="indiccorp_v2", split="hin_Deva"/
  "npi_Deva"`) was discovered and corrected after the initially assumed `config=<lang>` scheme
  failed against the live API.
- Gated/unavailable sources are reported with the real error, not aborted — a partial ingest
  (Wikipedia only, until Hub credentials exist) still succeeds.
- 6 tests via a monkeypatched `datasets.load_dataset` (no network dependency for the test suite).

### 9. Tokenizer training and analysis (commit `36166fa`)

`lma/tokenizer/` — trains a fresh SentencePiece model per language, never a pretrained tokenizer.

- `train_sp.py`: BPE and Unigram, `byte_fallback=True` throughout — this is *why* the UNK rate
  will be exactly 0%, a reported design choice, not a lucky vocabulary.
- `analyze.py`: fertility, chars/token, UNK rate, vocab utilization, token-frequency tables,
  length-distribution histograms, side-by-side cross-language tokenization examples.
- **Verified**: trained small real BPE and Unigram models end to end (not mocked). Confirmed
  exact roundtrip on training text, **lossless** roundtrip on completely unseen text via byte
  fallback, 0% UNK rate, and — the mechanical proof behind "separate vocab per model" —
  independently-trained Hindi and Nepali vocabularies are provably different piece sets.
  (Discovered along the way: `byte_fallback=True` reserves 256 meta-pieces + 4 specials before
  any real vocabulary, so a vocab_size below ~305 fails outright — fixed test configs to 350.)
- 10 tests.

### 10. Dataset statistics and report visualizations (commit `944cf16`)

`lma/text/stats.py` (pure computation) and `lma/viz/plots.py` (pure plotting — kept strictly
separate per the assignment's guideline).

- Stats: corpus-wide word/char/byte/token counts, manual-vs-downloaded token split (strict
  all-or-nothing degrade-to-`None` if any document lacks a token count — never silently
  under-counts a partial corpus), per-source doc/token counts, length distributions, a
  `PipelineFunnel` helper for stage-survival rates.
- Plots: cleaning-cascade funnel chart, length histograms, domain distribution, tokenizer
  fertility-vs-vocab-size (multi-language), Zipf rank-frequency plot, language-ID confusion
  matrix heatmap, manual-vs-downloaded pie chart, training-loss-curve plotter (reused in Phase 2).
  Every function sets title/x-label/y-label and a legend where more than one series is drawn.
- **Fixed a flaky test**: matplotlib could auto-select an interactive `TkAgg` backend depending
  on environment, failing outright without a working Tk install. Forced `matplotlib.use("Agg")`
  explicitly — figure generation is now environment-independent.
- 27 tests (11 stats-logic + 8 plot smoke tests + fixes).

### 11. Corpus-building orchestration script (commit `2448450`, fix in `4dedca5`)

`scripts/build_corpus.py` — ties every module above into the actual runnable Phase-1 pipeline:
load raw shards (crawl+OCR+downloaded) → normalize → train a small Wikipedia-seeded langid/
quality model on the fly (self-contained, no separate offline training step needed) →
language-filter → heuristic-filter → quality-filter → dedup → document-level 98/1/1 split →
stats + figures.

- **Real bug found by running this against the live crawl for the first time**: dedup correctly
  collapsed 745 of ~950 heuristic-surviving Hindi documents down to a single representative —
  inspection showed they were all `amarujala.com`'s cookie/data-collection disclaimer, appearing
  verbatim on every `/news-archives`, `/events`, `/kavya/*` "hub" page that has no article body
  of its own. Dedup worked exactly as designed; the real fix was two-pronged:
  - Added the two exact observed boilerplate phrases (cookie notice, app-install nag) to
    `BOILERPLATE_PHRASES` in `lma/text/filters.py`, so this is rejected at stage 3 and never
    displaces real crawl budget in the kept corpus at all.
  - Extended `lma/crawl/linkextract.py`'s denylist to also exclude `/events` and any path
    segment matching `archives` (substring match — amarujala's URLs are like
    `/news-archives/auto-archives`), reducing wasted fetches on these hub pages going forward.
  - Both fixes locked in with regression tests using the exact observed text/URLs.
- Shard reads tolerate a crawler-in-progress truncated gzip tail (the crawler may still be
  appending to a shard while this script reads it) rather than crashing the whole build.
- **Run against the live, still-accumulating corpus**: Hindi 130 docs / 56,247 words kept after
  the fix (was artificially inflated with 1 boilerplate doc before it); Nepali 797 docs /
  220,893 words kept (much healthier ratio — only 5 near-duplicates removed from 807).

### 12. Tokenizer sweep script (commit `ab2157f`)

`scripts/tokenizer_sweep.py` — trains BPE and Unigram at {8k, 16k, 24k, 32k, 48k} per language,
evaluates on held-out val text, selects by an elbow criterion on fertility (UNK rate is ~0% at
every size due to byte-fallback, so fertility is the operative signal — matches the assignment's
stated selection criterion).

- **Dry-run against the current tiny corpus correctly failed informatively**: every vocab size
  attempt was skipped with a clear reason (`"Vocabulary size too high (48000). Please set it to
  a value <= 1950."`) since 75-130 documents cannot support a 32k-48k vocabulary. This is a data-
  scale limitation, not a code bug — the training/evaluation mechanism itself is proven
  separately by `test_tokenizer.py`'s 10 passing tests against adequately-sized data.
  **Needs to be re-run once the corpus is large enough** (see "What's left").

### 13. Independence verification script (commit `dbb5996`)

`scripts/verify_independence.py` — mechanically proves the assignment's hardest constraint
(Model H and Model L share no data/tokenizer/vocab/weights) rather than asserting it in prose.

- Checks: no cross-language path references in code/configs; zero `content_sha256` document
  overlap between the two corpora; tokenizer `.model` files and vocab piece sets differ
  (byte-identity or identical piece sets both hard-fail); checkpoint files differ (Phase 2+,
  skips gracefully — no checkpoints exist yet).
- **Run against real data**: `hindi=130 docs, nepali=797 docs, overlap=0` — confirmed.
- Exits non-zero on any real failure, so it doubles as a pre-push guard. 6 tests isolate the
  check logic from actual repo state via a monkeypatched `REPO_ROOT`.

### 14. Corpus encoding to memmap format (commit `d22b94e`)

`lma/tokenizer/encode_corpus.py` — encodes a cleaned/split corpus (one document per line, as
`build_corpus.py` writes) into the nanoGPT-style format Phase 2's trainer will read directly:
a flat `uint16` token-id array (`.bin`) plus a `uint64` document-boundary offset array (`.idx`),
opened via `np.memmap` so a ~500M-token corpus never needs to fully load into RAM.

- `uint16` validity (vocab must stay under 65,536) is asserted explicitly at encode time, not
  left to silently wrap token ids if a future vocab choice exceeded it.
- `document_slice()` recovers a single document's token span from the offset array without
  re-tokenizing or re-scanning the original text — needed for Phase-2 generation-prefix
  evaluation.
- **Verified**: trained a small real tokenizer, encoded real text end to end, confirmed exact
  match against direct `encode()`, correct EOS-append behavior, genuine memmap (not fully
  materialized) access, and the uint16-overflow guard. 8 tests.

### 15. Google Drive upload script (commit `89b1299`)

`scripts/upload_to_drive.py` — uploads large artifacts (raw/cleaned corpus, tokenizer files,
later Phase-2/3 checkpoints) to Drive, since git must never hold them.

- Wraps `rclone` rather than implementing a Google Drive OAuth flow ourselves — rclone handles
  token refresh and resumable uploads for free, and lets whichever machine actually holds the
  real artifacts (this dev machine, or more likely Kaggle/Colab) authenticate with its own
  owner's Google account via a one-time `rclone config`, with no credentials ever living in this
  repository.
- Prints the manual shareable-link step after every successful upload (rclone can't set
  link-sharing permissions itself), so it's never forgotten before the README is updated.
- `rclone` is not installed on this dev machine — expected, since real uploads run wherever the
  real corpus/checkpoints are, not here. 5 tests cover what's testable without it: path
  resolution, error messages, shareable-link instruction text.

### 16. Two major crawler bugs found and fixed by watching a live restart (commits `e72cacd`, `0104273`)

Restarting the crawlers to pick up the seeding fix (below) surfaced a **second**, related bug
immediately — both are documented together since they were found in the same investigation.

**Bug A — restarts silently starved every domain but the first.** After ~3.5 hours and several
thousand fetched Hindi documents, the frontier was checked per-host for the first time: **100%
`www.amarujala.com`, zero URLs from any of the other 7 Hindi news/encyclopedic/government
domains.** Root cause: `run_crawl()` only called the sitemap-seeding step when
`frontier.stats().total == 0`. Once the first domain in the list contributed any URL (on the very
first run), every later restart — and there were several, while fixing the bbc.com budget and
link-following issues — saw a non-empty frontier and skipped seeding entirely, permanently
starving every domain after the first. **Fix**: a new `seeded_domains` table in the frontier
tracks completion per domain, not globally; `_seed_frontier_from_domains()` now always runs but
skips any domain already marked seeded. Same "resume, don't restart" principle the URL-level
frontier already used, applied one level up.

**Bug B — the sitemap budget reset per candidate, not per domain.** Restarting to pick up Bug A's
fix, the Hindi frontier immediately ballooned to 500,000+ URLs while still stuck seeding
`amarujala.com` alone for many minutes, blocking every other domain (and all real fetching)
behind it. Root cause: a domain's sitemap *candidates* (its robots.txt-declared sitemap(s) plus
`sitemap.py`'s 4 common-path guesses) each called `fetch_urls_from_sitemap()` with no shared
budget, so each candidate that turned out to be a large index got its **own** fresh
`max_sitemaps_total=40` allowance — true per-domain cost could be `candidates_count × 40`
instead of a bounded ~40. **Fix**: one budget list created per domain in
`_seed_frontier_from_domains()`, passed to every candidate's `fetch_urls_from_sitemap()` call.

**Confirmed converging correctly after both fixes, watched live without further interruption**:
`amarujala.com` finished seeding bounded at 79,861 URLs and the crawler moved on —
`jagran.com` (128 URLs) and `navbharattimes.indiatimes.com` (1,564 URLs) seeded next. Nepali's
frontier reached **all 10 configured domains seeded** (previously only 6 had ever been reached).

9 new tests reproduce both exact bug scenarios against `httpx.MockTransport` (no real network):
already-seeded domain makes zero requests on restart; an unseeded domain still gets seeded even
when another domain already has frontier URLs; a domain with two large-index sitemap candidates
stays bounded near the shared 40-fetch budget instead of ballooning toward 80+. 191 total tests,
1 skipped.

**Lesson for the remaining Phase-1 days**: do not kill and restart the crawlers casually anymore
— each restart before a domain finishes its (now-bounded) seeding walk re-attempts that domain's
seeding from scratch, adding real but avoidable overhead. Let them run continuously.

### 17. `claim_batch` FIFO bias fixed to round-robin across hosts (commit `b7f0316`)

Found by re-running `build_corpus.py` on the freshly-seeded Nepali frontier to check the new
domains' contribution: **1,759 of 1,766 fetched Nepali documents came from `www.bbc.com` alone**,
despite all 10 configured domains now being seeded (section 16). `claim_batch()`'s plain
`SELECT ... LIMIT n` returns rows in insertion order in practice; `bbc.com`'s sitemap bulk-seeded
4,824+ URLs in one shot while every sitemap-less domain (seeded only via link-following) produced
a handful each — insertion-order FIFO meant `bbc.com`'s huge early batch got claimed first, every
single batch, indefinitely.

**Fix**: `claim_batch` now uses a window-ranked query (`ROW_NUMBER() OVER (PARTITION BY host
ORDER BY rowid)`), taking every host's earliest-discovered pending URL first, then every host's
second, and so on — true round-robin fairness regardless of how lopsided pending counts are. This
directly serves the report's domain-distribution figure: a corpus that is 99.6% one source
undermines "diverse manually-collected corpus."

2 new tests (a dominant host does not crowd out a minor host within one batch; several small
hosts are all represented before a large host dominates). 193 total, 1 skipped.

This fix lives in code executed on every batch (not just at crawler startup like the seeding
fixes), so it required a restart to take effect — judged safe this time since the seeding logic
is now itself restart-safe and bounded (sections 16), unlike earlier restarts that interrupted an
in-progress unbounded seeding walk.

**Confirmed working, ~2 minutes after restart**: Hindi's `done` set went from 1 domain to 9
(amarujala, bhaskar, navbharattimes, livehindustan, jagran, hi.wikipedia, hi.wikisource,
hi.wiktionary, aajtak — abplive joining shortly after); Nepali's went from 1 to 9 as well.
Re-running `build_corpus.py` on this fresh data confirms real diversity, not just frontier
diversity: **Hindi 442 docs / 241,420 words across 8 sources** (amarujala 218, livehindustan 80,
navbharattimes 85, aajtak 20, bhaskar 14, hi.wikisource 15, hi.wikipedia 7, hi.wiktionary 3);
**Nepali 2,028 docs / 698,403 words across 8 sources** (bbc 1,779, ekantipur 105, onlinekhabar
70, setopati 51, wikipedia 15, nagariknews 5, himalkhabar 2, wikisource 1). bbc.com is still
Nepali's largest single source — expected, since it had a multi-hour head start before the
fairness fix landed, and round-robin fairness only affects claiming *going forward*, it does not
retroactively rebalance work already done. Its share will keep shrinking in relative terms as the
other 9 domains get an equal claim priority per round from here on.

### 18. Operational finding: background crawlers do not survive a session boundary

Checked on the crawlers after some time had passed and found **both python processes gone**,
with no crash in the logs — Nepali's log simply stops mid-stream (last line at 00:11:37, no
error, no exit message). Root cause: processes launched via `nohup ... &` from this assistant's
shell are tied to that shell/session's lifecycle in this environment; they do not reliably
survive a boundary between one turn of work and the next, even with `disown`.

**No data was lost** — this is exactly why the frontier is SQLite-backed with WAL mode and
atomic claim/mark-done transitions (section 3): every fetch that completed before the process
died is durably recorded, `in_progress=0` on reopen confirms the requeue-on-crash guarantee held,
and both crawlers resumed from their exact prior state (`done` counts unchanged, `seeded_domains`
intact) with a simple relaunch.

**This changes the operational plan for the remaining Phase-1 days.** Relying on this assistant
to keep a background crawl alive unattended for multiple days is not reliable in this
environment. Recommended going forward: run
`python scripts/run_crawl.py --lang hindi --max-docs 500000 --concurrency 24 --min-delay 1.5`
(and the `nepali` equivalent) **directly in your own terminal window** (or a `screen`/`tmux`
session, or a Kaggle/Colab notebook cell) so it runs independently of this assistant's session.
The crawler is fully resume-safe either way — restarting it from either side never loses
progress — but it needs to actually be *running* to make progress, and that now needs to happen
outside a single assistant turn.

---

## What's left to do

Ordered by what's actually blocking:

1. **Let the crawlers keep running — but see section 18: this needs to happen outside a single
   assistant session now.** They need real days of wall-clock to approach the ~500M-token
   targets. Current combined fetch rate: roughly 1,000-2,000 docs/hour across both languages,
   growing as link-following discovers more distinct hosts. Run
   `python scripts/run_crawl.py --lang hindi --max-docs 500000 --concurrency 24 --min-delay 1.5`
   (and `--lang nepali`) in your own terminal/tmux/Kaggle notebook so it survives independently.
2. **Watch for further crawl-quality issues** as new domains start contributing (each domain
   can have its own boilerplate patterns, like `amarujala.com`'s cookie notice did) — re-run
   `build_corpus.py` periodically and inspect `heuristic_rejection_counts` / dedup stats for
   anything that looks like a repeat of the hub-page problem.
3. **OCR**: needs to actually run somewhere with the `tesseract` binary (Kaggle/Colab) against
   real public-domain Hindi/Nepali books/PDFs — this machine only proved the code path is
   correct, it hasn't produced any real OCR'd text yet.
4. **HuggingFace authentication**: `huggingface-cli login` (with an account that has accepted
   OSCAR/CulturaX's terms) needs to happen on whichever machine does the real downloaded-corpus
   pull, so those two sources aren't limited to Wikipedia-only.
5. **Re-run `scripts/tokenizer_sweep.py`** once each language's train split is large enough to
   support the full {8k...48k} range — pick the final vocab size/algorithm from real fertility
   numbers, not the current placeholder.
6. ~~Encode the final corpus to `uint16` memmap `.bin`/`.idx` for fast Phase-2 training~~ —
   **done** (`lma/tokenizer/encode_corpus.py`, commit `d22b94e`); just needs to be run against
   the final chosen tokenizer once item 5 is complete.
7. **Write `report/phase1.md`** with the real final numbers, funnel charts, tokenizer sweep
   table, dataset statistics, and the manual-vs-downloaded split — once items 1-6 are far
   enough along that the numbers are representative, not a 130-document snapshot.
8. ~~Write the Google Drive upload script~~ — **done** (`scripts/upload_to_drive.py`, commit
   `89b1299`). **Still needed from you**: run `rclone config` once (see the script's docstring)
   to authorize it against your own Google account, then actually run the uploads and fill in
   the README's Drive-links table — I cannot authenticate this myself.
9. **Final `phase-1` push** before 19 Aug 2026, 11:59 PM.

---

## Change log

*(newest first; each entry = one working session's worth of changes)*

- **2026-08-14, night**: Added `scripts/upload_to_drive.py` (section 15) — rclone-based Drive
  upload for large artifacts, with the manual one-time setup and shareable-link steps documented
  since I cannot authenticate a Google account myself. 5 new tests, 182 total. Crawlers still
  running continuously (Hindi 2,217 / Nepali 1,187 docs fetched at time of this entry).

- **2026-08-14, night**: Added `lma/tokenizer/encode_corpus.py` (section 14) — nanoGPT-style
  `.bin`/`.idx` memmap corpus encoding, ready for Phase 2 once the tokenizer sweep is finalized.
  8 new tests, 177 total. Crawlers still running continuously in the background (Hindi 2,116 /
  Nepali 1,129 docs fetched at time of this entry, both still climbing).

- **2026-08-14, late night**: Found and fixed two more real crawler bugs (section 16) by
  watching a live restart: restarts silently starved every domain but the first (frontier was
  100% one domain after 3.5 hours), and the sitemap budget reset per-candidate instead of
  per-domain (frontier ballooned to 500k+ URLs stuck on one domain). Both confirmed fixed live:
  Nepali now has all 10 configured domains seeded (was 6), Hindi is climbing past its first
  domain for the first time. 9 new tests, 191 total. **Lesson recorded**: stop restarting the
  crawlers casually — let them run continuously from here.

- **2026-08-14, very late night**: Found and fixed a third crawler bug (section 17):
  `claim_batch`'s plain FIFO selection let one bulk-seeded domain (bbc.com) crowd out every
  other domain almost entirely (1,759/1,766 Nepali docs from one source). Fixed with round-robin
  claiming across hosts; confirmed working within ~2 minutes of restart — both languages now
  actively fetching from 9 distinct domains each. Re-ran `build_corpus.py` on the fresh data:
  Hindi 442 docs / 241,420 words across 8 sources, Nepali 2,028 docs / 698,403 words across 8
  sources — real, measured diversity, not just frontier diversity. 2 new tests, 193 total.

- **2026-08-14, evening**: Built the entire Phase-1 pipeline end to end (sections 1-13 above),
  launched both crawlers, fixed three real bugs found by running the code against live data
  (bbc.com sitemap-blocking, Nepali sitemap starvation, amarujala.com boilerplate flooding),
  fixed one flaky test (matplotlib backend). 169 tests passing. Created this log.
