[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Monolingual Transformer LMs — Hindi (Model H) & Nepali (Model L)

CL3410 *Language Models and Agents*, Monsoon 2026 — Individual Project.

Two **fully independent** ~25M-parameter decoder-only Transformer language models, built,
pretrained, and analyzed from scratch in PyTorch:

| | Model H | Model L |
|---|---|---|
| Language | Hindi (`hi`) | Nepali (`ne`) |
| Tier | Higher-resource | Lower-resource (allowed list) |
| Script | Devanagari | Devanagari |
| Target pretraining tokens | ~500M | ~500M |
| **Actual pretraining tokens** | **517,640,653** | **572,506,257** |
| Manually collected share | 22.5% | 24.7% |
| Tokenizer | Unigram, 16k vocab | Unigram, 16k vocab |

**Independence statement.** Model H and Model L share **no** data, tokenizer, vocabulary, or
weights. They do share *code* — a common library (`lma/`) implementing the crawler, cleaning
pipeline, tokenizer trainer, Transformer architecture, trainer, and evaluation/attention tooling —
but every *artifact* (corpus shard, tokenizer model, vocabulary, checkpoint) is per-language and
lives under `hindi/` or `nepali/` respectively. This is mechanically verified by
[`scripts/verify_independence.py`](scripts/verify_independence.py) (document-hash intersection,
vocabulary-set overlap, tokenizer-file hashes, checkpoint embedding-dim divergence); its output is
included in `report/`.


## Phase 1 result

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Documents (train / val / test) | 3,577,045 / 5,000 / 5,000 | 5,526,928 / 5,000 / 5,000 |
| Tokens | 517,640,653 | 572,506,257 |
| Manual / downloaded words | 86.8M / 322.2M | 102.3M / 312.0M |
| Manual share | 22.5% (tokens) | 24.7% (tokens) |
| Tokenizer | Unigram 16k, fertility 1.273 | Unigram 16k, fertility 1.380 |

Both corpora were built by an independent per-language implementation of Yi's 14-step cleaning
cascade (`hindi/clean/`, `nepali/clean/`), with every threshold calibrated on that language's own
data — the two disagree about language-ID confidence, quality cutoff and Devanagari normalisation,
each with the measurement recorded in `clean/config.py`. Full numbers in
[`report/phase1.md`](report/phase1.md); the funnel and figures in [`result.md`](result.md) and
`report/figures/`.

## Repository layout

```
README.md       reproduction steps + Google Drive links + independence statement
lma/            shared code library (crawl, text-cleaning, dedup, tokenizer, model, train, eval, viz)
hindi/          Model H (Hindi) -- self-contained artifacts, each subdir has its own README:
  clean/          the language's own cleaning cascade (config, words, norm, lang, rules,
                  repair, score, dedup, safety, run) -- thresholds calibrated per language
  data/           seed_domains.py, splits/ (gitignored, on Drive), encoded/ (.bin/.idx), stats.json
  tokenizer/      SentencePiece model + vocab + sweep_results.json
  model/          per-language model config and exact parameter count      (Phase 2)
  train/          pretraining entry point, logs, loss curves               (Phase 2)
  eval/           PPL/BPB, BLEU/chrF/ROUGE-L, attention analysis           (Phase 2)
  reasoning/      synthetic reasoning data generator + finetuning          (Phase 3)
  configs/        run configuration files
nepali/         Model L (Nepali) -- identical structure, entirely separate artifacts
scripts/        run_crawl, run_wiki, fetch_downloaded, build_corpus, tokenizer_sweep,
                encode_splits, summarize_runs, verify_independence, reproduce.sh
tests/          pytest suite (filters, dedup, tokenizer roundtrip, crawler resume, ...)
report/         phase1.md / phase2.md / phase3.md + figures/ + tables/  (everything graded lives here)
```

The two language directories share *code* (`lma/`) but no *artifacts*: separate corpora,
tokenizers, vocabularies, and (from Phase 2) weights. `scripts/verify_independence.py` checks
this mechanically.

## Reproduction steps

```bash
python -m venv .venv && source .venv/bin/activate      # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .                                        # editable install of the `lma` package
pytest tests/ -q                                         # compliance suite
bash scripts/reproduce.sh                                 # small-sample end-to-end pipeline (no GPU needed)
```

Full pretraining runs on Kaggle (T4×2 / P100); notebooks in `notebooks/` are self-contained and
resume from the Drive-hosted checkpoints below.

## Dataset storage

The corpus artifacts are too large for git (19.9 GB essential, 25.7 GB with the regenerable
intermediates), so they are hosted in two places:

- **Google Drive** — [https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing](https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing)
- **HuggingFace** — [`YogendraPatel/LMA_Individual_Project`](https://huggingface.co/datasets/YogendraPatel/LMA_Individual_Project)

| Artifact | Language | Size | Where |
|---|---|---:|---|
| Raw corpus shards | Hindi | 3.1 GB | Drive / HF |
| Raw corpus shards | Nepali | 3.8 GB | Drive / HF |
| Cleaned splits (train/val/test) | Hindi | 5.5 GB | Drive / HF |
| Cleaned splits (train/val/test) | Nepali | 7.5 GB | Drive / HF |
| Encoded `.bin`/`.idx` | Hindi | 1.0 GB | Drive / HF |
| Encoded `.bin`/`.idx` | Nepali | 1.2 GB | Drive / HF |
| Tokenizer model + vocab | both | 1.4 MB | **committed** in `<lang>/tokenizer/` |
| Corpus statistics | both | 8 KB | **committed**: `<lang>/data/stats.json`, `<lang>/data/encoded/encoding.json`, `result.md` |
| Pretrained checkpoint (best + last) | Hindi (Model H) | — | _pending — Phase 2_ |
| Pretrained checkpoint (best + last) | Nepali (Model L) | — | _pending — Phase 2_ |
| Finetuned (reasoning) checkpoint | both | — | _pending — Phase 3_ |

## Status

- [x] Phase 1 — Data Collection and Tokenizer Construction (due 19 Aug 2026) — corpora built,
      cleaned, deduplicated, decontaminated, tokenized and encoded; see
      [`result.md`](result.md) and [`report/phase1.md`](report/phase1.md). Remaining: Drive upload.
- [ ] Phase 2 — Model Implementation, Pretraining, Evaluation (due 5 Sep 2026)
- [ ] Phase 3 — Reasoning Finetuning, Attention Analysis, Final Report (due 16 Sep 2026)

## Development log

[`PHASE1_LOG.md`](PHASE1_LOG.md) is a running, commit-by-commit development log for Phase 1 —
what was built and why, every bug found by running the pipeline against live data (and how it
was fixed), current corpus/frontier numbers, and an explicit "what's left" list. It is a working
log, not the graded deliverable; [`report/phase1.md`](report/phase1.md) is the polished one.

[`result.md`](result.md) is a generated side-by-side summary of both languages' cleaning runs
(funnel, thresholds, dedup counts, decontamination, final sizes). Regenerate it after any run with
`python scripts/summarize_runs.py` rather than editing it by hand.
