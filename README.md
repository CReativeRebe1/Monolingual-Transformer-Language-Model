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

## Phase 2 result

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Parameters | 23,846,784 | 23,846,784 |
| Pretraining tokens (1 epoch) | 515,899,392 | 571,211,776 |
| Test perplexity | 31.01 | 52.16 |
| **Test bits-per-byte** | 0.4819 | **0.4561** |

Full report: [`report/phase2.md`](report/phase2.md).

## Phase 3 result

Each model finetuned on a synthetic comparative/transitive reasoning dataset in its own language
(20k train; test splits cross seen/unseen entities × seen/held-out wording). Exact match, SFT with
pretraining replay, mean ± std over 3 seeds:

| | pretrained | test_id | test_entity | test_wording | test_both | val PPL before → after |
|---|---:|---:|---:|---:|---:|---:|
| Model H | 0.000 | 0.422 ± 0.045 | 0.129 ± 0.009 | 0.397 ± 0.036 | 0.102 ± 0.005 | 30.3 → 35.5 |
| Model L | 0.000 | 0.368 ± 0.011 | 0.134 ± 0.004 | 0.352 ± 0.011 | 0.107 ± 0.006 | 51.2 → 76.3 |

Finetuning teaches the format and generalises across wording but not across entity names (the model
answers with memorised training names). Replay prevents catastrophic forgetting (plain SFT: H 142,
L 418,137).
Answer-pointer heads appear in the last two layers. Full report with the four project questions
answered: [`report/phase3.md`](report/phase3.md).

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

**Phase 1 — corpora and tokenizers.** `scripts/run_crawl.py`, `scripts/fetch_downloaded.py` and
`scripts/build_corpus.py --lang <lang>` build the cleaned splits; `scripts/tokenizer_sweep.py` and
`scripts/encode_splits.py` produce the tokenizer and the encoded `.bin/.idx`. Or download the
finished artifacts from Drive (below).

**Phase 2 — pretraining and evaluation.** Full runs used Kaggle (2×T4); the notebook cells are in
`notebooks/phase2_kaggle.md`, and every run resumes from `ckpt_last.pt`.

```bash
torchrun --standalone --nproc_per_node=2 hindi/train/pretrain.py      # or nepali/train/pretrain.py
python hindi/eval/evaluate.py --checkpoint hindi/train/hindi_ckpt_best.pt --split test
python hindi/eval/attention_report.py --checkpoint hindi/train/hindi_ckpt_best.pt
```

**Phase 3 — reasoning finetuning** (one 4 GB GPU is enough; ~50 minutes per language). Place the
pretrained checkpoints from Drive at `hindi/train/hindi_ckpt_best.pt` and
`nepali/train/nepali_ckpt_best.pt`, and the encoded splits at `<lang>/data/encoded/` (replay and
perplexity read them).

```bash
python hindi/reasoning/generate.py                 # dataset, byte-reproducible (also run by the next line)
python scripts/run_phase3.py --lang hindi          # baselines, LR sweep, 3 seeds × {SFT, replay},
python scripts/run_phase3.py --lang nepali         #   data-size curve, summary, attention
python scripts/phase3_compare.py                   # Model H vs Model L table + figure
python scripts/verify_independence.py              # includes finetuned-checkpoint lineage
```

Individual steps: `<lang>/reasoning/finetune.py --condition {sft,replay}`,
`<lang>/reasoning/evaluate.py --run <run> | --pretrained --shots N | --summarize`,
`<lang>/reasoning/attention_compare.py --run <run>`. See `<lang>/reasoning/README.md`.

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
| Pretrained checkpoint `hindi_ckpt_best.pt` | Hindi (Model H) | 286 MB | [Drive](https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing) |
| Pretrained checkpoint `nepali_ckpt_best.pt` | Nepali (Model L) | 286 MB | [Drive](https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing) |
| Finetuned checkpoints `reasoning/checkpoints/<run>/ckpt_best.pt` (10 runs) | Hindi (Model H) | 286 MB each (weights + optimizer, resume-capable); 2.9 GB for all 10 | [Drive](https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing) — recommended: `replay_lr0.0001_s0` |
| Finetuned checkpoints `reasoning/checkpoints/<run>/ckpt_best.pt` (10 runs) | Nepali (Model L) | 286 MB each (weights + optimizer, resume-capable); 2.9 GB for all 10 | [Drive](https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing) — recommended: `replay_lr0.0003_s0` |
| Reasoning train split `reasoning/data/train.jsonl` + per-example eval rows | both | 19 MB + 19 MB per language | [Drive](https://drive.google.com/drive/folders/1lkP2GJKqgr-uzzz-WLeChOM_flqOe8xQ?usp=sharing) (train regenerable byte-for-byte by `generate.py`) |
| Reasoning val + test splits, dataset stats, logs, eval summaries | both | ~6 MB | **committed** in `<lang>/reasoning/` |

## Status

- [x] Phase 1 — Data Collection and Tokenizer Construction (due 19 Aug 2026) — corpora built,
      cleaned, deduplicated, decontaminated, tokenized and encoded; see
      [`result.md`](result.md) and [`report/phase1.md`](report/phase1.md). Remaining: Drive upload.
- [x] Phase 2 — Model Implementation, Pretraining, Evaluation (due 5 Sep 2026) — see
      [`report/phase2.md`](report/phase2.md).
- [x] Phase 3 — Reasoning Finetuning, Attention Analysis, Final Report (due 16 Sep 2026) — see
      [`report/phase3.md`](report/phase3.md) and [`PHASE3_LOG.md`](PHASE3_LOG.md).

## Development log

[`PHASE1_LOG.md`](PHASE1_LOG.md) is a running, commit-by-commit development log for Phase 1 —
what was built and why, every bug found by running the pipeline against live data (and how it
was fixed), current corpus/frontier numbers, and an explicit "what's left" list. It is a working
log, not the graded deliverable; [`report/phase1.md`](report/phase1.md) is the polished one.

[`result.md`](result.md) is a generated side-by-side summary of both languages' cleaning runs
(funnel, thresholds, dedup counts, decontamination, final sizes). Regenerate it after any run with
`python scripts/summarize_runs.py` rather than editing it by hand.
