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
| Target pretraining tokens | ~500M | ~500M (see Phase-1 report for actual — likely short, and reported as such) |

**Independence statement.** Model H and Model L share **no** data, tokenizer, vocabulary, or
weights. They do share *code* — a common library (`lma/`) implementing the crawler, cleaning
pipeline, tokenizer trainer, Transformer architecture, trainer, and evaluation/attention tooling —
but every *artifact* (corpus shard, tokenizer model, vocabulary, checkpoint) is per-language and
lives under `hindi/` or `nepali/` respectively. This is mechanically verified by
[`scripts/verify_independence.py`](scripts/verify_independence.py) (document-hash intersection,
vocabulary-set overlap, tokenizer-file hashes, checkpoint embedding-dim divergence); its output is
included in `report/`.

## Repository layout

```
README.md       reproduction steps + Google Drive links + independence statement
lma/            shared code library (crawl, text-cleaning, dedup, tokenizer, model, train, eval, viz)
hindi/          Model H (Hindi) -- self-contained artifacts, each subdir has its own README:
  data/           seed_domains.py, splits/ (train/val/test + manifests), stats.json
  tokenizer/      SentencePiece model + vocab + sweep_results.json
  model/          per-language model config and exact parameter count      (Phase 2)
  train/          pretraining entry point, logs, loss curves               (Phase 2)
  eval/           PPL/BPB, BLEU/chrF/ROUGE-L, attention analysis           (Phase 2)
  reasoning/      synthetic reasoning data generator + finetuning          (Phase 3)
  configs/        run configuration files
nepali/         Model L (Nepali) -- identical structure, entirely separate artifacts
scripts/        run_crawl, run_wiki, build_corpus, tokenizer_sweep, verify_independence, reproduce.sh
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

## Google Drive links

*(populated as each artifact is produced — see the corresponding phase report for the exact
upload date and file manifest)*

| Artifact | Language | Link |
|---|---|---|
| Raw + cleaned corpus shards | Hindi | _pending — Phase 1_ |
| Raw + cleaned corpus shards | Nepali | _pending — Phase 1_ |
| Tokenizer model + vocab | Hindi | committed in `hindi/tokenizer/` (small; also see report) |
| Tokenizer model + vocab | Nepali | committed in `nepali/tokenizer/` (small; also see report) |
| Pretrained checkpoint (best + last) | Hindi (Model H) | _pending — Phase 2_ |
| Pretrained checkpoint (best + last) | Nepali (Model L) | _pending — Phase 2_ |
| Training logs (JSONL) | both | committed in `hindi/train/logs/`, `nepali/train/logs/` |
| Finetuned (reasoning) checkpoint | Hindi | _pending — Phase 3_ |
| Finetuned (reasoning) checkpoint | Nepali | _pending — Phase 3_ |

## Status

- [ ] Phase 1 — Data Collection and Tokenizer Construction (due 19 Aug 2026)
- [ ] Phase 2 — Model Implementation, Pretraining, Evaluation (due 5 Sep 2026)
- [ ] Phase 3 — Reasoning Finetuning, Attention Analysis, Final Report (due 16 Sep 2026)

## Development log

[`PHASE1_LOG.md`](PHASE1_LOG.md) is a running, commit-by-commit development log for Phase 1 —
what was built and why, every bug found by running the pipeline against live data (and how it
was fixed), current corpus/frontier numbers, and an explicit "what's left" list. It is a working
log, not the graded deliverable: the polished `report/phase1.md` (tables, figures, written
analysis) will be written once the corpus reaches representative scale, close to the deadline.
