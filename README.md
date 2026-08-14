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
lma/            shared code library (crawl, text-cleaning, dedup, tokenizer, model, train, eval, viz)
hindi/          Model H artifacts: configs, data manifests, tokenizer, training/eval entry points
nepali/         Model L artifacts: same structure, entirely separate
notebooks/      Kaggle/Colab-runnable notebooks (data → tokenizer → pretrain → eval → finetune)
scripts/        reproduce.sh, verify_independence.py, upload_to_drive.py
tests/          pytest compliance suite (masking, checkpoint resume, tokenizer roundtrip, dedup, ...)
report/         phase1.md / phase2.md / phase3.md + figures/ + tables/  (everything graded lives here)
```

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
