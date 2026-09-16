# Phase 3 Log — Reasoning Finetuning, Attention Analysis, Final Report

Running development log for Phase 3 of the CL3410 individual project (Hindi = Model H, Nepali =
Model L). Same convention as `PHASE1_LOG.md` and `PHASE2_LOG.md`: a **working log, appended as
changes happen**, distinct from `report/phase3.md` (the polished, consolidated final report).

**Branch:** `phase-3` (from `phase-2`). **Deadline:** 16 Sep 2026, 11:59 PM — also the final project
deadline. **Marks:** 35.

---

## What Phase 3 has to deliver

Per language, fully independently (assignment §3):

1. A **synthetic comparative / transitive reasoning dataset**, generated programmatically in the
   language's own script and natural phrasing, with train / val / test splits, a report of size and
   template variety, and an explicit account of how train–test leakage is avoided. The generator is
   released so the dataset is reproducible.
2. **Finetuning** of that language's own pretrained checkpoint, tokenizer and vocabulary fixed, every
   hyperparameter documented, checkpoints in the Phase-2 resume-capable format. Pretrained vs
   finetuned exact match, qualitative successes and failures, and a Model H vs Model L comparison.
3. **Post-finetune attention analysis** with the Phase-2 toolkit: pretrained vs finetuned heatmaps
   for at least one early and one late layer, and whether finetuning changed local vs long-range
   attention or head specialisation, especially on comparative prompts.
4. A **final report consolidating the whole project**, answering the assignment's four questions,
   plus a README with reproduction steps and every Drive link.

The Module-5 lectures add the evaluation discipline this phase is graded against (L9, slide 34):
train on one entity pool and test on a disjoint one; write four surface forms, train on three and
hold the fourth back; report the three accuracies separately with more than one seed; and report the
pretraining validation perplexity before and after finetuning.

## Inputs inherited from Phase 2

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Checkpoint | `hindi/train/hindi_ckpt_best.pt` | `nepali/train/nepali_ckpt_best.pt` |
| Parameters | 23,846,784 | 23,846,784 |
| Tokenizer | Unigram 16,000 | Unigram 16,000 |
| Test perplexity / BPB | 31.01 / 0.4819 | 52.16 / 0.4561 |

---

## Current status snapshot

*(updated live)*

| | status |
|---|---|
| Log | done |
| Lexicons + templates (both languages) | done |
| Generator, solver, splits, leakage checks | done — 20k / 1k / 4×1k per language, all leakage counts 0 |
| SFT data, scoring, baselines | done — pretrained 0-shot and 3-shot evaluated, both languages |
| SFT loop + smoke run | done — SFT and replay smoke runs; resume verified |
| `tests/test_reasoning.py` | done — 42 tests green |
| LR sweep + main runs + size curve | done — 20 runs, both languages |
| Evaluation + error analysis | done — `<lang>/reasoning/results/summary.md` |
| Attention comparison | done — `<lang>/reasoning/results/attention_compare.json` + figures |
| `verify_independence.py` extended to finetuned checkpoints | done — 5/5 checks pass |
| Final report + README | done — `report/phase3.md`, `report/phase3.tex`, `README.md`; 500 tests pass |
| Drive upload of finetuned checkpoints + train split + eval rows | **to do (manual)** |

---

## Findings carried in from planning

**The tokenizers destroy newlines.** Neither Phase-1 tokenizer round-trips `\n` — a byte-fallback
Unigram trained on one-document-per-line text never needed it. Every reasoning prompt is therefore
single-line, with the question and answer separated by natural markers (`प्रश्न:` … `उत्तर:`), which
round-trip exactly. There is no special `<ANS>` token, and the vocabulary must stay fixed, so a
natural-language marker is the only option that respects the assignment.

**The two languages write numbers in different scripts.** Measured on the Phase-1 validation splits:
Hindi uses ASCII digits 92% of the time (33,838 vs 2,907 Devanagari), Nepali uses Devanagari digits
90% of the time (21,226 vs 2,390 ASCII). "Natural phrasing for that language" therefore means `25` in
the Hindi dataset and `२५` in the Nepali one. This is the Phase-3 counterpart of Phase 1's chandrabindu
divergence: a rule that is correct for one language is wrong for the other.

**Multi-digit numbers tokenize inconsistently** (`450` → `4` + `50`), so numeric comparison carries a
tokenization confound that transitive comparison does not. Accuracy is therefore reported per task
type rather than only in aggregate.

---

## Entries

### 2026-09-14 — dataset: facts, lexicons, templates, builder

**Facts first, words second.** `lma/reasoning/facts.py` generates each problem as structured facts
(values or pairwise relations plus a question) and `solve()` recomputes the answer from those facts
alone — for chain tasks by transitive closure over the stated relations, never from the hidden chain
order. Five task types cover every example style in §3.1: `cmp2`, `cmp3` (both with a tie → बराबर),
`ext3`, `pair3` (the multi-hop A-vs-C question), `ext4`. In `cmp3` the tie is placed *at the asked
extreme*; a tie between the other two values would not change the answer and would teach nothing.

**Spellings were chosen from the pretraining corpus, not a dictionary.** A word-frequency probe on
the Phase-1 validation splits settled several choices:
- Hindi writes ज्यादा (366) not ज़्यादा (20), वजन (39) not वज़न (0), चीज not चीज़ → templates use the
  no-nukta forms the model actually saw.
- Hindi's normaliser folds chandrabindu → anusvara, so plurals are चूड़ियां, not चूड़ियाँ.
- **Nepali's normaliser strips a word-final virama**: `norm("छन्") == "छन"`. The corpus therefore
  contains छन / छिन, never छन् / छिन्, and the Nepali templates write the halant-less forms. A
  textbook-correct template would have put spellings in front of the model that it never saw once.
- Nepali keeps chandrabindu (गह्रौँ, आँप) and uses यीमध्ये (the frequent form).

**Hindi-only constraint: no masculine noun ending in -ā.** पंखा becomes पंखे before a postposition, so
the answer would not appear in the prompt in the same form — breaking exact match and the pointer
analysis. All such objects (पंखा, जूता, छाता …) were replaced by feminine or consonant-final nouns.
Nepali has no such problem: postpositions attach to an unchanged stem (हरिभन्दा).

**Substring clashes removed** (a test asserts none remain): Hindi राधा⊂अनुराधा, नेहा⊂स्नेहा,
ललित⊂ललिता; Nepali बिमल⊂बिमला, रबिन⊂प्रबिन, रमिला⊂प्रमिला, बत्ती⊂मैनबत्ती, मनिष/मनिषा, बिजय/बिजया.
Pools: 200 persons + 60 objects per language; no person name shared across languages.

**Bug caught by the template-variety count.** The first build cycled task×attribute (period 30) and
family (period 3) independently. Since 3 divides 30, F1 was only ever rendered for *age* and *price*.
The observed-skeleton count (F1: 22 vs F4: 51) exposed it; the builder now cycles the full
task × attribute × family product and a test asserts full coverage (5 × 6 × 3 = 90 combinations).

**Heuristic baselines read the premises only.** Computed over the whole prompt, "last-mentioned"
scored 0.51 on `pair3` — only because the question names the two options last, which is a coin flip
between the two right candidates rather than a leak. Over premises only, every heuristic sits at or
below chance (0.30 overall; `cmp2` first/last ≈ 0.38 = 0.75 non-tie × 0.5).

**Final datasets** (seed 1234): leakage counts all 0 (prompt, fact-level, pool, family); tokenizer
round-trip failures 0; prompt length median 32–35 tokens, max 56. Observed sentence templates
per family: Hindi F1 50 / F2 43 / F3 39 / F4 51; Nepali 53 / 42 / 39 / 54. Full tables in
`<lang>/reasoning/results/dataset_stats.md`.

### 2026-09-14 — SFT machinery, baselines, smoke runs

- Sequence format `[EOS] प्रश्न: … उत्तर: ▁answer [EOS]`, loss on answer + EOS only. The leading EOS
  reproduces pretraining's document-start context; prompt and answer are encoded separately so every
  candidate shares identical prompt tokens.
- Generation batches by exact prompt length (no KV cache / padding mask in `GPT.generate`).
- **Pretrained baselines reproduce L9 slide 4:** exact match 0.000 on every split for both models,
  zero- and three-shot. Candidate-choice accuracy is 0.22–0.30, at or *below* chance (0.30) — the
  pretrained models do not rank the right entity higher even when format is taken out of the
  equation. Three-shot raises the stop rate (Nepali 1% → ~20%) but not accuracy: the demonstrations
  steer the format, not the comparison (L8).
- Phase-2 val perplexity of the pretrained checkpoints, recomputed by the Phase-3 evaluator: H 30.34,
  L 51.23 — the "before" numbers for forgetting.
- Smoke run (40 steps): SFT+replay val EM 0.00 → 0.27; resume continues step 40 → 50 in the same
  append-only log.
- `verify_independence.py` gained a Phase-3 check: every finetuned checkpoint's init checkpoint and
  tokenizer are its own language's; embedding cosine to its own pretrained model ≈ 0.98, to the
  other language's ≈ 0.00.

### 2026-09-14 — Model H: LR sweep and main grid

**Sweep (seed 0, scored on val only):** 5e-5 → 0.368, **1e-4 → 0.447**, 3e-4 → 0.362. 1e-4 selected;
only then were the test splits evaluated. Each full SFT run is ~938 steps / ~3 min on the RTX 3050.

**Main grid, mean ± std over 3 seeds (exact match):**

| | val | test_id | test_entity | test_wording | test_both | Phase-2 val PPL |
|---|---|---|---|---|---|---|
| pretrained 0-shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 30.34 |
| SFT | 0.408 ± 0.038 | 0.388 ± 0.049 | 0.136 ± 0.043 | 0.331 ± 0.056 | 0.102 ± 0.020 | 142.4 ± 85.2 |
| SFT + replay | 0.434 ± 0.029 | 0.422 ± 0.045 | 0.129 ± 0.009 | 0.397 ± 0.036 | 0.102 ± 0.005 | 35.5 ± 0.1 |

Three findings, each one the design was built to detect:

1. **Unseen entities, not unseen wording, is what breaks.** Held-out phrasing F4 costs 3–6 points;
   unseen names cost ~28. On `test_both`, ~560–600 of ~900 errors per seed are *out-of-prompt
   names*: the model answers with a training-pool name that is not in the question. This is L9 slide
   24's failure (0.27 train loss, 0.000 on unseen names), reproduced with a 140-name pool rather
   than 20 — a larger pool did not by itself stop the model binding answers to name identity.
2. **On seen entities the errors are about direction, not format.** On `test_id` the stop rate is
   100%, out-of-prompt names ~0, and the dominant error is *reversed extreme* (~350 / ~600): the model
   picks one of the two extreme entities but not reliably the one the polarity word asks for.
   Candidate-choice accuracy tracks EM closely, so the gap is the comparison, not the output format —
   consistent with L10: format is learnt reliably, the reasoning much less.
3. **Replay removes forgetting at no cost to the task.** Plain SFT raises Phase-2 validation
   perplexity from 30.3 to 142 ± 85 (one seed 189 — L9's 1.33 → 194.7 pattern); with 8 × 256
   pretraining tokens mixed into each step it stays at 35.5 ± 0.1, while EM is equal or better on
   every split and the seed variance shrinks.

Data-size curve so far: 1,000 examples → val EM 0.297 (chance 0.300) with 100% stop rate — the
format is learnt from 1k examples, the task is not.

### 2026-09-15 — Model H: data size, attention

**Memorisation grows with data** (EM):

| | test_id | test_entity | test_both | out-of-prompt errors (test_both) | PPL |
|---|---|---|---|---|---|
| SFT 1k | 0.296 | 0.227 | 0.208 | 151 | – |
| SFT 5k | 0.330 | 0.158 | 0.109 | 523 | – |
| SFT 20k (3 seeds) | 0.388 | 0.136 | 0.102 | 560 | 142 |

- Going from 1k to 20k examples raises seen-entity EM (0.30 → 0.39) and *lowers* unseen-entity EM
  (0.23 → 0.14): extra data is spent binding answers to training names.
- Candidate choice on `test_both` is 0.32–0.34 for SFT and replay (chance 0.30): neither learned
  the comparison for names it had not seen.

**Attention (SFT s0 vs pretrained, 178 entity-answer test_id prompts):**
- **Answer-pointer heads appear in the last two layers.** Pretrained, no head puts more attention on
  the correct entity than on distractors (max score 0.001). After SFT, every layer-9 head gains
  (+0.08 to +0.12; L9H5 and L9H1 largest) plus L8H0 and L8H3; layers 0–7 are unchanged or slightly
  negative (L7H4 −0.13). Finetuning *repurposed the top of the network* rather than changing its
  lower layers.
- **Global attention statistics barely move.** Layer-mean normalised entropy changes by at most 0.09
  (layer 4, reasoning prompts: 0.771 → 0.863) and by ≤0.06 on ordinary text; mean distance on text changes by ≤6 tokens, largest at
  layer 9. Role counts: on reasoning prompts diffuse-global heads 38 → 43, mixed 21 → 17. The
  Phase-2 "layer 6 focusing layer" (lowest mean distance on text, 17.5 tokens) survives: 19.1 after
  SFT.
- Figure fixes: the SentencePiece `▁` and the Devanagari answer in the suptitle rendered as boxes;
  labels now use `_` and the suptitle uses the Devanagari font. Figures regenerated after the runs.

### 2026-09-15 — Model L: LR sweep diverges from Hindi

Nepali sweep (val EM): 5e-5 → 0.339, 1e-4 → 0.356, **3e-4 → 0.370** — the highest LR wins, where
Hindi's peaked at 1e-4. Selected by the same val-only rule. Plain SFT at 3e-4 is catastrophic for
language modelling: Phase-2 val perplexity 51.2 → 97,249 (s0) and 724,433 (s1) — the model stops
being a Nepali LM at all while gaining ~2 points of val EM. The replay condition at the same LR is
the controlled test of whether mixing pretraining text prevents that.

### 2026-09-15 — Model L: results, and the cross-language picture

| Model L, 3 seeds (EM) | val | test_id | test_entity | test_wording | test_both | val PPL |
|---|---|---|---|---|---|---|
| pretrained 0-shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 51.23 |
| SFT (3e-4) | 0.368 ± 0.003 | 0.353 ± 0.007 | 0.103 ± 0.002 | 0.322 ± 0.002 | 0.085 ± 0.003 | 418,137 ± 313,846 |
| SFT + replay (3e-4) | 0.371 ± 0.005 | 0.368 ± 0.011 | 0.134 ± 0.004 | 0.352 ± 0.011 | 0.107 ± 0.006 | 76.32 ± 0.19 |

Data-size curve (seed 0): 1k examples test_entity 0.154, 5k 0.108.

Every Hindi finding replicates in Nepali: format learnt perfectly, seen-entity accuracy modest,
unseen entities collapse through out-of-prompt names, replay fixes forgetting, more data widens the
seen/unseen gap. Differences:
- Model H is ~5 points better on seen entities and wording; the two are equal on unseen entities.
- 170/200 Hindi names are single tokens vs 116/200 Nepali names (mean 1.18 vs 1.54 tokens); Nepali's
  out-of-prompt errors include prefix-completions (नबिन → नविना).
- Attention: pointer heads appear in L9H0/L9H2 (+0.15/+0.14) and head 5 of layers 7–9; Model L's
  attention shifted more overall (layer 5 on text 32.0 → 19.8 tokens; layers 7–9 on reasoning prompts
  entropy 0.72/0.61/0.69 → 0.44/0.52/0.49), consistent with its much larger LM damage. The layer-6
  focusing layer survives in both (H 17.5 → 19.1, L 17.4 → 16.0).

### 2026-09-15 — wrap-up

- `scripts/phase3_compare.py` → `report/phase3_comparison.md` + `report/figures/phase3_h_vs_l_em.png`.
- `verify_independence.py`: 10 + 10 finetuned checkpoints, own-language lineage, embedding cosine to
  own pretrained ≥ 0.875, to the other ≤ 0.0046, no identical tensors across languages.
- Hindi attention figures regenerated after the label/suptitle font fix.
- Full suite: 496 passed.
- Report numbers were re-checked against the result files before finalising. Corrected: the
  out-of-prompt share (56–67%, not 55–65%); the unseen-entity candidate-choice range (0.27–0.34 on
  `test_both`); the location of Model L's pointer gains (layers 7–9, not 8–9); and the runtime.

### 2026-09-16 — attention analysis moved to the recommended model (SFT + replay)

The attention comparison was first run on plain SFT seed 0. It now uses **SFT + replay seed 0**
(H `replay_lr0.0001_s0`, L `replay_lr0.0003_s0`), the model the report recommends. The switch
matters: Model L's plain-SFT checkpoint has perplexity 418k, so its attention changes mixed task
learning with forgetting. `run_phase3.py` and the `attention_compare.py` default now use the replay
run, so the pipeline reproduces what the report shows.

What changed in the findings:
- **Answer-pointer heads: same conclusion.** H best head L9H5 0.091 (gains L9H1 +0.117, L9H5 +0.114,
  L9H4 +0.109, L8H3 +0.094); L best head L9H0 0.101 (L9H2 +0.124, L9H0 +0.113). Largest gain in any
  head of layers 0–6: H +0.008, L +0.035.
- **Global attention on text: now essentially unchanged.** Mean distance within 4.1 tokens and
  normalised entropy within 0.04 at every layer, in both models. Plain SFT's Model-L shifts (layer 5
  on text 32.0 → 19.8 tokens; layer 7 text entropy 0.603 → 0.448) do not appear with replay, so they
  were forgetting, not task learning.
- **Task-specific focusing only on reasoning prompts in Model L:** layers 7–9 entropy 0.716 / 0.612 /
  0.687 → 0.546 / 0.537 / 0.587, while the same layers on text barely move.
- Layer-6 focusing layer: H 17.5 → 16.6, L 17.4 → 18.1.
- Concrete example: Model L L9H0 attends from `उत्तर:` to the first token of the correct answer
  सुनैना after finetuning.

Report §9 (md and tex), the §10.4 evidence row, the headline and the limitations were updated.
