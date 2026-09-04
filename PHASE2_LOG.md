# Phase 2 Log — Model Implementation, Pretraining, Evaluation

Running development log for Phase 2 of the CL3410 individual project (Hindi = Model H, Nepali =
Model L). Same convention as `PHASE1_LOG.md`: this is a **working log, appended as changes happen**,
distinct from `report/phase2.md` (the polished final report). Every entry states **what was built**,
**why**, **how it was verified**, and — once pushed — **which commit** it lives in.

**Branch:** `phase-2`. **Deadline:** 5 Sep 2026. **Marks:** 40.

---

## What Phase 2 has to deliver

Per language, fully independently:

1. A ~25M-parameter decoder-only Transformer **implemented from scratch** in PyTorch (no
   `nn.Transformer`, no HuggingFace `modeling_*`, no pretrained weights).
2. Pretraining on that language's own corpus, with logged loss curves and resumable checkpoints.
3. Evaluation: perplexity, bits-per-byte, generation metrics, diversity diagnostics — plus the
   attention diagnostics that Phase 3 will report on.

## Inputs inherited from Phase 1

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Tokenizer | `hindi/tokenizer/hindi_unigram_16000.model` | `nepali/tokenizer/nepali_unigram_16000.model` |
| Vocabulary | 16,000 (Unigram, `byte_fallback=True`) | 16,000 (Unigram, `byte_fallback=True`) |
| Train tokens | 515,972,223 (3,577,045 docs) | 571,227,702 (5,526,928 docs) |
| Val tokens | 821,762 (5,000 docs) | 634,788 (5,000 docs) |
| Test tokens | 846,668 (5,000 docs) | 643,767 (5,000 docs) |
| **Total** | **517,640,653** | **572,506,257** |
| Encoded format | `uint16` `.bin` + `uint64` cumulative `.idx` | same |

Both corpora are already decontaminated against their own held-out splits (Phase 1 §, held-out-side
direction), so val/test perplexity is measured on text the model has genuinely never seen.

---

## Current status snapshot

*(updated live)*

| | status |
|---|---|
| Step 0 — this log | done |
| Step 1 — model (`lma/model/`) | **done** — 23,846,784 params, 29 tests |
| Step 2 — data loader | **done** — rank-disjoint shuffled windows |
| Step 3 — checkpoint + append-only logging | **done** — atomic writes, 3-session resume verified |
| Step 4 — training loop | **done** — fp16 + GradScaler, DDP, grad accumulation |
| Step 5 — local smoke run | **done** — both languages, resume verified live |
| Step 6 — Kaggle 2×T4 DDP run | **done** — both models trained to 1.000 epochs |
| Step 7 — evaluation | **done** — run on both checkpoints |
| Step 7b — attention diagnostics | **done** — run on both checkpoints |
| Step 8 — independence check | **done** — passes on the real checkpoints |
| Step 8 — report | not started |

Test suite: **451 passed** (354 carried in from Phase 1, 97 new).

**Everything except the actual training run is built and verified.** The remaining work is: run
Kaggle for each language (~3 h each), pull the logs back, then write the report against real numbers.

### Measured so far

| | value | where from |
|---|---|---|
| Parameters | 23,846,784 (17,702,784 non-embedding) | `analytic_param_count`, verified against a built model |
| Embedding share | 25.8% | 16,000 × 384, tied |
| Checkpoint size | 286 MB | weights + AdamW moments, measured |
| FLOPs/token | 1.53e8 | `GPT.flops_per_token()` |
| Tokens/optimizer step | 131,072 | 16 × 1024 × 4 × 2 GPUs |
| Derived step count | 3,936 (Hi) / 4,358 (Ne) | one epoch at 131,072 tok/step |

---

## Decisions taken before any code was written

| Decision | Choice | Why |
|---|---|---|
| Parallelism | DDP, 2×T4, one language at a time | Kaggle's free tier gives two T4s in one session. DDP over `DataParallel` because DDP runs one process per GPU with no Python-side gather, which is the only version that actually scales. |
| Architecture | `d_model=384`, 10 layers, 6 heads, ctx 1024 | Deeper-and-narrower beats wider-and-shallower at this scale, and it keeps the tied embedding table to 25.8% of the budget instead of 33%. |
| Positional encoding | RoPE | Zero parameters. A learned table at 1024×384 would cost 393,216 params — 1.6% of the budget spent on something rotary embeddings do for free — and would hard-cap the context at training length. |
| Precision | fp16 autocast + `GradScaler` | T4 is Turing (sm75): fp16 tensor cores, **no bf16**. bf16 on sm75 either errors or silently emulates. |
| Attention | dual path — SDPA for training, explicit `softmax(QK^T/√d_k + M)V` for analysis | The explicit path is what makes attention weights extractable at all; the fast path is what makes training finish. A test asserts they agree to `atol=1e-5`, because otherwise every attention figure would describe a model subtly different from the one trained. |
| Attention analysis | built in Phase 2 | The assignment schedules it for Phase 3, but the machinery belongs next to the model. Phase 3 then becomes reporting rather than engineering. |

### Architecture, exactly

```
vocab                16,000   (per language; tied input/output embedding)
d_model                 384
n_layers                 10
n_heads                   6   (head_dim 64)
mlp hidden            1,024   (SwiGLU: 3 matrices, so hidden is 8/3·d_model rounded, not 4·d_model)
context               1,024
norm             pre-LN RMSNorm
activation           SwiGLU
```

| component | count | share |
|---|---:|---:|
| Token embedding (16,000 × 384, tied to `lm_head`) | 6,144,000 | 25.8% |
| Attention per layer (fused QKV + output proj, 4·d²) | 589,824 | |
| SwiGLU per layer (gate + up + down, 3·d·h) | 1,179,648 | |
| RMSNorm gains per layer (2 × 384) | 768 | |
| **Per layer** | **1,770,240** | |
| All 10 layers | 17,702,400 | 74.2% |
| Final norm | 384 | |
| **Total** | **23,846,784** | 100% |

The SwiGLU width is the number worth checking. Writing `mlp_hidden = 4 * d_model = 1536` — the
plain-MLP convention — gives 2,360,064 parameters per layer and **29,745,024** in total, 19% over
budget. A gated FFN has three matrices where a plain one has two, so the width that preserves the
same parameter count is (8/3)·d_model = 1024, not 4·d_model. Both language configs assert the total
lands in 23–25M at import so this cannot be broken silently.

---

## Entries

### 1. Model architecture (`lma/model/`)

`rope.py`, `attention.py`, `block.py`, `transformer.py` — RMSNorm, RoPE, causal self-attention,
SwiGLU, tied embeddings, GPT-2 residual init scaling. Built from `nn.Linear`/`nn.Embedding` and
explicit tensor maths; no `nn.Transformer`, no `nn.MultiheadAttention`, no pretrained anything.

**Attention is written twice on purpose.** `forward(output_attentions=False)` routes to
`F.scaled_dot_product_attention`, which never materialises the `[B,H,T,T]` score matrix — at B=16,
H=6, T=1024 that matrix alone is 1.5 GB per layer in fp32 and would not fit on a T4 beside the model
and optimizer. `forward(output_attentions=True)` runs the explicit `softmax(QK^T/√d_k + M)V` and
returns `A`, which is the only way to get attention weights out at all. **A test asserts the two
agree to `atol=1e-5`**; without it a divergence would mean every attention figure in the report
describes a model that was never trained, and the heatmaps would still look like plausible attention.

Verified: causality tested *behaviourally* (change a later token, assert an earlier token's logits
do not move — stronger than inspecting the mask tensor); RoPE's dot product shown to depend only on
relative distance; rotation shown to preserve vector norm; untrained loss lands at ln(V).

**Finding, not a bug — weight tying makes the identity mapping partly free at initialisation.** A
first version of the untrained-loss test used `targets=idx`, i.e. predict the *current* token, and
measured 3.98 against the expected ln(97)=4.57. The cause is real: with a tied head, the residual
stream at position *i* still carries `emb(idx[i])`, so `logits[i, idx[i]]` is that vector's dot
product with itself. Worth knowing when reading the first steps of a real loss curve. Both
behaviours are now asserted (`test_weight_tying_makes_the_identity_mapping_partly_free_at_init`).

### 2. Data loader (`lma/train/data.py`)

Memmapped `uint16` stream, windows drawn from a **shuffled permutation** rather than sampled with
replacement — sampling 504k windows uniformly from 504k covers only ~63% of the corpus, so a
nominal "one epoch" would never see a third of the data.

**Ranks are interleaved, not split into contiguous halves.** The splits are written manual-first,
downloaded-second (the same ordering that forced stride sampling for tokenizer training in Phase 1).
Giving rank 0 the first half of the stream would train it almost entirely on manual crawl and rank 1
almost entirely on downloaded text; DDP averages the gradients so the model would survive, but the
per-rank loss curves would diverge for reasons unrelated to optimisation. `test_rank_shards_are_
interleaved_not_contiguous` asserts each rank's draw is spread across the whole stream.

### 3. Checkpointing and append-only logging (`checkpoint.py`, `logger.py`)

Atomic writes: `torch.save` to `.tmp`, then `os.replace`. Kaggle kills a session without a catchable
signal, and a kill during the ~2 s it takes to stream 286 MB would otherwise leave a truncated file
*and* have already destroyed the previous good one. `test_an_interrupted_write_leaves_the_previous_
checkpoint_intact` simulates exactly that.

Logging opens every file `"a"`, every time. A logger opened `"w"` — the default nearly everywhere —
erases hours of loss curve the instant a resumed session starts.

**Bug found by a test I wrote for something else.** `read_log` originally keyed records by step and
replaced on collision. The loop writes train metrics every `log_every` steps and validation metrics
every `eval_every` steps, so on an eval step the same step number is written twice — and the val
record was deleting the train record. In a real run that would have silently removed a training
point every 250 steps. Fixed to *merge* within a run_id and *replace* across run_ids, which is the
correct treatment of both cases (the second being a resumed run recomputing steps after its last
checkpoint).

### 4. Training loop (`lma/train/loop.py`) and entry points

AdamW (β₂=0.95), warmup→cosine to a 10% floor, grad clip 1.0, fp16 autocast + `GradScaler`,
gradient accumulation with `no_sync` on all but the last micro-step (otherwise 4 all-reduces per
optimizer step, 3 of them discarded).

**fp16, not bf16, and this is hardware not taste.** Kaggle's T4 is Turing (sm75): fp16 tensor cores,
no bf16 support at all. The scaler's scale is logged every step, because a collapsing scale is the
signature of gradient overflow and is otherwise indistinguishable from a plateau.

The LR schedule is a pure function of the step rather than a `lr_scheduler` object, so a resume
cannot drift it out of phase with the optimizer.

### 5. Local smoke run — three bugs found by running

`python hindi/train/pretrain.py --smoke` (50 steps, 256-token blocks, single process).

1. **`TypeError: RNG state must be a torch.ByteTensor` on the first resume.** `torch.load(...,
   map_location=cuda)` faithfully moves *every* tensor in the payload to the GPU, RNG states
   included, and both `torch.set_rng_state` and `torch.cuda.set_rng_state_all` require them on the
   CPU. On Kaggle this would have surfaced only *after* a session died — i.e. after hours of
   training. Fixed with explicit `.cpu()` on restore, plus loading the checkpoint to CPU in the
   first place (which also avoids a transient second 286 MB copy in GPU memory).
2. **Smoke runs were appending into the real run's log.** A 50-step throwaway run wrote into the same
   permanently-append-only `train_log.jsonl` the report's loss curve is plotted from, and left a
   50-step `ckpt_last.pt` for the real run to resume from. `--smoke` now redirects to `logs/smoke/`
   and `checkpoints/smoke/`.
3. **`train.log` was gitignored.** A `*.log` rule for LaTeX build artifacts (line 115 of
   `.gitignore`) was silently excluding a Phase-2 deliverable. Fixed with negations placed at the
   *end* of the file — gitignore is last-match-wins, and a negation before that rule was itself
   overridden, which `git check-ignore -v` reported.

Verified live across three sessions: step counter continued 50 → 80 → 100, val loss kept improving
7.29 → 7.14 → 7.07, `run_ids` accumulated to three entries, and the merged log holds a gapless
record of steps 5–100 with no duplicates and nothing lost.

### 6. DDP verification without torchrun (`scripts/check_ddp.py`)

`torchrun` cannot start on this Windows PyTorch build — it is compiled without libuv, so the
rendezvous store fails outright (`use_libuv was requested but PyTorch was build without libuv
support`). Kaggle is Linux and will not hit this, but it meant the DDP path could not be exercised
locally at all, which is exactly the code most likely to waste a session.

`scripts/check_ddp.py` initialises the process group through a `FileStore` instead and spawns its
own ranks, so it runs anywhere. It asserts: the group forms; `all_reduce_mean` genuinely averages
across ranks rather than echoing rank 0; **the ranks' parameters remain bit-identical after a step
taken on deliberately different per-rank data** (the property that fails silently if gradients are
not being synchronised); and the two samplers draw disjoint windows. All pass on 2 CPU ranks over
gloo. It is also cell 3 of the Kaggle runbook, there against nccl.

### 7. Evaluation (`lma/eval/`)

- **`perplexity.py`** — PPL and bits-per-byte. BPB divides total surprisal by the UTF-8 byte length
  of the decoded text, measured by decoding rather than estimated from a fertility constant. This
  is the number that compares Model H to Model L: the two tokenizers are trained separately, so a
  token is a different amount of language in each and perplexity across them is not comparable.
- **`textmetrics.py`** — BLEU, chrF, ROUGE-L from first principles, cross-checked against
  `sacrebleu` (the pattern Phase 1 used with `datasketch` and HuggingFace `tokenizers`).
- **`diversity.py`** — distinct-n (pooled, so cross-sample repetition is visible), within-sample
  repetition rate, self-BLEU.
- **`generate.py`** — the continuation protocol stated explicitly (128-token prompt from a real
  held-out document opening, 128 generated, scored against the true continuation) plus the
  5-point decoding grid.

**The diversity metrics earned their place immediately.** Run against a deliberately under-trained
30-step checkpoint, greedy decoding scored distinct-2 = 0.001 and self-BLEU = 90.9 — total
collapse — while perplexity looked like an ordinary early-training number. That is precisely the
failure perplexity cannot see, because it never samples.

### 8. Attention diagnostics (`lma/eval/attention.py`, `<lang>/eval/attention_report.py`)

Entropy, mean attention distance, previous-token and first-token mass, a head taxonomy, and
heatmaps with real Devanagari token labels.

**Entropy is reported normalised as well as raw.** Under causal masking, query *i* can attend over
only *i+1* keys, so the maximum achievable entropy at position *i* is log2(i+1), not log2(T). Raw
mean entropy therefore rises with position for purely arithmetic reasons, and a head behaving
identically everywhere reads as "spreading out over the sequence". `H / log2(i+1)` removes that.
Both closed forms are asserted: a one-hot row scores exactly 0, a uniform-over-history head scores
exactly 1.0 normalised, and a pure previous-token head scores mean distance exactly 1.

Memory is handled by construction: extraction runs at 192 tokens with batch 1 and consumes one
layer at a time, because `[1,6,1024,1024]` fp32 is 25 MB per layer and 250 MB across ten.

**Font handling.** `lma/viz/plots.devanagari_font()` probes for an Indic-capable family (Nirmala UI
on Windows, Noto Sans Devanagari on Linux/Kaggle) and raises a loud `RuntimeWarning` on fallback.
matplotlib does not warn on a missing glyph — it draws a hollow box, and a figure full of boxes
reads as a rendering setting rather than a missing font. Verified by rendering a heatmap and reading
the PNG back: Devanagari shapes correctly (हर, बोलना, छोटी, करें, लगातार, आपको …) and the causal
mask is visible as the dark upper-right triangle.

Sanity check on the 30-step checkpoint: all 60 heads classified `diffuse_global` at normalised
entropy ≈ 0.99 — i.e. attention is still essentially uniform, exactly right for an untrained model,
and confirmation that the normalisation behaves (uniform → 1.0). The interesting result comes after
real training.

### 9. Independence check (`scripts/verify_independence.py`)

The Phase-2 placeholder is replaced with a real weight-level check. A file-hash comparison alone is
weak evidence — two checkpoints of the *same* model at different steps also differ byte for byte —
so four checks are done: file hashes differ; **no two tensors share storage** (compared by
`data_ptr` with both models loaded into one process); **no parameter tensor is numerically
identical** between the models; and the embedding tables' cosine similarity is reported (≈0 is what
independently trained models must show).

Vocabulary size is deliberately *not* the proof. Phase 1 selected 16,000 pieces for both languages,
so the embedding tables are the same **shape** — which is exactly why the check has to look at
values and storage rather than dimensions.

Currently: 3 checks pass, checkpoints skipped (none exist yet).

### 11. RESULTS — both models trained and evaluated

Trained on Kaggle 2×T4, fp16 + DDP, one epoch each, no token seen twice.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Steps | 3,936 / 3,936 | 4,358 / 4,358 |
| Tokens seen | 515,899,392 (1.000 epochs) | 571,211,776 (1.000 epochs) |
| Sessions | 1 | **2 — died and resumed** |
| Median throughput | 75,699 tok/s | 79,651 tok/s |
| Peak GPU memory | 6.72 GB/device | 6.72 GB/device |
| Final val loss | 3.4237 | 3.9371 |
| **Val perplexity** | **30.34** | **51.23** |
| **Val bits-per-byte** | **0.4744** | **0.4545** |
| Bytes per token | 10.38 | 12.50 |
| Val loss monotone | yes, all 16 evals | yes, all 18 evals |
| Train − val gap | +0.0135 | −0.1116 |
| fp16 scaler backoffs | 0 | 0 |
| Steps clipped (>1.0) | 4 / 394 | 5 / 436 |

**The perplexity ranking reverses under bits-per-byte, and that is the headline result.** Hindi's
perplexity (30.34) looks 40% better than Nepali's (51.23), which would invite the conclusion that
Model H is the stronger model. It is not a fair comparison: the two tokenizers were trained
separately, and Nepali's packs 12.50 bytes into a token against Hindi's 10.38. Each Nepali token is
therefore a *larger* prediction problem, and its higher per-token perplexity is partly the tokenizer's
doing rather than the model's. Normalised per byte the order flips — Nepali 0.4545 vs Hindi 0.4744,
i.e. Model L models its language slightly *better* per unit of text. This is exactly the artefact BPB
exists to remove, and it is why the report leads with BPB and reports perplexity with the caveat
attached.

Arithmetic check: ln(30.337)/ln(2) = 4.923 bits/token ÷ 10.376 bytes/token = 0.4745. ✓

**Neither model overfitted.** Val loss fell monotonically at every single evaluation in both runs, and
Nepali's final training loss is *higher* than its validation loss (−0.112). Both are the direct
consequence of the one-epoch design: nothing was memorised because nothing was seen twice.

**fp16 was never at risk.** Zero scaler backoffs in either run; the scale *grew* (Hindi 65k→131k,
Nepali 65k→262k), and GradScaler only doubles after 2,000 consecutive overflow-free steps — so the
growth is itself proof there were none. Gradient norms averaged ~0.44 against a 1.0 clip; only 9 of
830 logged steps clipped across both runs, all during warmup.

**Resume was proven in production, not just in tests.** The Nepali session died and was restarted;
the log carries two `run_id`s, the step counter continued rather than restarting, and no history was
lost. That is the append-only + atomic-checkpoint design doing exactly the job it was built for.

### 12. Attention findings

| layer | Hindi entropy(norm) | Hindi distance | Nepali entropy(norm) | Nepali distance |
|---|---|---|---|---|
| 0 | 0.903 | 43.53 | 0.873 | 41.11 |
| 3 | 0.841 | 33.63 | 0.866 | 36.42 |
| **6** | **0.665** | **17.46** | **0.654** | **17.43** |
| 7 | 0.673 | 25.45 | 0.603 | 17.30 |
| 9 | 0.588 | 34.56 | 0.660 | 30.23 |

Head roles — Hindi: 46 diffuse-global, 9 mixed, 4 long-range, 1 previous-token.
Nepali: 42 diffuse-global, 10 mixed, 7 long-range, 1 previous-token.

**Attention specialises with depth in both models.** Early layers are near-uniform (normalised
entropy ~0.9, reaching 40+ tokens back); late layers focus sharply (0.59–0.66) and previous-token
mass rises from ~0.04 at layer 0 to 0.16 (Hindi, layer 8) and 0.24 (Nepali, layer 7). The
distance-vs-entropy scatter shows this as a clean colour gradient from top-right (early, diffuse) to
bottom-left (late, focused).

**The most striking result is a cross-language coincidence.** Both models independently converge on
layer 6 as their "focusing" layer, with mean attention distances of **17.46 and 17.43 tokens** — two
models trained separately, on different languages, from different random initialisations, agreeing
to three significant figures. The architecture is identical and the training recipe is identical, so
this is evidence that the depth at which a decoder switches from diffuse mixing to local
composition is a property of the architecture and token budget rather than of the language. That is
a genuine Phase-3 finding, set up here.

Heatmaps confirm it qualitatively: `hindi_attn_L9H3.png` shows a crisp near-diagonal band with
Devanagari token labels — a learned recency head. The same figure on the 30-step sanity checkpoint
was uniform noise.

### 13. Generation and diversity

Protocol: 200 held-out documents, first 128 tokens as prompt, 128 tokens generated, scored against
that document's own next 128 tokens. BLEU is corpus-level BLEU-4. The four settings the assignment
names are run as **pure** temperature sweeps (no top-k, no top-p): mixing nucleus truncation into a
row labelled "temperature 1.0" would make the number describe the truncation as much as the
temperature, and the rows would not be comparable to each other.

| | | Hindi chrF | Hindi BLEU | Hindi R-L | Hindi dist-2 | Nepali chrF | Nepali BLEU | Nepali R-L | Nepali dist-2 |
|---|---|---|---|---|---|---|---|---|---|
| greedy | T=0 | 15.95 | 1.33 | 10.98 | 0.150 | 17.67 | 0.80 | 7.00 | 0.165 |
| **T=0.5** | | 21.55 | **2.32** | **13.37** | 0.441 | 23.82 | **0.88** | **8.64** | 0.570 |
| **T=1.0** | | **22.88** | 0.92 | 11.11 | 0.842 | **26.09** | 0.00 | 7.47 | 0.911 |
| **T=1.5** | | 20.01 | **0.00** | 4.42 | 0.994 | 24.00 | **0.00** | 3.17 | 0.994 |

**The three metrics peak at different temperatures, and that disagreement is the finding.** BLEU and
ROUGE-L peak at T=0.5, chrF at T=1.0. They are measuring different things: BLEU and ROUGE-L reward
matching *this particular* reference continuation, which favours conservative decoding, while chrF's
character n-grams keep scoring plausible Devanagari morphology even when the words diverge from the
reference. At T=1.5 BLEU is exactly **0.00** for both models — not one 4-gram matched across 200
continuations — while distinct-2 reaches 0.994. That is the fidelity/diversity trade-off at its
limit, and greedy is the other end: lowest distinct-2 (0.150 / 0.165), highest repetition.

Why each metric is or is not informative here, for the report: **BLEU is close to uninformative** —
it requires exact 4-gram overlap with one reference, and there is no single correct continuation of
a document prefix, so a fluent and sensible paragraph scores near zero exactly as gibberish does.
**ROUGE-L is slightly better** because its longest-common-subsequence allows gaps. **chrF is the
least bad of the three** for Devanagari: it scores character n-grams, so a correct word form
differing by one inflectional suffix costs a fraction of a match rather than four whole n-grams,
which matters in two heavily inflected languages.

**Greedy decoding no longer collapses.** On the 30-step sanity checkpoint, greedy scored distinct-2
= 0.001 and self-BLEU = 90.9 — total degeneration. Trained: distinct-2 0.150 / 0.165 with self-BLEU
under 0.5. Self-BLEU is ~0 at nearly every setting in both models, so there is no cross-prompt
repetition — the models are not emitting one memorised paragraph regardless of input.

**Reproducibility caveat, recorded in `metrics_val.json`.** `continuations()` seeds once per call,
so the batch shape determines how many values each `multinomial` draw consumes: the same model and
the same seed at a different `--gen-batch` walks a different random stream and moves chrF by a few
tenths. The generation protocol (n_docs, prefix/continuation lengths, gen_batch, seed, BLEU order)
is therefore stored alongside the numbers so the table can be regenerated exactly.

### 13b. Qualitative notes on coherence and language correctness

Read directly from `<lang>/eval/results/samples.json` (25 continuations each, T=0.5).

**Both models produce grammatically well-formed text in the right language.** Hindi output carries
correct postpositions and case marking (को, की, से, में), correct verb agreement, and picks up the
news register from the prompt — "उन्होंने कहा कि", "ग्रामीणों का आरोप है कि". Nepali output has
correct Nepali verb morphology (भएको, गरेको, बताउँछन्, हुनुपर्छ), the ergative -ले, and honorific
forms. **Nepali chandrabindu is preserved correctly** in generated text (बताउँछन्, खाँचो) — a direct
downstream confirmation that the Phase-1 decision *not* to fold ँ to anusvara for Nepali was right;
had it been folded, the model would be generating misspelt Nepali here.

Script purity is essentially reference-level: 99.71% Devanagari for Hindi against the reference's
99.67%. Nepali generates 0.40% Latin against the reference's **0.00%** — a handful of stray ASCII
characters (one sample opens mid-word with a bare "m"), which is byte-fallback tokens surfacing.
Small, but a real defect and worth naming rather than rounding away.

**Local coherence is good; global coherence is absent.** Continuations stay on topic — a piece about
librarians stays about librarians, one about the forest department stays about the forest department
— but they do not develop an argument, and they contradict themselves across sentences. One Hindi
sample states "बच्चों को स्कूल में नहीं पढ़ाना चाहिए" and then, two sentences later, "बच्चों को पढ़ाना
चाहिए". That is what a 24M-parameter model with no instruction tuning does.

**The dominant failure is phrase-level looping**, and it is worse in Nepali. One Nepali sample emits
"संविधान असफल भएको हो" three times and then cycles "संविधान असफल भयो" several more; a Hindi sample
repeats "वन विभाग की कार्यशैली पर सवाल उठाए" four times. The type-token ratio quantifies it:
generated text has **25% unique words for Hindi against the reference's 47%**, and **36% against 68%
for Nepali** — so Nepali's TTR gap is the wider of the two, matching its higher looping rate.

Both models over-produce sentence boundaries — danda per 100 words is 7.0 (Hindi) and 9.0 (Nepali)
against references at 5.5 and 6.0. The model resolves uncertainty by ending the sentence, which is a
cheap way to stay locally fluent.

One Hindi sample drifts out of prose into what is plainly a Wikipedia navigation fragment
("कॉपीराइट के बारे में नए जुड़े पन्नों की सूची"). That is a training-data artifact surfacing at
generation time: boilerplate the Phase-1 cascade did not remove is reproducible from the model, which
is a more honest audit of corpus cleanliness than any filter-pass-rate statistic.

### 14. Independence, verified on the real weights

`scripts/verify_independence.py` — all four checks pass:

- 73 tensors compared across the two checkpoints: **0 share storage, 0 are numerically identical**
- Embedding tables are the same *shape* (16000 × 384) — which is why the check compares values and
  storage rather than dimensions — with **cosine similarity +0.00448**, i.e. indistinguishable from
  unrelated, as independently trained models must be
- Zero document overlap between corpora (3,587,040 vs 5,536,919 documents)
- Tokenizer hashes and piece sets differ (5,695 of 16,000 pieces shared, all Devanagari substrings)

### 10. Kaggle runbook (`notebooks/phase2_kaggle.md`)

Written as markdown, not a committed `.ipynb`: notebook JSON diffs are unreadable and stored outputs
bloat the repo. Covers building the private `lma-encoded` dataset, the `compute_cap 7.5` check that
justifies fp16, a mandatory 2-minute DDP smoke cell before the real run, and the recovery
procedure — which is simply to re-run the same cell, since resume is automatic.
