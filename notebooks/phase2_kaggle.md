# Phase 2 on Kaggle — 2×T4 DDP pretraining

Runbook for pretraining Model H and Model L on Kaggle's free two-GPU session.

Written as markdown rather than a committed `.ipynb` because notebook JSON diffs are unreadable and
stored outputs bloat the repo; the cells below are pasted into a fresh Kaggle notebook. The
resulting `.ipynb` is gitignored.

Code **and** data ship in one archive, so nothing has to be cloned at runtime — that removes the two
things that can fail before training starts (a clone against a private repo needing credentials, and
a dataset whose layout does not match what these cells assume).

---

## Step 1 — build the archive (local, once)

```powershell
python scripts/pack_kaggle.py
```

Produces `dist/lma-phase2.zip` — **1.64 GB**, from 2.26 GB of raw files. Contents:

```
code/     lma/ hindi/ nepali/ scripts/ tests/ setup.py requirements.txt
          + each language's selected tokenizer .model/.vocab
data/
  hindi/  train.bin train.idx val.bin val.idx test.bin test.idx encoding.json
  nepali/ same
```

Deliberately excluded: `<lang>/tokenizer/sweep/` (260 MB per language of Phase-1 vocabulary-sweep
models nothing in Phase 2 reads) and `<lang>/data/splits/*.txt` (2.7 GB of plaintext already
represented by the encoded `.bin`). Including either would roughly triple the upload for no benefit.

## Step 2 — upload as a Kaggle Dataset (once)

1. <https://www.kaggle.com/datasets> → **New Dataset**
2. Drag in `dist/lma-phase2.zip`. Kaggle auto-extracts it, so `code/` and `data/` become real
   directories — do not unzip it yourself first.
3. Title: **`lma-phase2`**. Visibility: **Private**.
4. Create, and wait for it to finish processing (a few minutes after the upload).

The mount path varies: Kaggle uses `/kaggle/input/<slug>/` for some notebooks and
`/kaggle/input/datasets/<username>/<slug>/` for others, and appends a numeric suffix to the slug if
the title is already taken. Cell 1 therefore **locates the dataset by content** (globbing for
`setup.py`) rather than assuming a path, so none of that matters.

## Step 3 — new notebook, correct settings

**New Notebook**, then in the right-hand panel:

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 ×2** |
| Internet | **On** (needed for `pip install -e .`) |
| Persistence | Variables and Files |

Then **Add Input** → your `lma-phase2` dataset.

---

## Cell 1 — environment and paths

```python
import os, glob, subprocess

print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,compute_cap",
                      "--format=csv"], capture_output=True, text=True).stdout)

# Kaggle mounts a dataset at /kaggle/input/<slug>/ OR /kaggle/input/datasets/<user>/<slug>/
# depending on how it was added, so locate it by content instead of assuming a path.
hits = glob.glob("/kaggle/input/**/setup.py", recursive=True)
assert hits, "dataset not attached: right panel -> Add Input -> your dataset -> +"
DATA = os.path.dirname(os.path.dirname(hits[0]))     # the dir holding code/ and data/
REPO = "/kaggle/working/repo"
OUT  = "/kaggle/working/out"

for lang in ("hindi", "nepali"):
    assert os.path.isfile(f"{DATA}/data/{lang}/train.bin"), f"missing {lang} train.bin under {DATA}"
print("DATA =", DATA)

!rm -rf {REPO} && cp -r {DATA}/code {REPO}
%cd {REPO}
!pip install -q -e . sentencepiece sacrebleu
```

Only `sentencepiece` and `sacrebleu` are installed on top of the Kaggle image, **not**
`requirements.txt`. That file pins `torch>=2.5.0`, and letting pip act on it risks reinstalling
torch — minutes of a limited session, and a real chance of ending up with a build that does not
match the driver. Everything Phase-2 training and evaluation imports is either in the Kaggle image
already (torch, numpy, matplotlib) or in that two-package list. The Phase-1 dependencies
(fastText, trafilatura, xxhash, …) belong to the data pipeline, which is not being run here.

`compute_cap` must print **7.5** (Turing). That is the line justifying fp16 rather than bf16
throughout `lma/train/loop.py`: sm75 has fp16 tensor cores and no bf16 at all. Two `Tesla T4` rows
must appear — one means the accelerator setting did not take, and `--nproc_per_node=2` will fail in
cell 3.

The `rm -rf {REPO}` makes the cell re-runnable: without it a second run copies `code/` *inside* the
existing `repo/`, and the `%cd` then lands somewhere with no `setup.py`.

The copy to `/kaggle/working` is only the code (a few MB); the 2.1 GB of `.bin` stays read-only on
`/kaggle/input` and is memory-mapped from there. Copying it would waste both time and the 20 GB
working-disk budget for nothing.

## Cell 2 — confirm the data mounted

```python
import json
for lang, expected in [("hindi", 517640653), ("nepali", 572506257)]:
    got = json.load(open(f"{DATA}/data/{lang}/encoding.json"))["total_tokens"]
    print(f"{lang:7} {got:,} tokens", "OK" if got == expected else f"MISMATCH (expected {expected:,})")
!ls -la {DATA}/data/hindi/
```

A mismatch means the wrong build was packed. Finding that out here costs a minute; finding it out
from a finished model costs the run.

## Cell 3 — sanity-check the code, then the DDP path (~3 min)

```python
!python -m pytest tests/test_model.py tests/test_train.py tests/test_eval.py -q 2>&1 | tail -3
```

Expect **99 passed**. This runs on the exact tree that is about to train, so an import broken by
packaging shows up here rather than 40 minutes into a session.

Only the three Phase-2 files, not the whole 453-test suite: the Phase-1 tests import the data
pipeline's dependencies (fastText, trafilatura, xxhash), which are deliberately not installed here.
Running them would report import errors that say nothing about whether training will work. The full
suite runs locally, where those dependencies exist.

One test in this set only *exists* on a GPU — `test_resume_works_when_the_checkpoint_is_mapped_to_
cuda` is skipped locally on a CPU-only box, so Kaggle is the first place it actually executes. It
covers the RNG-state bug that made the first real resume raise `TypeError: RNG state must be a
torch.ByteTensor`.

```python
!torchrun --standalone --nproc_per_node=2 hindi/train/pretrain.py --smoke \
    --data-dir {DATA}/data/hindi --out-dir {OUT}/hindi/checkpoints --log-dir {OUT}/hindi/logs
```

**Never skip this.** A NCCL failure or a rank mismatch found here costs two minutes; found in the
real run it costs the session. Requirements to continue: loss falls, and `world_size=2` appears in
the logged `RUN_START` record.

Smoke runs write to `checkpoints/smoke/` and `logs/smoke/` automatically, even though the same
`--out-dir` is given here as in cell 4 — so this 50-step throwaway can never be picked up as a
resume point by the real run, and never appends into the log the report's loss curve is plotted from.

## Cell 4 — the real run (Hindi, ~2–3 h)

```python
!torchrun --standalone --nproc_per_node=2 hindi/train/pretrain.py \
    --data-dir {DATA}/data/hindi \
    --out-dir {OUT}/hindi/checkpoints \
    --log-dir {OUT}/hindi/logs
```

3,936 steps × 131,072 tokens = one epoch over 516M tokens.

**If the session dies, re-run this exact cell.** It finds `ckpt_last.pt`, resumes from it, and
appends to the same log. With `save_every=250`, at most ~8 minutes of work is ever at risk.

Watch the first logged line. `tokens_per_s` replaces the 2–3 h estimate with a measurement — if the
extrapolation exceeds your remaining session time, stop and restart with `--max-steps N` set to a
round number the session can finish. Report the reduced token budget rather than letting the run be
cut off mid-schedule: the cosine decay assumes it reaches the end, and a truncated schedule leaves
the model at a learning rate it was never meant to stop at.

**If a GPU runs out of memory** (it should not — measured peak is 6.1 GiB per GPU against 15 GiB
available):

```python
--micro-batch 8 --grad-accum 8      # same 131,072 tokens/step, half the activation memory
```

Keep `micro_batch × grad_accum` at 64 so tokens-per-step, and therefore the learning-rate schedule,
are unchanged.

## Cell 5 — Nepali

Identical to cells 3–4 with `hindi` → `nepali` throughout. 4,358 steps.

**Run it in a separate session.** Both languages in one session puts the 12-hour limit within reach,
and they have no reason to share one.

## Cell 6 — retrieve the results

```python
import shutil
for lang in ["hindi"]:                       # add "nepali" in that session
    shutil.make_archive(f"/kaggle/working/{lang}_logs", "zip", f"{OUT}/{lang}/logs")
!du -h {OUT}/*/checkpoints/*.pt
```

Download from the notebook's **Output** tab:

- `<lang>_logs.zip` → unpack into `<lang>/train/logs/` in the repo. These are the committed record
  the report's loss curves are plotted from.
- `ckpt_best.pt` and `ckpt_last.pt` (~286 MB each) → Google Drive. They are gitignored.

---

## After Kaggle

Nothing further needs a GPU session. With the checkpoints local:

```powershell
python hindi/eval/evaluate.py         --checkpoint hindi/train/checkpoints/ckpt_best.pt
python hindi/eval/attention_report.py --checkpoint hindi/train/checkpoints/ckpt_best.pt
python nepali/eval/evaluate.py        --checkpoint nepali/train/checkpoints/ckpt_best.pt
python nepali/eval/attention_report.py --checkpoint nepali/train/checkpoints/ckpt_best.pt
python scripts/verify_independence.py
```

Run `evaluate.py --split test` **once**, at the very end, after every decision is settled. A test set
consulted while tuning is a validation set with a misleading name.

## Numbers to expect

| | value | source |
|---|---:|---|
| Parameters | 23,846,784 | asserted at import by both model configs |
| Peak GPU memory / T4 | 6.1 GiB | measured, extrapolated from micro-batch 1–3 at block 1024 |
| Tokens per optimizer step | 131,072 | 16 × 1024 × 4 accum × 2 GPUs |
| Steps (Hindi / Nepali) | 3,936 / 4,358 | one epoch, derived from `encoding.json` |
| Checkpoint size | 286 MB | weights + AdamW moments, measured |
| Starting loss | ≈ 9.68 | ln(16,000), i.e. a uniform prior over the vocabulary |

## Kaggle limits worth knowing

| Limit | Value | Consequence here |
|---|---|---|
| Session length | 12 h | One language per session; ~3 h leaves wide margin |
| Weekly GPU quota | 30 h | Enough for both languages plus reruns |
| `/kaggle/working` | 20 GB | Two 286 MB checkpoints per language — not close |
| Idle timeout | ~20 min without interaction | The reason resume exists, not a reason to babysit |
