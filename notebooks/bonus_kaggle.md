# Bonus on Kaggle — Model H without positional embeddings (NoPE)

Runbook for the optional bonus ablation: retrain Model H (Hindi) with RoPE removed, run the full
Phase-2 evaluation suite on it, and compare it with the standard Model H. One Kaggle session, 2×T4.

The encoded Hindi corpus is **not** uploaded again: it is already in the Phase-2 `lma-phase2` Dataset.
Only the bonus code (~2 MB) is a new upload.

| What | Time on 2×T4 |
|---|---|
| Cells 1–3: setup, tests, smoke | ~5 min |
| Cell 4: pretraining, 3,936 steps (identical to Phase 2) | ~2 h |
| Cells 5–7: evaluation, attention, position probe | ~30–45 min |
| Cell 8: comparison + results zip | < 1 min |

---

## Step 1 — build the code archive (local, once)

```powershell
python scripts/pack_bonus.py code
```

Produces `dist/lma-bonus-code.zip`. It contains `code/` with only what the ablation imports:

- `lma/{model,train,eval,tokenizer,viz}`
- `hindi/{model,train,eval,nope}` and the Hindi tokenizer
- the standard model's committed Phase-2 results, training log and position probe (the comparison
  baseline)
- the Phase-2 model/train/eval tests

## Step 2 — upload it as a Kaggle Dataset (once)

1. <https://www.kaggle.com/datasets> → **New Dataset** → drag in `dist/lma-bonus-code.zip`
   (Kaggle extracts it; do not unzip first).
2. Title **`lma-bonus-code`**, visibility **Private**.

## Step 3 — new notebook

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 ×2** |
| Internet | **On** |
| Persistence | Variables and Files |

**Add Input** → both **`lma-phase2`** (the data) and **`lma-bonus-code`** (the code).

---

## Cell 1 — locate both datasets, install

```python
import os, glob, json, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)

# Both datasets contain a code/ directory, so each is found by a file only it has.
code_hits = glob.glob("/kaggle/input/**/code/hindi/nope/pretrain.py", recursive=True)
data_hits = glob.glob("/kaggle/input/**/data/hindi/train.bin", recursive=True)
assert code_hits, "attach the lma-bonus-code dataset"
assert data_hits, "attach the lma-phase2 dataset"
CODE_SRC = code_hits[0].split("/code/")[0] + "/code"
DATA = os.path.dirname(data_hits[0])                  # .../data/hindi
REPO = "/kaggle/working/repo"
print("code:", CODE_SRC, "\ndata:", DATA)

got = json.load(open(f"{DATA}/encoding.json"))["total_tokens"]
print("hindi tokens", f"{got:,}", "OK" if got == 517640653 else "MISMATCH")

!rm -rf {REPO} && cp -r {CODE_SRC} {REPO}
%cd {REPO}
!pip install -q -e . sentencepiece sacrebleu
```

Everything below runs **inside `/kaggle/working/repo` with the repository's default output paths**,
so checkpoints, logs, results and figures land exactly where they would locally
(`hindi/nope/...`, `report/figures/`), and the results zip in cell 8 is built from the same layout.

## Cell 2 — tests on the exact tree that will train

```python
!python -m pytest tests/test_model.py tests/test_eval.py tests/test_train.py -q \
    -k "not language and not separate and not smoke_runs" 2>&1 | tail -3
```

The deselected tests check the *Nepali* configs, which are deliberately not in this archive.
`test_model.py` includes the four NoPE tests (identical tokens get identical layer-0 attention without
RoPE; unchanged parameter count and causal mask; the fused and manual attention paths agree; old
checkpoints still load with RoPE).

## Cell 3 — smoke run through DDP (~2 min)

```python
!torchrun --standalone --nproc_per_node=2 hindi/nope/pretrain.py --smoke --data-dir {DATA}
```

Requirements to continue: the first line prints `use_rope=False`, `world_size=2` appears in the
`RUN_START` record, and the loss falls. Smoke output goes to `hindi/nope/checkpoints/smoke/` and
`hindi/nope/logs/smoke/`, so it can never become the real run's resume point.

## Cell 4 — the real run (~2 h)

```python
!torchrun --standalone --nproc_per_node=2 hindi/nope/pretrain.py --data-dir {DATA}
```

Same `TrainConfig` object as Phase 2: 3,936 steps × 131,072 tokens, peak LR 6e-4, seed 42.

**If the session dies, re-run this cell.** It resumes from `hindi/nope/checkpoints/ckpt_last.pt` and
appends to the same log.

**If the loss diverges** (NaN or a sustained rise): stop and report it. A NoPE model is
not expected to be unstable at this learning rate; if it is, that is itself a result.

## Cell 5 — Phase-2 evaluation suite, test split (~20–30 min)

```python
CK = "hindi/nope/checkpoints/ckpt_best.pt"
COMMON = "--results-dir hindi/nope/results --figure-prefix hindi_nope --label model_h_nope"
!python hindi/eval/evaluate.py --checkpoint {CK} --data-dir {DATA} --split test {COMMON}
!python hindi/eval/evaluate.py --checkpoint {CK} --data-dir {DATA} --split val --skip-generation {COMMON}
```

This is the *unchanged* Phase-2 script: perplexity and bits-per-byte, BLEU / chrF / ROUGE-L over the
same 200 documents and 7 decoding settings, and the diversity diagnostics. It writes
`hindi/nope/results/metrics_{test,val}.json` and `report/figures/hindi_nope_diversity_grid.png`.

## Cell 6 — attention analysis (~2 min)

```python
!python hindi/eval/attention_report.py --checkpoint {CK} --data-dir {DATA} {COMMON} \
    --extra-heads 0:3,0:4,0:5,5:0,5:1,5:2,9:0,9:3
```

Same protocol as Phase 2 (8 validation documents × 192 tokens): entropy, mean distance, head
taxonomy, and heatmaps. `--extra-heads` adds the exact heads plotted for the standard model in
Phase 2, so the heatmaps can be compared head for head.

## Cell 7 — position probe (~3 min)

```python
!python hindi/nope/position_probe.py --checkpoint {CK} --label model_h_nope --data-dir {DATA}
```

Measures loss by position in the window and how much shuffling the context hurts. The standard
model's probe is already in the archive (`hindi/nope/results/position_probe_model_h.json`).

## Cell 8 — comparison and the results zip

```python
!python hindi/nope/compare.py
!python scripts/pack_bonus.py results --out /kaggle/working/lma-bonus-results.zip
!du -h hindi/nope/checkpoints/*.pt /kaggle/working/lma-bonus-results.zip
```

**Download** from the Output tab:

- `lma-bonus-results.zip` — logs, result JSON/markdown and figures. Unpack it locally with
  `logs/* → hindi/nope/logs/`, `results/* → hindi/nope/results/`, `figures/* → report/figures/`.
- `repo/hindi/nope/checkpoints/ckpt_best.pt` (286 MB) → Google Drive, next to the Phase-2 checkpoints.

Kaggle's Output tab only lists files directly under `/kaggle/working`, so copy the checkpoint up a
level first if it does not appear:

```python
!cp hindi/nope/checkpoints/ckpt_best.pt /kaggle/working/hindi_nope_ckpt_best.pt
```
