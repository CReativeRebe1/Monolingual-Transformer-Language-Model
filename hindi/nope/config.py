"""
Bonus ablation — Model H without positional embeddings ("NoPE"): configuration and paths.

The ablation changes exactly one thing. Model H's only positional signal is RoPE applied to the
queries and keys in every attention layer (`lma/model/rope.py`); there is no learned or sinusoidal
position table anywhere. Setting `use_rope=False` therefore removes all explicit position
information, and nothing else:

* **same architecture and parameter count** -- RoPE has no parameters, so the model is still
  23,846,784 parameters (asserted below), and the comparison is not confounded by capacity
* **same data, tokenizer and vocabulary** -- Hindi's Phase-1 encoded corpus and 16k Unigram model
* **same optimisation** -- the Phase-2 `TrainConfig` object itself is reused, not copied: peak LR
  6e-4, 2% warmup, cosine to 10%, AdamW (0.9, 0.95), weight decay 0.1, clip 1.0, 131,072 tokens per
  step, exactly one epoch (3,936 steps), seed 42, fp16 on 2×T4
* **same evaluation** -- `hindi/eval/evaluate.py` and `hindi/eval/attention_report.py`, unchanged
  apart from where they write

The causal mask stays. It is not position information in the embedding sense, but it is the one
channel through which a NoPE decoder can still infer order (see `lma/eval/position.py`).

Hindi was chosen for the ablation because its Phase-2 run completed in a single uninterrupted session,
so the baseline being compared against has the cleanest possible training record.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hindi.model.config import CONFIG as BASE_CONFIG, ENCODED_DIR, TOKENIZER_PATH  # noqa: E402
from hindi.train.config import CONFIG as TRAIN_CONFIG, SMOKE as SMOKE_CONFIG  # noqa: E402
from lma.model.transformer import GPT, analytic_param_count  # noqa: E402

MODEL_NAME = "model_h_nope"
BASELINE_NAME = "model_h"

CONFIG = replace(BASE_CONFIG, use_rope=False)

NOPE_DIR = REPO_ROOT / "hindi" / "nope"
OUT_DIR = NOPE_DIR / "checkpoints"
LOG_DIR = NOPE_DIR / "logs"
RESULTS_DIR = NOPE_DIR / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
FIGURE_PREFIX = "hindi_nope"

# The standard Model H's committed Phase-2 record, compared against.
BASELINE_RESULTS_DIR = REPO_ROOT / "hindi" / "eval" / "results"
BASELINE_LOG = REPO_ROOT / "hindi" / "train" / "logs" / "train_log.jsonl"
BASELINE_CHECKPOINT = REPO_ROOT / "hindi" / "train" / "hindi_ckpt_best.pt"

# The heads plotted for the standard model in Phase 2, so the NoPE heatmaps can be compared head for
# head (attention_report.py --extra-heads).
BASELINE_HEATMAP_HEADS = "0:3,0:4,0:5,5:0,5:1,5:2,9:0,9:3"

N_PARAMS = analytic_param_count(CONFIG)["total"]
if N_PARAMS != analytic_param_count(BASE_CONFIG)["total"]:
    raise AssertionError("the NoPE ablation must not change the parameter count")


def build() -> GPT:
    """Constructs Model H with positional embeddings removed."""
    return GPT(CONFIG)


__all__ = ["CONFIG", "TRAIN_CONFIG", "SMOKE_CONFIG", "ENCODED_DIR", "TOKENIZER_PATH", "build"]
