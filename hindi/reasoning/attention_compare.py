"""
Post-finetuning attention analysis for Model H (Hindi), with the Phase-2 attention toolkit.

    python hindi/reasoning/attention_compare.py                       # SFT + replay, seed 0 (the default)
    python hindi/reasoning/attention_compare.py --run sft_lr0.0001_s0

Pretrained vs finetuned, on the same comparative prompt and the same held-out text:
* heatmaps for an early and a late layer (default layers 0 and 9, heads 0, 3, 5)
* per-layer entropy and mean distance, on reasoning prompts and on ordinary validation text
* head-role counts before and after, and a Δ-entropy [layer x head] grid
* answer-pointer score (attention from the answer position to the correct entity minus distractors)
  before, after, and its Δ grid

Writes `hindi/reasoning/results/attention_compare.json` and figures under `report/figures/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hindi.reasoning.config import SPEC  # noqa: E402
from lma.reasoning.cli import cmd_attention  # noqa: E402

if __name__ == "__main__":
    cmd_attention(SPEC)
