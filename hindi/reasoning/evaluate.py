"""
Reasoning evaluation for Model H (Hindi).

    python hindi/reasoning/evaluate.py --pretrained --shots 0
    python hindi/reasoning/evaluate.py --pretrained --shots 3
    python hindi/reasoning/evaluate.py --run sft_lr0.0001_s0
    python hindi/reasoning/evaluate.py --summarize

Each model is scored on val and the four test splits (generative exact match, candidate-choice
accuracy, prefix match, error taxonomy, per task and per family) and on Phase-2 validation perplexity,
the forgetting measure. `--summarize` groups seeds into mean ± std tables
(`hindi/reasoning/results/summary.md`) and draws the report figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hindi.reasoning.config import SPEC  # noqa: E402
from lma.reasoning.cli import cmd_evaluate  # noqa: E402

if __name__ == "__main__":
    cmd_evaluate(SPEC)
