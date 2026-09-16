"""
Reasoning finetuning for Model H (Hindi), starting from its own Phase-2 checkpoint.

    python hindi/reasoning/finetune.py --smoke                         # 40-step check
    python hindi/reasoning/finetune.py --condition sft --lr 1e-4 --seed 0
    python hindi/reasoning/finetune.py --condition replay --lr 1e-4 --seed 1
    python hindi/reasoning/finetune.py --condition sft --lr 1e-4 --train-size 1000

Checkpoints go to `hindi/reasoning/checkpoints/<run>/` (ckpt_best.pt by validation exact match,
ckpt_last.pt for resume) and append-only logs to `hindi/reasoning/logs/<run>/`. The tokenizer and
vocabulary are Hindi's own and are never modified. A run resumes automatically; --fresh opts out.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hindi.reasoning.config import SPEC  # noqa: E402
from lma.reasoning.cli import cmd_finetune  # noqa: E402

if __name__ == "__main__":
    cmd_finetune(SPEC)
