"""
Reasoning finetuning for Model L (Nepali), starting from its own Phase-2 checkpoint.

    python nepali/reasoning/finetune.py --smoke                         # 40-step check
    python nepali/reasoning/finetune.py --condition sft --lr 1e-4 --seed 0
    python nepali/reasoning/finetune.py --condition replay --lr 1e-4 --seed 1
    python nepali/reasoning/finetune.py --condition sft --lr 1e-4 --train-size 1000

Checkpoints go to `nepali/reasoning/checkpoints/<run>/` (ckpt_best.pt by validation exact match,
ckpt_last.pt for resume) and append-only logs to `nepali/reasoning/logs/<run>/`. The tokenizer and
vocabulary are Nepali's own and are never modified. A run resumes automatically; --fresh opts out.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nepali.reasoning.config import SPEC  # noqa: E402
from lma.reasoning.cli import cmd_finetune  # noqa: E402

if __name__ == "__main__":
    cmd_finetune(SPEC)
