"""
Model H (Hindi): Phase-3 reasoning configuration -- dataset seed, SFT hyperparameters, paths.

Per language, like every other config in this project, even where the numbers currently coincide
with Nepali's: the learning rate is chosen by *this* model's validation sweep, and a shared file would
make it impossible for the two to disagree when the sweep says they should.

Starting values and why:
* **lr 1e-4** -- the middle of the sweep {5e-5, 1e-4, 3e-4}. Pretraining peaked at 6e-4 from random
  init; finetuning starts from a good solution and should move it far less. `finetune.py --lr` runs
  the sweep; `scripts/run_phase3.py` picks the winner on validation exact match.
* **batch 64 x 3 epochs**, warmup 5% then cosine to 10% -- ~940 steps on 20,000 examples.
* **AdamW beta = (0.9, 0.95), weight decay 0.1 on matrices, clip 1.0** -- unchanged from pretraining,
  so the optimiser is not a confound in the before/after comparison.
* **dropout 0.1** -- SFT repeats examples; pretraining never did.
* **replay 8 x 256 tokens per step, weight 1** -- about a third of each step's tokens come from the
  pretraining distribution, enough to anchor it without drowning the ~5,000 answer-bearing tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hindi.model.config import ENCODED_DIR, LANGUAGE, MODEL_NAME, TOKENIZER_PATH  # noqa: E402
from hindi.reasoning import lexicon, templates  # noqa: E402
from lma.reasoning.build import BuildConfig  # noqa: E402
from lma.reasoning.cli import LangSpec  # noqa: E402
from lma.reasoning.sft import SFTConfig  # noqa: E402

REASONING_DIR = REPO_ROOT / "hindi" / "reasoning"
PRETRAINED = REPO_ROOT / "hindi" / "train" / "hindi_ckpt_best.pt"
FIGURES_DIR = REPO_ROOT / "report" / "figures"

BUILD = BuildConfig(seed=1234)

SFT = SFTConfig(
    lr=1e-4, min_lr_ratio=0.1, warmup_fraction=0.05, weight_decay=0.1, beta1=0.9, beta2=0.95,
    grad_clip=1.0, batch_size=64, epochs=3.0, dropout=0.1,
    replay_batch=8, replay_block=256, replay_weight=1.0,
    eval_every=100, val_em_examples=1000, patience=4, log_every=10,
)

# Shapes, masking, resume and logging in a minute: never used for a reported number.
SMOKE_SFT = SFTConfig(**{**SFT.__dict__, "max_steps": 40, "batch_size": 16, "eval_every": 20,
                         "val_em_examples": 64, "log_every": 5})

LR_SWEEP = (5e-5, 1e-4, 3e-4)
SEEDS = (0, 1, 2)
DATA_SIZES = (1_000, 5_000, 20_000)

SPEC = LangSpec(
    lang=LANGUAGE, model_name=MODEL_NAME, lexicon=lexicon, templates=templates,
    tokenizer_path=TOKENIZER_PATH, encoded_dir=ENCODED_DIR, pretrained=PRETRAINED,
    root=REASONING_DIR, figures_dir=FIGURES_DIR, build=BUILD, sft=SFT, smoke_sft=SMOKE_SFT,
    lr_sweep=LR_SWEEP, seeds=SEEDS, data_sizes=DATA_SIZES,
)
