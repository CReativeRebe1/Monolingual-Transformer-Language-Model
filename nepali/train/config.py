"""
Model L (Nepali): pretraining hyperparameters.

Separate from Hindi's, per the project's per-language rule. The values currently coincide because
the two corpora are close in size and the two models are the same shape -- but the step count is
derived from THIS corpus (571,227,702 training tokens vs Hindi's 515,972,223), so the two runs are
different lengths, and this file is where a Nepali-specific divergence would be recorded.

At 23,846,784 parameters, Chinchilla-optimal is ~477M tokens; Nepali's split holds 571M, so a single
epoch is comfortably past compute-optimal with no repetition. As with Hindi, that means no token is
ever seen twice and val loss reads as generalisation rather than memorisation.
"""
from __future__ import annotations

from pathlib import Path

from lma.train.loop import TrainConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "nepali" / "data" / "encoded"
OUT_DIR = REPO_ROOT / "nepali" / "train" / "checkpoints"
LOG_DIR = REPO_ROOT / "nepali" / "train" / "logs"

TRAIN_TOKENS = 571_227_702          # nepali/data/encoded/encoding.json

CONFIG = TrainConfig(
    peak_lr=6e-4,
    min_lr_ratio=0.1,
    warmup_fraction=0.02,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,

    block_size=1024,
    micro_batch_size=16,
    grad_accum_steps=4,             # 131,072 tokens per optimizer step on 2 GPUs

    max_steps=0,                    # derived from TRAIN_TOKENS
    target_epochs=1.0,

    eval_every=250,
    eval_batches=40,                # 655,360 val tokens -- the Nepali val split is 634,788, so this
                                    # sweeps all of it and the final short batch is included
    log_every=10,
    save_every=250,
    seed=42,
)

SMOKE = TrainConfig(
    **{**CONFIG.__dict__, "max_steps": 50, "micro_batch_size": 2, "grad_accum_steps": 2,
       "block_size": 256, "eval_every": 25, "eval_batches": 4, "log_every": 5,
       "save_every": 25, "smoke": True}
)
