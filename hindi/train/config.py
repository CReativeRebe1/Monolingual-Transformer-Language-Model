"""
Model H (Hindi): pretraining hyperparameters.

Per-language for the same reason the cleaning thresholds were: these are claims about *this*
corpus. Hindi's training split is 515,972,223 tokens; Nepali's is 571,227,702, so at an identical
batch size the two runs are different lengths, and hard-coding one language's step count into a
shared file would quietly truncate the other.

The token budget is set by Chinchilla (Hoffmann et al. 2022): compute-optimal training for a model
of N parameters uses roughly 20N tokens. At 23,846,784 parameters that is ~477M tokens, and Hindi's
training split holds 516M -- so a single epoch is very slightly past compute-optimal and no
repetition is needed. That is a comfortable place to be: the model never sees a token twice, so
nothing in the loss curve can be memorisation, and val loss can be read as generalisation directly.
"""
from __future__ import annotations

from pathlib import Path

from lma.train.loop import TrainConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "hindi" / "data" / "encoded"
OUT_DIR = REPO_ROOT / "hindi" / "train" / "checkpoints"
LOG_DIR = REPO_ROOT / "hindi" / "train" / "logs"

TRAIN_TOKENS = 515_972_223          # hindi/data/encoded/encoding.json

CONFIG = TrainConfig(
    peak_lr=6e-4,
    # 6e-4 is the standard peak for a ~25M decoder at ~130k tokens per step. Larger models need
    # smaller peaks; this one is small enough that 1e-3 would also converge, but 6e-4 leaves margin
    # for the fp16 path, where a loss spike costs a scaler backoff and several wasted steps.
    min_lr_ratio=0.1,
    warmup_fraction=0.02,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,

    block_size=1024,
    micro_batch_size=16,            # 16 x 1024 = 16,384 tokens per GPU per forward
    grad_accum_steps=4,             # x4 x2 GPUs = 131,072 tokens per optimizer step

    max_steps=0,                    # 0 -> derived from TRAIN_TOKENS; see loop.train()
    target_epochs=1.0,

    eval_every=250,
    eval_batches=40,                # 40 x 16 x 1024 = 655,360 val tokens, ~80% of the val split
    log_every=10,
    save_every=250,                 # ~8 minutes of work at risk if the session dies
    seed=42,
)

SMOKE = TrainConfig(
    **{**CONFIG.__dict__, "max_steps": 50, "micro_batch_size": 2, "grad_accum_steps": 2,
       "block_size": 256, "eval_every": 25, "eval_batches": 4, "log_every": 5,
       "save_every": 25, "smoke": True}
)
