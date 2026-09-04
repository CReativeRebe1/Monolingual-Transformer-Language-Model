"""
Learning-rate schedule: linear warmup, then cosine decay to a floor.

A plain function of the step rather than a `torch.optim.lr_scheduler` object, for one practical
reason: resume. A scheduler object carries internal state that must be saved and restored in step
with the optimizer, and getting that off by one is a classic silent resume bug -- the run continues
at a learning rate a few hundred steps out of place and nothing reports it. A pure function of the
step number cannot drift: given the step, the learning rate is whatever it always was.

Warmup matters more than it looks. At initialisation the gradient is dominated by the softmax over
16,000 classes trying to flatten itself, and the resulting updates are large enough to knock a
pre-LN model into a plateau it takes thousands of steps to leave. Ramping in over the first ~2% of
training costs almost nothing and removes that failure mode.
"""
from __future__ import annotations

import math


def lr_at(
    step: int, peak_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> float:
    """Learning rate for `step` (0-indexed).

    Decays to `min_lr_ratio * peak_lr` rather than to zero: the last steps of a cosine-to-zero
    schedule take effectively no gradient steps at all, so the compute spent on them is wasted. A
    10% floor is the usual choice and keeps the tail of training useful.
    """
    if warmup_steps > 0 and step < warmup_steps:
        # step+1 so the very first step is not taken at lr=0, which would waste it entirely.
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return peak_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
