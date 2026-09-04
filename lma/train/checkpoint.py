"""
Checkpoint save/load, written to survive Kaggle killing the session mid-write.

Kaggle's free tier stops a notebook at 12 hours, and a session can also die on an idle timeout or a
browser disconnect. Whenever it happens, it happens without warning and without a signal the process
can catch -- so the guarantee this module has to provide is not "save on exit" but "whatever is on
disk is always a loadable checkpoint".

That is what the atomic write is for. `torch.save` straight to `ckpt_last.pt` spends a second or two
streaming 286 MB to disk; a kill in that window leaves a truncated file that `torch.load` refuses,
and the previous good checkpoint is already gone. Writing to `ckpt_last.pt.tmp` and then calling
`os.replace` means the visible file switches from the old checkpoint to the new one in a single
filesystem operation -- there is no instant at which it is neither.

What is saved is everything needed to make the resumed run indistinguishable from the run that was
killed:

* model weights (DDP-unwrapped -- see `ddp.unwrap` for why the prefix matters)
* optimizer state -- AdamW's first and second moments are two thirds of the file, and dropping them
  restarts the optimizer cold, which shows up as a visible loss spike at the resume point
* GradScaler state -- the fp16 loss scale takes hundreds of steps to find its level again
* step, tokens seen, and the sampler position, so no window is repeated or skipped
* RNG states for torch, numpy and CUDA, so dropout and sampling continue rather than restart
* the model config, so a checkpoint can be loaded without the code that built it agreeing by luck
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lma.model.transformer import GPT, GPTConfig
from lma.train.ddp import unwrap


@dataclass
class TrainState:
    """Everything about a run that is not weights or optimizer moments."""
    step: int = 0
    tokens_seen: int = 0
    sampler_position: int = 0
    best_val_loss: float = float("inf")
    run_ids: tuple[str, ...] = ()          # one per session; a resumed run appends rather than replaces


def _rng_states() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(states: dict[str, Any] | None) -> None:
    """RNG states are ByteTensors and both setters require them on the CPU.

    The `.cpu()` calls are not defensive padding -- without them the first resume raises
    `TypeError: RNG state must be a torch.ByteTensor`, because `torch.load(..., map_location=cuda)`
    faithfully moves EVERY tensor in the payload to the GPU, RNG states included. Caught by the
    local smoke run; on Kaggle it would have surfaced only when a session was resumed, i.e. after
    hours of training had already been done.
    """
    if not states:
        return
    torch.set_rng_state(states["torch"].cpu())
    np.random.set_state(states["numpy"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        # Only restorable when the resumed session has the same number of GPUs. A 2-GPU run resumed
        # on 1 GPU is a legitimate thing to do on Kaggle when the second card is unavailable, so
        # this degrades rather than raising.
        cuda_states = [s.cpu() for s in states["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def save(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scaler: Any, state: TrainState, config: GPTConfig, extra: dict | None = None,
    schedule: dict | None = None,
) -> Path:
    """Atomically writes a checkpoint to `path`. Returns the path written.

    `schedule` records the learning-rate schedule's parameters. There is no scheduler *object* to
    serialise here -- `lma.train.schedule.lr_at` is a pure function of the step, chosen precisely so
    a resume cannot drift it out of phase with the optimizer. But a checkpoint still has to be
    self-describing: given only this file, the LR at any step must be recoverable, and that needs
    the peak, the warmup length, the total length and the floor. Those four numbers ARE the
    scheduler state for this design.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": schedule or {},
        "state": vars(state),
        "config": config.to_dict(),
        "rng": _rng_states(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    # os.replace is atomic on both POSIX and Windows (unlike os.rename on Windows, which fails if
    # the destination exists). The previous checkpoint survives intact until this line completes.
    os.replace(tmp, path)
    return path


def load(
    path: Path, model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None, scaler: Any = None,
    map_location: str | torch.device = "cpu", restore_rng: bool = True,
) -> tuple[TrainState, GPTConfig, dict]:
    """Restores into whichever of model/optimizer/scaler are provided. Returns (state, config, extra).

    `weights_only=False` is required because the payload holds RNG states and a config dict, not
    just tensors. This is our own file written by our own code on the same machine, which is the
    case the flag exists to distinguish from loading an untrusted checkpoint off the internet.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        unwrap(model).load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        _restore_rng(payload.get("rng"))
    return TrainState(**payload["state"]), GPTConfig(**payload["config"]), payload.get("extra", {})


def load_model(path: Path, map_location: str | torch.device = "cpu") -> tuple[GPT, TrainState, dict]:
    """Builds a fresh GPT from the checkpoint's own config and loads the weights into it.

    Used by evaluation, which has no reason to know which language's config module produced the
    file -- the checkpoint is self-describing, so `lma/eval/` stays language-neutral.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = GPTConfig(**payload["config"])
    model = GPT(config)
    model.load_state_dict(payload["model"])
    return model, TrainState(**payload["state"]), payload.get("extra", {})


def latest(out_dir: Path) -> Path | None:
    """`ckpt_last.pt` if it exists and loads. Returns None if there is nothing to resume from.

    The load probe is not paranoia: a checkpoint written by a session that ran out of *disk* rather
    than time can pass the exists() check and still be unreadable, and discovering that after the
    training loop has already reported "resuming from step 3,250" is worse than discovering it here.
    """
    candidate = Path(out_dir) / "ckpt_last.pt"
    if not candidate.exists():
        return None
    try:
        torch.load(candidate, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return candidate
