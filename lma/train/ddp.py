"""
DistributedDataParallel setup for Kaggle's 2x T4, with a single-process path that is not an
afterthought.

Every function here works when `torchrun` never ran. That is deliberate: the smoke run, the test
suite and all of evaluation execute single-process, and a training loop littered with
`if is_distributed():` branches is where rank-only bugs hide. Instead `world_size()` returns 1,
`is_main()` returns True, and `all_reduce_mean()` returns its argument -- so the loop is written
once, for the distributed case, and the single-GPU case falls out of it.

DDP rather than DataParallel: DataParallel runs one Python process driving both GPUs, replicating
the model and gathering outputs through the host every step, and the GIL makes it slower than one
GPU well before it is faster than two. DDP runs one process per GPU and overlaps the gradient
all-reduce with the backward pass itself.

Backend: NCCL on CUDA. Gloo is the fallback for CPU-only environments (the test suite), where NCCL
is not available at all.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def local_rank() -> int:
    """Which GPU on THIS machine. Equals rank() on a single node, which Kaggle always is, but the
    two are different concepts and conflating them breaks the moment there is a second node."""
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    """Rank 0. Everything that writes -- checkpoints, logs, stdout -- is gated on this, so N ranks
    do not produce N interleaved copies of the same log file."""
    return rank() == 0


def setup() -> torch.device:
    """Initialises the process group if torchrun launched us, and returns this process's device.

    `RANK` in the environment is torchrun's signal. Without it we are a plain single process, and
    initialising a one-process group would only add a barrier to every collective for no benefit.
    """
    if "RANK" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        return torch.device("cuda", local_rank())
    return torch.device("cpu")


def teardown() -> None:
    if is_distributed():
        dist.barrier()          # nobody exits while another rank is still mid-collective
        dist.destroy_process_group()


def wrap(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not is_distributed():
        return model
    device_ids = [device.index] if device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The bare module, for checkpointing. Saving the DDP wrapper writes every key with a `module.`
    prefix, and the resulting file will not load into a single-process model -- which is exactly the
    configuration evaluation runs in, so the failure surfaces only after training has finished."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Averages a Python scalar across ranks. Used for the reported loss.

    Without this, the logged loss is rank 0's loss on rank 0's shard of the data, which is a noisier
    estimate than the one the optimizer actually stepped on -- the gradients were averaged over all
    ranks, so the reported loss should be too.
    """
    if not is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size())


@contextmanager
def no_sync_unless_last(model: torch.nn.Module, is_last_micro_step: bool):
    """Suppresses DDP's gradient all-reduce on every micro-step but the last of an accumulation group.

    Without this, gradient accumulation over 4 micro-steps performs 4 full all-reduces per optimizer
    step instead of 1, and 3 of them are thrown away -- the gradients are still accumulating. On two
    T4s over PCIe that is most of the communication budget spent on nothing.
    """
    if isinstance(model, DistributedDataParallel) and not is_last_micro_step:
        with model.no_sync():
            yield
    else:
        yield
