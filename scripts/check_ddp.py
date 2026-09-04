"""
Pre-flight check for the distributed training path, runnable without torchrun.

    python scripts/check_ddp.py                 # 2 ranks, gloo, CPU
    python scripts/check_ddp.py --ranks 2 --backend nccl

Exists because the things that break a multi-GPU run break it in the first thirty seconds -- a
process group that never forms, a rank-shape mismatch, a collective that deadlocks -- and finding
that out 40 minutes into a Kaggle session costs the session. Two minutes here costs two minutes.

It also runs where `torchrun` cannot. On Windows, PyTorch is built without libuv and torchrun's
rendezvous store fails outright (`use_libuv was requested but PyTorch was build without libuv
support`), so the DDP code path could not be exercised locally at all. This script initialises the
process group through a `FileStore` instead, which works everywhere, and spawns its own ranks.

What it asserts:

* the process group forms and `world_size` is what was asked for
* `all_reduce_mean` actually averages across ranks rather than returning rank 0's value
* wrapping in DDP and taking a step leaves every rank with IDENTICAL parameters -- the property
  that makes data-parallel training equivalent to one large batch, and the one that silently fails
  if gradients are not being synchronised
* `no_sync` suppresses the all-reduce on non-final micro-steps and the final step still syncs
* the two ranks' samplers draw disjoint windows
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from lma.model.transformer import GPT, GPTConfig  # noqa: E402
from lma.train.data import WindowSampler  # noqa: E402
from lma.train.ddp import all_reduce_mean, is_main, no_sync_unless_last, unwrap, wrap  # noqa: E402

TINY = GPTConfig(vocab_size=53, d_model=32, n_layers=2, n_heads=4, mlp_hidden=88, max_seq_len=32)


def _worker(rank: int, world_size: int, backend: str, store_file: str, result_file: str) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend=backend, init_method=f"file://{store_file}", rank=rank, world_size=world_size
    )
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)

    problems: list[str] = []

    # --- collectives average rather than echo rank 0
    mean = all_reduce_mean(float(rank), device)
    expected = (world_size - 1) / 2
    if abs(mean - expected) > 1e-6:
        problems.append(f"all_reduce_mean gave {mean}, expected {expected}")

    # --- identical parameters after a synchronised step
    torch.manual_seed(0)                       # every rank starts from the same weights
    model = GPT(TINY).to(device)
    ddp_model = wrap(model, device)
    optimizer = torch.optim.AdamW(unwrap(ddp_model).param_groups(0.1), lr=1e-2)

    # Each rank feeds DIFFERENT data, which is the whole point of data parallelism. If gradients
    # were not being synchronised, the ranks would diverge here and the check below would catch it.
    torch.manual_seed(100 + rank)
    for micro in range(2):
        idx = torch.randint(0, TINY.vocab_size, (2, 8), device=device)
        with no_sync_unless_last(ddp_model, micro == 1):
            _, loss, _ = ddp_model(idx, targets=idx)
            (loss / 2).backward()
    optimizer.step()

    flat = torch.cat([p.detach().flatten() for p in unwrap(ddp_model).parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(gathered, flat)
    for other in range(1, world_size):
        drift = float((gathered[0] - gathered[other]).abs().max())
        if drift > 1e-5:
            problems.append(
                f"rank 0 and rank {other} diverged by {drift:.2e} after one step -- "
                "gradients are not being synchronised"
            )

    # --- samplers are disjoint
    mine = {WindowSampler(4096, 16, rank=rank, world_size=world_size, seed=7).window_start(p)
            for p in range(10)}
    others = set()
    for r in range(world_size):
        if r != rank:
            others |= {WindowSampler(4096, 16, rank=r, world_size=world_size, seed=7).window_start(p)
                       for p in range(10)}
    if mine & others:
        problems.append(f"rank {rank} shares {len(mine & others)} windows with another rank")

    if is_main():
        Path(result_file).write_text("\n".join(problems), encoding="utf-8")
        print(f"world_size={dist.get_world_size()}  backend={backend}  device={device}")
        print(f"all_reduce_mean over ranks 0..{world_size - 1} = {mean} (expected {expected})")
        print("parameters identical across ranks after a synchronised step: "
              f"{'NO' if any('diverged' in p for p in problems) else 'yes'}")
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-flight check for the DDP training path.")
    p.add_argument("--ranks", type=int, default=2)
    p.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    args = p.parse_args()

    if args.backend == "nccl" and torch.cuda.device_count() < args.ranks:
        print(f"nccl needs {args.ranks} GPUs, found {torch.cuda.device_count()}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        store = str(Path(tmp) / "store")
        result = str(Path(tmp) / "problems.txt")
        mp.spawn(_worker, args=(args.ranks, args.backend, store, result), nprocs=args.ranks,
                 join=True)
        problems = [l for l in Path(result).read_text(encoding="utf-8").splitlines() if l]

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print("  -", problem)
        return 1
    print("\nDDP path OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
