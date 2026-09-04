"""
The pretraining loop, shared by both languages.

Language-neutral by construction: it takes a built model, a `TrainConfig`, and paths, and knows
nothing about Hindi or Nepali. `hindi/train/pretrain.py` and `nepali/train/pretrain.py` supply those
and nothing else, which is how two independent models come out of one loop without the loop being
able to mix them.

Precision note, which is hardware-specific and not a matter of taste: Kaggle's T4 is Turing (sm75).
It has fp16 tensor cores and **no bf16 support at all**. bf16 is the modern default because its
exponent range matches fp32 and loss scaling becomes unnecessary -- but requesting it here either
raises or silently runs in fp32 at a third of the speed. So this loop uses fp16 autocast with a
`GradScaler`, and logs the scaler's scale every step: a scale that collapses toward zero is the
signature of gradient overflow, and it is visible immediately rather than as a mysterious plateau.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lma.train import checkpoint as ckpt
from lma.train.data import TokenBatcher, iter_eval_windows, to_device
from lma.train.ddp import (
    all_reduce_mean, is_main, no_sync_unless_last, rank, unwrap, world_size,
)
from lma.train.logger import TrainLogger
from lma.train.schedule import lr_at, set_lr


@dataclass
class TrainConfig:
    # ---- optimisation
    peak_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_fraction: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    # beta2=0.95 rather than Adam's default 0.999: the second-moment estimate then adapts over ~20
    # steps instead of ~1000, which is what keeps a small model stable through the loss spikes that
    # a 16k-way softmax produces early in training. Every GPT-family recipe uses 0.95 for this.
    grad_clip: float = 1.0
    eps: float = 1e-8

    # ---- batch
    block_size: int = 1024
    micro_batch_size: int = 16          # sequences per GPU per forward
    grad_accum_steps: int = 4           # -> 16 * 1024 * 4 * world_size tokens per optimizer step

    # ---- length
    max_steps: int = 0                  # 0 means "derive one epoch from the corpus size"
    target_epochs: float = 1.0

    # ---- cadence
    eval_every: int = 250
    eval_batches: int = 40              # ~650k val tokens at block 1024, batch 16
    log_every: int = 10
    save_every: int = 250

    # ---- misc
    seed: int = 42
    compile_model: bool = False
    # torch.compile is off by default. It gives a solid speedup on Ampere and later, but on Turing
    # the gain is small, the first-step compile takes minutes of a limited session, and a graph
    # break in the attention branch would silently fall back anyway. Available behind a flag.
    smoke: bool = False
    extra: dict = field(default_factory=dict)

    def tokens_per_optimizer_step(self, n_ranks: int) -> int:
        return self.micro_batch_size * self.block_size * self.grad_accum_steps * n_ranks


@torch.no_grad()
def evaluate(model, bin_path: Path, idx_path: Path, cfg: TrainConfig, device) -> float:
    """Mean cross-entropy over a deterministic prefix of a held-out split.

    Deterministic on purpose: two evaluations of the same checkpoint must return the same number, or
    "val loss improved" is partly a coin flip and `ckpt_best.pt` records noise. `iter_eval_windows`
    sweeps the split in order rather than sampling it.
    """
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for x, y in iter_eval_windows(bin_path, idx_path, cfg.block_size,
                                  cfg.micro_batch_size, max_batches=cfg.eval_batches):
        x, y = to_device(x, y, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            _, loss, _ = model(x, targets=y)
        total += float(loss)
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def train(
    model: torch.nn.Module, cfg: TrainConfig, data_dir: Path, out_dir: Path, log_dir: Path,
    device: torch.device, model_label: str, resume: bool = True,
) -> dict:
    """Runs pretraining to `cfg.max_steps`, resuming from `out_dir/ckpt_last.pt` if one exists.

    Returns a summary dict. Everything durable goes to disk as it happens, so the return value is a
    convenience for the caller, never the record.
    """
    out_dir, log_dir, data_dir = Path(out_dir), Path(log_dir), Path(data_dir)
    n_ranks = world_size()
    torch.manual_seed(cfg.seed + rank())

    log = TrainLogger(log_dir, enabled=is_main())
    raw_model = unwrap(model)

    batcher = TokenBatcher(
        data_dir / "train.bin", data_dir / "train.idx",
        block_size=cfg.block_size, batch_size=cfg.micro_batch_size,
        rank=rank(), world_size=n_ranks, seed=cfg.seed,
    )

    tokens_per_step = cfg.tokens_per_optimizer_step(n_ranks)
    max_steps = cfg.max_steps or max(
        1, int(batcher.info.n_tokens * cfg.target_epochs) // tokens_per_step
    )
    warmup_steps = max(1, int(max_steps * cfg.warmup_fraction))

    optimizer = torch.optim.AdamW(
        raw_model.param_groups(cfg.weight_decay),
        lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # The four numbers that fully determine lr_at() at any step -- this design's scheduler state.
    # Stored in every checkpoint so the file is self-describing without the training config.
    schedule_state = {
        "kind": "warmup_cosine", "peak_lr": cfg.peak_lr, "warmup_steps": warmup_steps,
        "total_steps": max_steps, "min_lr_ratio": cfg.min_lr_ratio,
    }

    state = ckpt.TrainState()
    resume_path = ckpt.latest(out_dir) if resume else None
    if resume_path is not None:
        # map_location="cpu", not the training device: `load_state_dict` copies into parameters that
        # are already on the GPU, and the optimizer moves its own state to match its params, so
        # loading straight to CUDA only buys a transient second copy of the whole checkpoint in GPU
        # memory -- on a 16 GB T4 that is 286 MB of headroom given away for nothing.
        state, _, _ = ckpt.load(resume_path, model, optimizer, scaler, map_location="cpu")
        batcher.position = state.sampler_position
        log.event("resume", checkpoint=str(resume_path), step=state.step,
                  tokens_seen=state.tokens_seen, previous_sessions=len(state.run_ids))
        log.console(f"resuming {model_label} from step {state.step:,} "
                    f"({state.tokens_seen:,} tokens already seen)")
    else:
        log.console(f"starting {model_label} from scratch")

    state.run_ids = tuple(state.run_ids) + (log.run_id,)

    if cfg.compile_model:
        model = torch.compile(model)

    total_params, non_emb = raw_model.param_count()
    log.event(
        "run_start", model=model_label, run_id=log.run_id, params=total_params,
        non_embedding_params=non_emb, world_size=n_ranks, device=str(device),
        max_steps=max_steps, warmup_steps=warmup_steps, tokens_per_step=tokens_per_step,
        peak_lr=cfg.peak_lr, block_size=cfg.block_size,
        micro_batch_size=cfg.micro_batch_size, grad_accum_steps=cfg.grad_accum_steps,
        train_tokens=batcher.info.n_tokens, smoke=cfg.smoke,
    )
    log.console(
        f"{model_label}: {total_params:,} params | {n_ranks}x {device} | "
        f"{max_steps:,} steps x {tokens_per_step:,} tok/step = "
        f"{max_steps * tokens_per_step / 1e6:.0f}M tokens "
        f"({max_steps * tokens_per_step / batcher.info.n_tokens:.2f} epochs)"
    )

    model.train()
    t_step = time.time()

    for step in range(state.step, max_steps):
        lr = lr_at(step, cfg.peak_lr, warmup_steps, max_steps, cfg.min_lr_ratio)
        set_lr(optimizer, lr)

        accum_loss = 0.0
        for micro in range(cfg.grad_accum_steps):
            x, y = batcher.batch()
            x, y = to_device(x, y, device)
            with no_sync_unless_last(model, micro == cfg.grad_accum_steps - 1):
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    _, loss, _ = model(x, targets=y)
                    # Divided so the accumulated gradient is the MEAN over micro-batches, not the
                    # sum -- otherwise the effective learning rate scales with grad_accum_steps and
                    # changing the accumulation count silently changes the optimisation problem.
                    loss = loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()
            # .detach() before reading the scalar: `float(loss)` on a graph-attached tensor warns,
            # and holding the reference across the accumulation loop keeps that micro-step's graph
            # alive longer than it needs to be.
            accum_loss += loss.detach().item() * cfg.grad_accum_steps

        # Unscale before clipping: the gradients currently carry the scaler's factor, and clipping
        # them at 1.0 in that state would clip at 1.0/scale in real terms -- i.e. almost everything,
        # almost always. This ordering is the whole reason GradScaler exposes unscale_().
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        state.step = step + 1
        state.tokens_seen += tokens_per_step
        state.sampler_position = batcher.position

        if state.step % cfg.log_every == 0 or state.step == max_steps:
            dt = time.time() - t_step
            t_step = time.time()
            mean_loss = all_reduce_mean(accum_loss / cfg.grad_accum_steps, device)
            log.step(
                state.step,
                train_loss=round(mean_loss, 5),
                lr=round(lr, 8),
                grad_norm=round(grad_norm, 4),
                scaler_scale=round(float(scaler.get_scale()), 1),
                tokens=state.tokens_seen,
                tokens_per_s=round(tokens_per_step * cfg.log_every / max(dt, 1e-9)),
                gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if device.type == "cuda" else 0.0,
            )
            if not math.isfinite(mean_loss):
                log.event("diverged", step=state.step, loss=mean_loss)
                raise RuntimeError(f"loss became {mean_loss} at step {state.step}")

        is_eval_step = state.step % cfg.eval_every == 0 or state.step == max_steps
        if is_eval_step:
            # The unwrapped module, not the DDP wrapper: every rank evaluates the same deterministic
            # sweep of the val split, so there is nothing to reduce and no reason to route the
            # forward pass through DDP's reducer bookkeeping. It also means every rank computes an
            # identical val_loss, which keeps `best_val_loss` consistent across ranks -- otherwise
            # rank 0 could decide a checkpoint is "best" on a number no other rank agreed with.
            val_loss = evaluate(raw_model, data_dir / "val.bin", data_dir / "val.idx", cfg, device)
            log.step(state.step, val_loss=round(val_loss, 5),
                     val_ppl=round(math.exp(min(val_loss, 20)), 3))
            log.console(f"  step {state.step:>7,} | val loss {val_loss:.4f} "
                        f"| ppl {math.exp(min(val_loss, 20)):.2f}")
            if val_loss < state.best_val_loss:
                # `best_val_loss` is updated on EVERY rank, and only the write is gated on rank 0.
                # Gating the update too left non-zero ranks with best_val_loss = inf permanently,
                # which then leaked into the run summary and into any checkpoint a non-zero rank
                # ever wrote. Every rank computes an identical val_loss (see above), so they agree.
                state.best_val_loss = val_loss
                if is_main():
                    ckpt.save(out_dir / "ckpt_best.pt", model, optimizer, scaler, state,
                              raw_model.config, schedule=schedule_state,
                              extra={"model_label": model_label, "val_loss": val_loss})
                    log.event("checkpoint_best", step=state.step, val_loss=round(val_loss, 5))

        if (state.step % cfg.save_every == 0 or state.step == max_steps) and is_main():
            ckpt.save(out_dir / "ckpt_last.pt", model, optimizer, scaler, state,
                      raw_model.config, schedule=schedule_state,
                      extra={"model_label": model_label})
            log.event("checkpoint_last", step=state.step, tokens_seen=state.tokens_seen)

    log.event("run_end", step=state.step, tokens_seen=state.tokens_seen,
              best_val_loss=round(state.best_val_loss, 5))
    return {
        "model": model_label, "steps": state.step, "tokens_seen": state.tokens_seen,
        "best_val_loss": state.best_val_loss, "params": total_params, "run_ids": state.run_ids,
    }
