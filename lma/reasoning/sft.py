"""
The supervised-finetuning loop: masked answer loss, optional pretraining replay, val-EM selection.

A separate, small loop rather than a mode of `lma/train/loop.py`. The Phase-2 loop is graded code
built around 1024-token windows, gradient accumulation and DDP on two T4s; SFT is ~1M tokens of
short padded examples on one local GPU. Bending one into the other would touch code that already
works for a job it was not designed for. What *is* reused, unchanged: the checkpoint format
(`save`/`load`/`TrainState`, so finetuned checkpoints resume exactly like pretraining ones), the
append-only logger, the LR schedule, the parameter groups, and `TokenBatcher` for replay windows.

**Conditions.**
* `sft`    -- loss on the answer tokens only.
* `replay` -- the same, plus `replay_weight` x the ordinary LM loss on `replay_batch` pretraining
  windows drawn from the language's own `train.bin` at every step. This is InstructGPT's PPO-ptx
  idea applied to SFT: keep some gradient pointed at the original distribution so finetuning does
  not overwrite it. L9's example of what happens without it: validation perplexity 1.33 -> 194.7.

**Selection.** Checkpoints are chosen by *validation exact match*, not validation loss. The masked
loss can keep falling while greedy decoding already produces the right name, and the assignment's
metric is exact match. Early stopping: `patience` evaluations without a new best end the run.

**Dropout 0.1**, unlike pretraining's 0.0. Pretraining saw each token once; SFT runs several epochs
over 20,000 examples and *can* memorise, so the regulariser now has something to regularise. It is
set on the loaded model's modules (both residual nn.Dropout layers and the attention-probability
dropout inside CausalSelfAttention); the saved GPTConfig is left as pretrained, since dropout does
not change the architecture or the weights' meaning.

Resume is automatic: a loadable `ckpt_last.pt` in `out_dir` is continued from, with the optimizer
moments, step, sampler position, replay position, best score and patience counter all restored.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from lma.model.attention import CausalSelfAttention
from lma.reasoning.score import em_only
from lma.reasoning.sft_data import SFTDataset
from lma.train.checkpoint import TrainState, latest, load, save
from lma.train.data import TokenBatcher
from lma.train.logger import TrainLogger
from lma.train.schedule import lr_at, set_lr


@dataclass
class SFTConfig:
    """Every SFT hyperparameter. Saved in each checkpoint's `extra` and in the run_start log record."""
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_fraction: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 64
    epochs: float = 3.0
    max_steps: int = 0                 # 0 -> derived from epochs
    dropout: float = 0.1
    replay: bool = False
    replay_batch: int = 8
    replay_block: int = 256
    replay_weight: float = 1.0
    eval_every: int = 100
    val_em_examples: int = 1000
    patience: int = 4
    log_every: int = 10
    seed: int = 0
    train_size: int = 0                # 0 -> the whole train split
    eval_batch_size: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ helpers

def set_dropout(model: nn.Module, p: float) -> None:
    """Sets every dropout site in a loaded GPT to rate `p`."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
        if isinstance(m, CausalSelfAttention):
            m.dropout = p


@torch.no_grad()
def masked_val_loss(model: nn.Module, ds: SFTDataset, device, batch_size: int = 64) -> float:
    """Mean answer-token loss over a dataset (token-weighted, so long answers count per token)."""
    was = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in ds.eval_batches(batch_size):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss, _ = model(x, y)
        n = int((y != -100).sum())
        total += float(loss) * n
        count += n
    if was:
        model.train()
    return total / max(count, 1)


def device_autocast(device):
    """bf16 autocast on CUDA (the RTX 3050 is Ampere), a no-op context on CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


# ------------------------------------------------------------------ the loop

def finetune(model: nn.Module, tokenizer, train_records: list[dict], val_records: list[dict],
             cfg: SFTConfig, out_dir: Path, log_dir: Path, device: torch.device,
             replay_dir: Path | None = None, resume: bool = True, meta: dict | None = None) -> dict:
    """Finetunes `model` in place and returns a summary. Best checkpoint: `out_dir/ckpt_best.pt`."""
    out_dir, log_dir = Path(out_dir), Path(log_dir)
    meta = dict(meta or {})
    torch.manual_seed(cfg.seed)

    if cfg.train_size:
        train_records = train_records[:cfg.train_size]
    model.to(device)
    set_dropout(model, cfg.dropout)
    optimizer = torch.optim.AdamW(model.param_groups(cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  fused=device.type == "cuda")

    ds = SFTDataset(train_records, tokenizer, seed=cfg.seed)
    val_ds = SFTDataset(val_records, tokenizer, seed=0)
    val_em_records = val_records[:cfg.val_em_examples]
    total_steps = cfg.max_steps or math.ceil(cfg.epochs * len(ds) / cfg.batch_size)
    warmup = max(1, int(cfg.warmup_fraction * total_steps))
    schedule = {"peak_lr": cfg.lr, "warmup_steps": warmup, "total_steps": total_steps,
                "min_lr_ratio": cfg.min_lr_ratio}

    replay = None
    if cfg.replay:
        if replay_dir is None:
            raise ValueError("replay condition needs replay_dir (the language's encoded train split)")
        replay = TokenBatcher(Path(replay_dir) / "train.bin", Path(replay_dir) / "train.idx",
                              cfg.replay_block, cfg.replay_batch, seed=cfg.seed)

    state = TrainState()
    best_em, since_best, best_step = -1.0, 0, -1
    logger = TrainLogger(log_dir)
    ckpt = latest(out_dir) if resume else None
    if ckpt is not None:
        state, _, extra = load(ckpt, model, optimizer, map_location=device)
        best_em = extra.get("best_val_em", -1.0)
        since_best = extra.get("evals_since_best", 0)
        best_step = extra.get("best_step", -1)
        if replay is not None:
            replay.position = extra.get("replay_position", 0)
        logger.event("resume", from_step=state.step, checkpoint=str(ckpt), best_val_em=best_em)
    state.run_ids = tuple(state.run_ids) + (logger.run_id,)

    n_trainable = model.param_count()[0]
    logger.event("run_start", sft_config=cfg.to_dict(), schedule=schedule, meta=meta,
                 n_train=len(ds), n_val=len(val_ds), trainable_params=n_trainable,
                 device=str(device))
    logger.console(f"SFT {meta.get('run', '')}: {len(ds):,} examples, {total_steps:,} steps, "
                   f"lr {cfg.lr:g}, replay {cfg.replay}, "
                   f"{n_trainable:,} trainable params")

    def extra_payload() -> dict:
        return {"sft_config": cfg.to_dict(), "meta": meta, "best_val_em": best_em,
                "best_step": best_step, "evals_since_best": since_best,
                "replay_position": replay.position if replay is not None else 0}

    model.train()
    t0 = time.time()
    # A run that had already early-stopped is finished; resuming it must not train further.
    stopped_early = ckpt is not None and since_best >= cfg.patience
    step = state.step
    while step < total_steps and not stopped_early:
        set_lr(optimizer, lr_at(step, cfg.lr, warmup, total_steps, cfg.min_lr_ratio))
        x, y = ds.batch(state.sampler_position, cfg.batch_size)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with device_autocast(device):
            _, sft_loss, _ = model(x, y)
            loss = sft_loss
            replay_loss = None
            if replay is not None:
                xr, yr = replay.batch()
                _, replay_loss, _ = model(xr.to(device), yr.to(device))
                loss = sft_loss + cfg.replay_weight * replay_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]], cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        state.step = step
        state.sampler_position += cfg.batch_size
        state.tokens_seen += int((y != -100).sum())

        if step % cfg.log_every == 0 or step == 1:
            metrics = {"train_loss": round(float(sft_loss), 5), "lr": optimizer.param_groups[0]["lr"],
                       "grad_norm": round(float(grad_norm), 4),
                       "epoch": round(state.sampler_position / len(ds), 3)}
            if replay_loss is not None:
                metrics["replay_loss"] = round(float(replay_loss), 5)
            logger.step(step, **metrics)

        if step % cfg.eval_every == 0 or step == total_steps:
            val_loss = masked_val_loss(model, val_ds, device, cfg.eval_batch_size)
            val_em = em_only(model, tokenizer, val_em_records, device, cfg.eval_batch_size)
            model.train()
            improved = val_em > best_em
            if improved:
                best_em, best_step, since_best = val_em, step, 0
                state.best_val_loss = val_loss
                save(out_dir / "ckpt_best.pt", model, optimizer, None, state,
                     model.config, extra_payload(), schedule)
            else:
                since_best += 1
            save(out_dir / "ckpt_last.pt", model, optimizer, None, state, model.config,
                 extra_payload(), schedule)
            logger.step(step, val_loss=round(val_loss, 5), val_em=round(val_em, 4))
            logger.console(f"  step {step:>5}/{total_steps}  val_loss {val_loss:.4f}  "
                           f"val_em {val_em:.3f}  best {best_em:.3f}@{best_step}  "
                           f"{time.time() - t0:.0f}s")
            if since_best >= cfg.patience:
                stopped_early = True
                logger.event("early_stop", step=step, best_step=best_step, best_val_em=best_em)
                break

    summary = {"run": meta.get("run"), "steps": step, "total_steps": total_steps,
               "best_val_em": best_em, "best_step": best_step, "best_val_loss": state.best_val_loss,
               "stopped_early": stopped_early, "tokens_seen": state.tokens_seen,
               "trainable_params": n_trainable, "wall_s": round(time.time() - t0, 1),
               "sft_config": cfg.to_dict()}
    logger.event("run_end", **{k: v for k, v in summary.items() if k != "sft_config"})
    return summary
