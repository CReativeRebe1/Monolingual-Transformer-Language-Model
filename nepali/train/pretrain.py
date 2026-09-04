"""
Pretraining entry point for Model L (Nepali). The torchrun target.

    # local smoke run, one process
    python nepali/train/pretrain.py --smoke

    # Kaggle, both T4s
    torchrun --standalone --nproc_per_node=2 nepali/train/pretrain.py \
        --data-dir /kaggle/input/lma-encoded/nepali --out-dir /kaggle/working/nepali/checkpoints \
        --log-dir /kaggle/working/nepali/logs

Deliberately thin. Everything this file does is: pick Nepali's model config, pick Nepali's training
config, resolve paths, and hand them to the shared loop. Nothing about the architecture or the
optimisation lives here, and nothing about Hindi can reach it -- which is the mechanical form of
the assignment's independence requirement.

Resume is automatic and unconditional: if `--out-dir` already holds a loadable `ckpt_last.pt`, the
run continues from it. Re-running the identical Kaggle cell after a session expiry is therefore the
whole recovery procedure; `--fresh` is the explicit opt-out for starting over.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nepali.model.config import CONFIG as MODEL_CONFIG, MODEL_NAME, build  # noqa: E402
from nepali.train.config import CONFIG, DATA_DIR, LOG_DIR, OUT_DIR, SMOKE  # noqa: E402
from lma.train import ddp  # noqa: E402
from lma.train.loop import train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Pretrain Model L (Nepali).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="directory holding train/val/test .bin and .idx")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="checkpoint directory")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR, help="append-only log directory")
    p.add_argument("--smoke", action="store_true", help="50 tiny steps: shapes, resume, logging")
    p.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    p.add_argument("--max-steps", type=int, default=None, help="override the derived step count")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="sequences per GPU per forward. Halve this if the GPU OOMs; double "
                        "--grad-accum at the same time to keep tokens/step (and therefore the "
                        "learning-rate schedule) unchanged.")
    p.add_argument("--grad-accum", type=int, default=None, help="micro-steps per optimizer step")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (Ampere+)")
    args = p.parse_args()

    cfg = SMOKE if args.smoke else CONFIG
    if args.smoke:
        # Smoke runs ALWAYS get their own subdirectory -- including when --out-dir was passed
        # explicitly, which is the case that matters. The Kaggle runbook gives the smoke cell and
        # the real-run cell the same --out-dir, so without this the 50-step smoke checkpoint is
        # named ckpt_last.pt in the directory the real run then resumes from: the real run would
        # silently continue from a 50-step model trained at block_size 256 instead of starting
        # clean at 1024, and nothing in the loss curve would look wrong.
        # The same applies to the log: a throwaway run must never append into the permanently
        # append-only file the report's loss curve is plotted from.
        args.out_dir = args.out_dir / "smoke"
        args.log_dir = args.log_dir / "smoke"
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.micro_batch is not None:
        overrides["micro_batch_size"] = args.micro_batch
    if args.grad_accum is not None:
        overrides["grad_accum_steps"] = args.grad_accum
    if args.compile:
        overrides["compile_model"] = True
    if overrides:
        cfg = type(cfg)(**{**cfg.__dict__, **overrides})

    device = ddp.setup()
    # Captured BEFORE teardown. `ddp.teardown()` destroys the process group, after which
    # `is_distributed()` is False and `rank()` reports 0 for every process -- so every rank would
    # believe it was rank 0, and all of them would print and write summary.json. The last writer
    # wins, and a non-zero rank's summary is the wrong one. Observed live on Kaggle: rank 1 wrote
    # `best_val_loss: Infinity` over rank 0's real number.
    main = ddp.is_main()
    try:
        model = build().to(device)
        summary = train(
            model=ddp.wrap(model, device), cfg=cfg,
            data_dir=args.data_dir, out_dir=args.out_dir, log_dir=args.log_dir,
            device=device, model_label=MODEL_NAME, resume=not args.fresh,
        )
    finally:
        ddp.teardown()

    if main:
        summary["vocab_size"] = MODEL_CONFIG.vocab_size
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.log_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
