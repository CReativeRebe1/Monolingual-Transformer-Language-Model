"""
Pretraining entry point for the bonus NoPE ablation of Model H. The torchrun target.

    # local smoke run, one process
    python hindi/nope/pretrain.py --smoke

    # Kaggle, both T4s -- same command shape as hindi/train/pretrain.py
    torchrun --standalone --nproc_per_node=2 hindi/nope/pretrain.py \
        --data-dir /kaggle/input/.../data/hindi --out-dir /kaggle/working/out/nope/checkpoints \
        --log-dir /kaggle/working/out/nope/logs

Identical to `hindi/train/pretrain.py` except for the model it builds (`use_rope=False`) and where it
writes, so the only difference between this run and Model H's Phase-2 run is position information.
Resume is automatic from `ckpt_last.pt`; `--fresh` opts out; smoke runs are isolated in `smoke/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hindi.nope.config import (  # noqa: E402
    CONFIG as MODEL_CONFIG, LOG_DIR, MODEL_NAME, OUT_DIR, SMOKE_CONFIG, TRAIN_CONFIG, build,
)
from hindi.train.config import DATA_DIR  # noqa: E402
from lma.train import ddp  # noqa: E402
from lma.train.loop import train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Pretrain Model H without positional embeddings (bonus).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR)
    p.add_argument("--smoke", action="store_true", help="50 tiny steps: shapes, resume, logging")
    p.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--micro-batch", type=int, default=None,
                   help="halve on OOM and double --grad-accum, keeping tokens per step unchanged")
    p.add_argument("--grad-accum", type=int, default=None)
    args = p.parse_args()

    cfg = SMOKE_CONFIG if args.smoke else TRAIN_CONFIG
    if args.smoke:
        args.out_dir = args.out_dir / "smoke"
        args.log_dir = args.log_dir / "smoke"
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.micro_batch is not None:
        overrides["micro_batch_size"] = args.micro_batch
    if args.grad_accum is not None:
        overrides["grad_accum_steps"] = args.grad_accum
    if overrides:
        cfg = type(cfg)(**{**cfg.__dict__, **overrides})

    device = ddp.setup()
    main_rank = ddp.is_main()
    if main_rank:
        print(f"{MODEL_NAME}: use_rope={MODEL_CONFIG.use_rope} (positional embeddings removed)")
    try:
        model = build().to(device)
        summary = train(
            model=ddp.wrap(model, device), cfg=cfg,
            data_dir=args.data_dir, out_dir=args.out_dir, log_dir=args.log_dir,
            device=device, model_label=MODEL_NAME, resume=not args.fresh,
        )
    finally:
        ddp.teardown()

    if main_rank:
        summary["vocab_size"] = MODEL_CONFIG.vocab_size
        summary["use_rope"] = MODEL_CONFIG.use_rope
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.log_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
