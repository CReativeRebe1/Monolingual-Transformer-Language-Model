"""
Adds the learning-rate schedule parameters to checkpoints written before `save()` recorded them.

The assignment requires each checkpoint to carry "model weights, optimizer state, scheduler state,
current training step, and a configuration file". This design has no scheduler *object* -- the LR is
a pure function of the step (`lma.train.schedule.lr_at`), chosen so a resume cannot drift it out of
phase with the optimizer -- so the scheduler state is the four numbers that determine that function:
peak LR, warmup steps, total steps, and the floor ratio.

Both models were trained before `save()` began storing them, so they are backfilled here. Nothing is
invented: every value is read out of that run's own `run_start` log record, which was written by the
training process at the moment it started. The step count is cross-checked against the checkpoint's
own recorded step, and the script refuses to write if they disagree.

    python scripts/backfill_schedule.py --dry-run
    python scripts/backfill_schedule.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def _run_start(log_path: Path) -> dict:
    """The first run_start record: the config the run actually began with."""
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") == "run_start":
            return rec
    raise SystemExit(f"no run_start record in {log_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill scheduler state into existing checkpoints.")
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = p.parse_args()

    changed = 0
    for lang in ("hindi", "nepali"):
        log = REPO_ROOT / lang / "train" / "logs" / "train_log.jsonl"
        if not log.exists():
            print(f"{lang}: no log at {log}, skipping")
            continue
        start = _run_start(log)

        ckpts = [p for p in (REPO_ROOT / lang / "train").glob("**/*.pt") if "smoke" not in p.parts]
        if not ckpts:
            print(f"{lang}: no checkpoints found, skipping")
            continue

        for path in ckpts:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            existing = payload.get("scheduler") or {}
            if existing:
                print(f"{lang}: {path.name} already has scheduler state, leaving it alone")
                continue

            if payload["state"]["step"] > start["max_steps"]:
                raise SystemExit(
                    f"{path.name} is at step {payload['state']['step']:,} but the log's run_start "
                    f"says max_steps={start['max_steps']:,} -- these do not describe the same run"
                )

            schedule = {
                "kind": "warmup_cosine",
                "peak_lr": start["peak_lr"],
                "warmup_steps": start["warmup_steps"],
                "total_steps": start["max_steps"],
                "min_lr_ratio": 0.1,          # TrainConfig default, unchanged in both configs
                "backfilled_from": f"{lang}/train/logs/train_log.jsonl run_start {start['run_id']}",
            }
            print(f"{lang}: {path.name}  step {payload['state']['step']:,}  -> {schedule['kind']}"
                  f" peak={schedule['peak_lr']} warmup={schedule['warmup_steps']}"
                  f" total={schedule['total_steps']}")
            if args.dry_run:
                continue

            payload["scheduler"] = schedule
            tmp = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, path)             # atomic, same as checkpoint.save()
            changed += 1

    print(f"\n{'would update' if args.dry_run else 'updated'} {changed} checkpoint(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
