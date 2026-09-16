"""
Runs every Phase-3 experiment for one language, in order, skipping whatever is already done.

    python scripts/run_phase3.py --lang hindi
    python scripts/run_phase3.py --lang nepali --only sweep main

Stages (each is a list of the per-language entry-point commands, so any stage can also be reproduced
by hand from the README):

    data       generate.py (skipped if the JSONL files exist)
    baselines  pretrained zero-shot and 3-shot
    sweep      SFT at every LR in LR_SWEEP, seed 0, scored on **val only**
    main       winning LR: SFT seeds 1-2 (seed 0 is the sweep run), SFT+replay seeds 0-2, all splits
    size       data-size curve at the winning LR (the 20k point is the main run)
    report     --summarize, then the attention comparison on SFT + replay, seed 0

**The learning rate is chosen on validation exact match alone.** Sweep runs are evaluated on `val`
only; the winner is then re-evaluated on every split. Test numbers therefore never exist for a
configuration while it is still a candidate, so they cannot leak into the choice.

Done-ness is file-based: a training run is done when its `logs/<run>/summary.json` exists, an
evaluation when `results/eval/<label>.json` exists and covers the requested splits. An interrupted
training run is resumed by the finetune script itself.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.reasoning.build import TEST_SPLITS  # noqa: E402
from lma.reasoning.cli import run_name  # noqa: E402

STAGES = ("data", "baselines", "sweep", "main", "size", "report")


def sh(*args: str) -> None:
    """Runs one entry-point command, streaming its output, and stops everything if it fails."""
    cmd = [sys.executable, *args]
    print(">>", " ".join(args), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def evaluated(spec, label: str, splits: list[str]) -> bool:
    path = spec.results_dir / "eval" / f"{label}.json"
    if not path.exists():
        return False
    have = json.loads(path.read_text(encoding="utf-8"))["splits"]
    return all(s in have for s in splits)


def trained(spec, run: str) -> bool:
    return (spec.log_dir / run / "summary.json").exists()


def train_and_eval(spec, lang: str, run: str, train_args: list[str], splits: list[str], ppl: bool) -> None:
    if not trained(spec, run):
        sh(f"{lang}/reasoning/finetune.py", *train_args)
    if not evaluated(spec, run, splits):
        sh(f"{lang}/reasoning/evaluate.py", "--run", run, "--splits", *splits, *([] if ppl else ["--no-ppl"]))


def best_lr(spec) -> float:
    """The sweep LR with the highest validation exact match (ties: the smaller LR)."""
    scores = {}
    for lr in spec.lr_sweep:
        summary = spec.log_dir / run_name("sft", lr, 0) / "summary.json"
        scores[lr] = json.loads(summary.read_text(encoding="utf-8"))["best_val_em"]
    chosen = max(sorted(scores), key=lambda lr: scores[lr])
    print(f"LR sweep (val EM): {scores} -> {chosen:g}", flush=True)
    return chosen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument("--only", nargs="+", choices=STAGES, default=list(STAGES))
    args = p.parse_args()
    lang = args.lang
    spec = import_module(f"{lang}.reasoning.config").SPEC
    all_splits = ["val", *TEST_SPLITS]

    if "data" in args.only and not (spec.data_dir / "train.jsonl").exists():
        sh(f"{lang}/reasoning/generate.py")

    if "baselines" in args.only:
        if not evaluated(spec, "pretrained_0shot", all_splits):
            sh(f"{lang}/reasoning/evaluate.py", "--pretrained", "--shots", "0")
        if not evaluated(spec, "pretrained_3shot", all_splits):
            sh(f"{lang}/reasoning/evaluate.py", "--pretrained", "--shots", "3", "--no-ppl")

    if "sweep" in args.only:
        for lr in spec.lr_sweep:
            run = run_name("sft", lr, 0)
            train_and_eval(spec, lang, run, ["--condition", "sft", "--lr", str(lr), "--seed", "0"], ["val"], ppl=False)

    later = {"main", "size", "report"} & set(args.only)
    if not later:
        return 0
    lr = best_lr(spec)

    if "main" in args.only:
        for seed in spec.seeds:
            train_and_eval(spec, lang, run_name("sft", lr, seed),
                           ["--condition", "sft", "--lr", str(lr), "--seed", str(seed)], all_splits, ppl=True)
        for seed in spec.seeds:
            train_and_eval(spec, lang, run_name("replay", lr, seed),
                           ["--condition", "replay", "--lr", str(lr), "--seed", str(seed)], all_splits, ppl=True)

    if "size" in args.only:
        for n in spec.data_sizes:
            if n >= 20_000:
                continue                      # the full-data point is the main seed-0 run
            train_and_eval(spec, lang, run_name("sft", lr, 0, train_size=n),
                           ["--condition", "sft", "--lr", str(lr), "--seed", "0", "--train-size", str(n)],
                           all_splits, ppl=False)

    if "report" in args.only:
        sh(f"{lang}/reasoning/evaluate.py", "--summarize")
        sh(f"{lang}/reasoning/attention_compare.py", "--run", run_name("replay", lr, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
