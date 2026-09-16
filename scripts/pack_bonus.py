"""
Archives for the bonus NoPE ablation: the code needed to run it, and the results it produces.

    python scripts/pack_bonus.py code                     # -> dist/lma-bonus-code.zip
    python scripts/pack_bonus.py results                  # -> dist/lma-bonus-results.zip
    python scripts/pack_bonus.py results --root /kaggle/working/out --out /kaggle/working/lma-bonus-results.zip

**code** is uploaded to Kaggle as its own small Dataset, next to the Phase-2 `lma-phase2` Dataset
that already holds the encoded Hindi corpus -- so the 1.6 GB of data is not uploaded again. It
contains only what the ablation imports:

    code/lma/{__init__,model,train,eval,tokenizer,viz}   architecture, loop, metrics, plots
    code/hindi/{model,train,eval,nope}/                  Hindi config, Phase-2 eval entry points, ablation
    code/hindi/tokenizer/hindi_unigram_16000.{model,vocab}
    code/hindi/eval/results/*.json                       the standard model's Phase-2 results
    code/hindi/train/logs/{train_log.jsonl,summary.json} the standard model's training record
    code/hindi/nope/results/position_probe_model_h.json  the standard model's position probe
    code/tests/test_model.py, test_train.py, test_eval.py
    code/setup.py, requirements.txt, notebooks/bonus_kaggle.md, scripts/pack_bonus.py

Nothing from Nepali, the crawler, the cleaning pipeline or Phase 3 is included: the ablation needs none
of it, and an archive that contains only the bonus's dependencies is itself a check that the bonus
does not depend on anything else.

**results** collects whatever the run produced: the NoPE training logs, every result JSON
(metrics, attention statistics, position probes, the comparison) and the figures.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CODE_DIRS = ("lma/model", "lma/train", "lma/eval", "lma/tokenizer", "lma/viz",
             "hindi/model", "hindi/train", "hindi/eval", "hindi/nope")
CODE_FILES = ("lma/__init__.py", "setup.py", "requirements.txt",
              "notebooks/bonus_kaggle.md", "scripts/pack_bonus.py",
              "tests/__init__.py", "tests/conftest.py", "tests/test_model.py", "tests/test_train.py",
              "tests/test_eval.py",
              "hindi/tokenizer/hindi_unigram_16000.model", "hindi/tokenizer/hindi_unigram_16000.vocab")
CODE_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".model", ".vocab", ".txt"}
SKIP_PARTS = {"__pycache__", "checkpoints", "smoke", ".pytest_cache"}
# Log files are data, not code: keep the standard model's committed record, drop anything else.
KEEP_LOGS = {"hindi/train/logs/train_log.jsonl", "hindi/train/logs/summary.json"}


def code_entries() -> list[tuple[Path, str]]:
    """(source, archive path) for every file the bonus needs to run."""
    out: dict[str, Path] = {}
    for rel in CODE_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing {rel}")
        out[rel] = path
    for d in CODE_DIRS:
        for path in sorted((REPO_ROOT / d).rglob("*")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if not path.is_file() or path.suffix not in CODE_SUFFIXES or SKIP_PARTS & set(path.parts):
                continue
            if "/logs/" in f"/{rel}" and rel not in KEEP_LOGS:
                continue
            out[rel] = path
    return [(src, f"code/{rel}") for rel, src in sorted(out.items())]


def results_entries(root: Path) -> list[tuple[Path, str]]:
    """Every log, result and figure under `root` (a repo checkout, or Kaggle's output directory)."""
    patterns = {
        "logs": ["hindi/nope/logs/train_log.jsonl", "hindi/nope/logs/train.log", "hindi/nope/logs/summary.json",
                 "nope/logs/train_log.jsonl", "nope/logs/train.log", "nope/logs/summary.json"],
        "results": ["hindi/nope/results/*.json", "hindi/nope/results/*.md", "nope/results/*.json", "nope/results/*.md"],
        "figures": ["report/figures/hindi_nope_*.png", "figures/hindi_nope_*.png"],
    }
    out = []
    for group, globs in patterns.items():
        for pattern in globs:
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    out.append((path, f"{group}/{path.name}"))
    return out


def write(entries: list[tuple[Path, str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in entries:
            z.write(src, arc)
    total = sum(s.stat().st_size for s, _ in entries)
    print(f"wrote {dest}: {len(entries)} files, {total / 1e6:.2f} MB raw, {dest.stat().st_size / 1e6:.2f} MB zipped")


def main() -> int:
    p = argparse.ArgumentParser(description="Pack the bonus ablation's code or results.")
    p.add_argument("what", choices=["code", "results"])
    p.add_argument("--root", type=Path, default=REPO_ROOT, help="results: directory to collect from")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.what == "code":
        write(code_entries(), args.out or REPO_ROOT / "dist" / "lma-bonus-code.zip")
    else:
        entries = results_entries(args.root)
        if not entries:
            raise SystemExit(f"no results found under {args.root}")
        write(entries, args.out or REPO_ROOT / "dist" / "lma-bonus-results.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
