"""
Builds the single archive uploaded to Kaggle as a Dataset for Phase-2 pretraining.

    python scripts/pack_kaggle.py                       # -> dist/lma-phase2.zip
    python scripts/pack_kaggle.py --out D:/lma.zip --compress 0

Why one archive containing both the code and the data, rather than cloning the repo on Kaggle and
mounting the data separately: it removes the only two things that can fail before training starts
(a clone against a private repo needing credentials, and a dataset whose directory layout does not
match what the runbook assumes). One upload, one mount, one known layout.

Layout inside the archive -- Kaggle auto-extracts a .zip on upload, so this becomes the tree under
`/kaggle/input/<dataset-slug>/`:

    code/       the repo: lma/, hindi/, nepali/, scripts/, tests/, setup.py
                plus each language's SELECTED tokenizer .model/.vocab
    data/
      hindi/    train.bin train.idx val.* test.* encoding.json
      nepali/   same

`code/` is a few MB and gets copied to /kaggle/working to be writable and pip-installable; `data/`
stays where it is mounted and is read straight off /kaggle/input, which is the whole reason for
using a Dataset -- a 2.1 GB re-download at the start of every session would come out of a 12-hour
budget.

What is deliberately left out, and why it matters: `<lang>/tokenizer/sweep/` is 260 MB per language
of Phase-1 vocabulary-sweep models that nothing in Phase 2 reads, and `<lang>/data/splits/*.txt` is
2.7 GB of plaintext already represented by the encoded .bin. Including either would roughly triple
the upload for no benefit.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Code shipped to Kaggle. Everything Phase-2 training and evaluation imports, and nothing else.
CODE_DIRS = ("lma", "scripts", "tests")
CODE_FILES = ("setup.py", "requirements.txt", "README.md", "PHASE2_LOG.md",
              "notebooks/phase2_kaggle.md")
LANG_CODE_DIRS = ("model", "train", "eval", "clean")

# The selected tokenizers only -- see the module docstring on the 260 MB sweep directories.
TOKENIZER_GLOBS = ("*_unigram_16000.model", "*_unigram_16000.vocab")

ENCODED_FILES = ("train.bin", "train.idx", "val.bin", "val.idx",
                 "test.bin", "test.idx", "encoding.json")

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", "checkpoints", "logs",
                  "sweep", "raw", "interim", "shards", "splits", "encoded"}
SKIP_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc", ".bin", ".idx"}


def _iter_code_files() -> list[tuple[Path, str]]:
    """(absolute source, path inside the archive) for every code file."""
    out: list[tuple[Path, str]] = []

    for rel in CODE_FILES:
        path = REPO_ROOT / rel
        if path.exists():
            out.append((path, f"code/{rel}"))

    def walk(root: Path, prefix: str) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if SKIP_DIR_NAMES & set(path.relative_to(root).parts[:-1]):
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            out.append((path, f"{prefix}/{path.relative_to(root).as_posix()}"))

    for d in CODE_DIRS:
        walk(REPO_ROOT / d, f"code/{d}")

    for lang in ("hindi", "nepali"):
        for sub in LANG_CODE_DIRS:
            walk(REPO_ROOT / lang / sub, f"code/{lang}/{sub}")
        # __init__.py files and the data package marker, so imports resolve identically to local
        for extra in (REPO_ROOT / lang / "data" / "seed_domains.py",):
            if extra.exists():
                out.append((extra, f"code/{lang}/data/seed_domains.py"))
        for pattern in TOKENIZER_GLOBS:
            for path in sorted((REPO_ROOT / lang / "tokenizer").glob(pattern)):
                out.append((path, f"code/{lang}/tokenizer/{path.name}"))
    return out


def _iter_data_files() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for lang in ("hindi", "nepali"):
        encoded = REPO_ROOT / lang / "data" / "encoded"
        for name in ENCODED_FILES:
            path = encoded / name
            if not path.exists():
                raise SystemExit(
                    f"missing {path.relative_to(REPO_ROOT)} -- run scripts/encode_splits.py first"
                )
            out.append((path, f"data/{lang}/{name}"))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Pack code + encoded corpora for Kaggle.")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "dist" / "lma-phase2.zip")
    p.add_argument("--compress", type=int, default=1, choices=range(0, 10),
                   help="0 = store (fastest, largest). 1 (default) is the useful trade: the .bin "
                        "files are dense uint16 token ids, so higher levels cost minutes of CPU "
                        "for a couple of percent. Kaggle extracts on upload either way, so this "
                        "only affects upload time.")
    args = p.parse_args()

    entries = _iter_code_files() + _iter_data_files()
    total = sum(src.stat().st_size for src, _ in entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mode = zipfile.ZIP_STORED if args.compress == 0 else zipfile.ZIP_DEFLATED
    kwargs = {} if args.compress == 0 else {"compresslevel": args.compress}

    print(f"packing {len(entries)} files, {total / 1e9:.2f} GB raw -> {args.out}")
    written = 0
    # allowZip64 is not optional here: the archive is well past the 4 GB / 65,535-entry limits of
    # the original zip format, and without it this fails only once it is most of the way through.
    with zipfile.ZipFile(args.out, "w", mode, allowZip64=True, **kwargs) as z:
        for src, arcname in entries:
            z.write(src, arcname)
            written += src.stat().st_size
            if src.stat().st_size > 50_000_000:
                print(f"  {arcname:<32} {src.stat().st_size / 1e6:8.0f} MB "
                      f"({written / total:5.1%} done)")

    size = args.out.stat().st_size
    print(f"\nwrote {args.out}")
    print(f"  {size / 1e9:.2f} GB compressed from {total / 1e9:.2f} GB "
          f"({size / total:.0%} of raw)")
    print(f"\nUpload as a PRIVATE Kaggle Dataset. Then in the notebook the tree is:")
    print(f"  /kaggle/input/<dataset-slug>/code/")
    print(f"  /kaggle/input/<dataset-slug>/data/hindi/  and  /data/nepali/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
