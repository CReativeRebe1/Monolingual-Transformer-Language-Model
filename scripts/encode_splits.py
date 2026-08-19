"""
Encodes a language's train/val/test splits to the memmap format Phase-2 training reads.

    python scripts/encode_splits.py --lang hindi
    python scripts/encode_splits.py --lang nepali --tokenizer hindi/tokenizer/x.model

Output per split: `<lang>/data/encoded/<split>.bin` (flat uint16 token ids) and `.idx` (uint64
cumulative document offsets, starting at 0, so document i is `bin[idx[i]:idx[i+1]]`). Both are
memmap-friendly, so Phase 2 can train over a ~500M-token corpus without loading it.

The tokenizer is taken from the language's `sweep_results.json` selection unless one is passed
explicitly, so this cannot silently encode with a stale model left over from an earlier sweep.

Encoded sizes are written to `<lang>/data/encoded/encoding.json` alongside the arrays -- the token
counts there are the authoritative ones for the report, since they come from the same encoder Phase 2
will read rather than from a separate estimate.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.tokenizer.encode_corpus import encode_split, load_encoded_split  # noqa: E402
from lma.tokenizer.train_sp import Tokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.encode_splits")

SPLITS = ("train", "val", "test")


def selected_tokenizer(lang_dir: Path) -> Path:
    """The model this language's sweep actually selected."""
    results_path = lang_dir / "tokenizer" / "sweep_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found -- run scripts/tokenizer_sweep.py first, or pass "
            f"--tokenizer explicitly"
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    model = REPO_ROOT / results["final_model_path"]
    if not model.exists():
        raise FileNotFoundError(f"selected model {model} is missing")
    return model


def encode_language(language: str, tokenizer_path: Path | None = None) -> dict:
    lang_dir = REPO_ROOT / language
    model_path = tokenizer_path or selected_tokenizer(lang_dir)
    tokenizer = Tokenizer(model_path)
    logger.info("=== encoding %s with %s (vocab %d) ===",
                language, model_path.name, tokenizer.vocab_size)

    out_dir = lang_dir / "data" / "encoded"
    summary = {"language": language, "tokenizer": str(model_path.relative_to(REPO_ROOT)),
               "vocab_size": tokenizer.vocab_size, "splits": {}}

    for split in SPLITS:
        text_path = lang_dir / "data" / "splits" / f"{split}.txt"
        if not text_path.exists():
            logger.warning("%s: %s missing, skipping", language, text_path.name)
            continue
        t0 = time.time()
        info = encode_split(tokenizer, text_path, out_dir, split)
        # Read the arrays back and check the document count matches what the encoder reported.
        # The .idx file is what Phase 2 uses to find document boundaries; if it disagreed with the
        # .bin, training would silently read documents that start mid-sentence.
        tokens, offsets = load_encoded_split(info.bin_path, info.idx_path)
        if len(offsets) - 1 != info.n_documents or len(tokens) != info.n_tokens:
            raise RuntimeError(
                f"{language}/{split}: encoder wrote {info.n_documents} documents / "
                f"{info.n_tokens} tokens but the files hold {len(offsets) - 1} / {len(tokens)}"
            )
        summary["splits"][split] = {
            "documents": info.n_documents,
            "tokens": info.n_tokens,
            "bin_mb": round(info.bin_path.stat().st_size / 1e6, 1),
            "idx_mb": round(info.idx_path.stat().st_size / 1e6, 1),
            "seconds": round(time.time() - t0, 1),
        }
        logger.info("  %-5s %s documents, %s tokens -> %.0f MB in %.0fs", split,
                    f"{info.n_documents:,}", f"{info.n_tokens:,}",
                    info.bin_path.stat().st_size / 1e6, time.time() - t0)

    summary["total_tokens"] = sum(s["tokens"] for s in summary["splits"].values())
    (out_dir / "encoding.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("%s: %s tokens encoded -> %s",
                language, f"{summary['total_tokens']:,}", out_dir)
    return summary


def main():
    p = argparse.ArgumentParser(description="Encode splits to .bin/.idx for Phase-2 training.")
    p.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="override the model (default: the one sweep_results.json selected)")
    args = p.parse_args()
    print(json.dumps(encode_language(args.lang, args.tokenizer), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
