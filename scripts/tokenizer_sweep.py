"""
Tokenizer vocabulary-size sweep: trains BPE and Unigram tokenizers at {8k, 16k, 24k, 32k, 48k}
for one language on its train split, evaluates each on the held-out val split (fertility,
chars/token, UNK rate, vocab utilization), and selects a final tokenizer per the assignment's
stated criterion -- "fertility / unknown-token rate on held-out text".

Since byte_fallback=True makes UNK rate exactly 0% at every vocab size (see
lma/tokenizer/train_sp.py's docstring), selection in practice is driven by fertility: the
smallest vocab size past which fertility stops improving by a meaningful margin (an elbow
criterion), so the final model is not needlessly large relative to the ~25M-parameter budget
Phase 2 imposes -- vocabulary size directly sets embedding-table size at that budget.

Usage:
    python scripts/tokenizer_sweep.py --lang hindi
    python scripts/tokenizer_sweep.py --lang nepali
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.tokenizer.analyze import evaluate_tokenizer  # noqa: E402
from lma.tokenizer.train_sp import Tokenizer, TokenizerTrainConfig, train_tokenizer  # noqa: E402
from lma.viz.plots import plot_fertility_vs_vocab_size  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.tokenizer_sweep")

VOCAB_SIZES = (8_000, 16_000, 24_000, 32_000, 48_000)
ALGORITHMS = ("bpe", "unigram")
ELBOW_IMPROVEMENT_THRESHOLD = 0.03  # stop growing vocab once fertility improves <3% per step


def _elbow_vocab_size(sweep_by_vocab: dict[int, float]) -> int:
    """Smallest vocab size past which fertility improvement falls below the elbow threshold."""
    sizes = sorted(sweep_by_vocab.keys())
    if len(sizes) == 1:
        return sizes[0]
    for i in range(1, len(sizes)):
        prev_fertility = sweep_by_vocab[sizes[i - 1]]
        curr_fertility = sweep_by_vocab[sizes[i]]
        if prev_fertility <= 0:
            continue
        improvement = (prev_fertility - curr_fertility) / prev_fertility
        if improvement < ELBOW_IMPROVEMENT_THRESHOLD:
            return sizes[i - 1]
    return sizes[-1]  # fertility kept improving meaningfully all the way to the largest size


def run_sweep(language: str) -> dict:
    lang_dir = REPO_ROOT / language
    train_path = lang_dir / "data" / "splits" / "train.txt"
    val_path = lang_dir / "data" / "splits" / "val.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found; run scripts/build_corpus.py --lang {language} first"
        )

    val_texts = []
    if val_path.exists() and val_path.stat().st_size > 0:
        val_texts = [l.strip() for l in val_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not val_texts:
        # Corpus too small yet to have a non-empty val split (e.g. early in the crawl); fall
        # back to a sample of train text for the sweep's own metrics, and note this explicitly
        # in the results so it is never confused with a real held-out evaluation.
        logger.warning(
            "no val split available yet for %s; sweep metrics use a TRAIN sample "
            "(re-run once the corpus is large enough for a real held-out val split)",
            language,
        )
        train_lines = [l.strip() for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        val_texts = train_lines[: max(1, len(train_lines) // 10)]
        used_train_as_val = True
    else:
        used_train_as_val = False

    sweep_dir = lang_dir / "tokenizer" / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []
    fertility_by_algo_vocab: dict[str, dict[int, float]] = {algo: {} for algo in ALGORITHMS}

    for algo in ALGORITHMS:
        for vocab_size in VOCAB_SIZES:
            out_prefix = sweep_dir / f"{language}_{algo}_{vocab_size}"
            config = TokenizerTrainConfig(algorithm=algo, vocab_size=vocab_size)
            logger.info("training %s vocab_size=%d ...", algo, vocab_size)
            try:
                model_path, _ = train_tokenizer([train_path], out_prefix, config)
            except RuntimeError as e:
                logger.warning("skipping %s vocab_size=%d: %s", algo, vocab_size, e)
                continue
            tokenizer = Tokenizer(model_path)
            stats = evaluate_tokenizer(tokenizer, val_texts, algorithm=algo)
            fertility_by_algo_vocab[algo][vocab_size] = stats.fertility
            results.append(
                {
                    "algorithm": algo,
                    "vocab_size": vocab_size,
                    "fertility": stats.fertility,
                    "chars_per_token": stats.chars_per_token,
                    "unk_rate": stats.unk_rate,
                    "vocab_utilization": stats.vocab_utilization,
                }
            )
            logger.info(
                "  fertility=%.3f chars/token=%.3f unk_rate=%.4f vocab_util=%.3f",
                stats.fertility, stats.chars_per_token, stats.unk_rate, stats.vocab_utilization,
            )

    # Select per-algorithm elbow vocab size, then pick the algorithm with the lower fertility
    # at its own elbow point (the assignment leaves BPE vs Unigram choice open; fertility on
    # held-out text is the stated selection criterion for vocab size AND is used here for
    # algorithm choice too, for consistency).
    best_overall = None
    for algo in ALGORITHMS:
        if not fertility_by_algo_vocab[algo]:
            continue
        elbow_size = _elbow_vocab_size(fertility_by_algo_vocab[algo])
        elbow_fertility = fertility_by_algo_vocab[algo][elbow_size]
        if best_overall is None or elbow_fertility < best_overall[2]:
            best_overall = (algo, elbow_size, elbow_fertility)

    if best_overall is None:
        raise RuntimeError(f"tokenizer sweep produced no successful runs for {language}")

    best_algo, best_vocab_size, best_fertility = best_overall
    logger.info("selected: algorithm=%s vocab_size=%d fertility=%.3f", best_algo, best_vocab_size, best_fertility)

    # Copy the selected sweep model into the final tokenizer/ location (the actual Phase-1
    # deliverable path), rather than leaving it buried in sweep/.
    final_dir = lang_dir / "tokenizer"
    src_model = sweep_dir / f"{language}_{best_algo}_{best_vocab_size}.model"
    src_vocab = sweep_dir / f"{language}_{best_algo}_{best_vocab_size}.vocab"
    dst_model = final_dir / f"{language}_{best_algo}_{best_vocab_size}.model"
    dst_vocab = final_dir / f"{language}_{best_algo}_{best_vocab_size}.vocab"
    dst_model.write_bytes(src_model.read_bytes())
    dst_vocab.write_bytes(src_vocab.read_bytes())

    fig_dir = REPO_ROOT / "report" / "figures"
    plot_fertility_vs_vocab_size(
        {f"{language}-{algo}": list(sizes.items()) for algo, sizes in fertility_by_algo_vocab.items() if sizes},
        fig_dir / f"{language}_tokenizer_fertility_sweep.png",
    )

    output = {
        "language": language,
        "used_train_as_val": used_train_as_val,
        "sweep_results": results,
        "selected": {"algorithm": best_algo, "vocab_size": best_vocab_size, "fertility": best_fertility},
        "final_model_path": str(dst_model.relative_to(REPO_ROOT)),
    }
    results_path = final_dir / "sweep_results.json"
    results_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", results_path)
    return output


def main():
    parser = argparse.ArgumentParser(description="Sweep tokenizer vocab size/algorithm for one language.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    args = parser.parse_args()
    result = run_sweep(args.lang)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
