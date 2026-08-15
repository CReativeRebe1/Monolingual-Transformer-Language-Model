"""
Generates the Phase-1 report's cross-language tokenization example table: a handful of real
sentences (drawn from each language's own held-out val split), each tokenized under BOTH the
Hindi and the Nepali tokenizer -- visible, concrete proof the two vocabularies are genuinely
different piece sets (not just asserted by scripts/verify_independence.py's hash check), per the
plan's explicit requirement for "~10 side-by-side tokenization examples of the same Devanagari
sentence under both tokenizers."

Wraps lma/tokenizer/analyze.py's side_by_side_examples(), which was already implemented and
unit-tested but -- same "built, never actually called by an orchestration script" pattern found
elsewhere in this pipeline (decontamination, the manual-vs-downloaded plot, PII redaction) -- had
no script that ever generated this table for the report.

Needs BOTH languages' final tokenizers to exist (i.e. scripts/tokenizer_sweep.py already run for
both), which is why this is its own script rather than folded into tokenizer_sweep.py itself --
that script operates one language at a time.

Usage:
    python scripts/tokenizer_comparison.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.tokenizer.analyze import side_by_side_examples  # noqa: E402
from lma.tokenizer.train_sp import Tokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.tokenizer_comparison")

N_EXAMPLES_PER_LANGUAGE = 5  # -> up to 10 total sentences (5 Hindi + 5 Nepali), per the plan


def _load_final_tokenizer(language: str) -> Tokenizer:
    sweep_results_path = REPO_ROOT / language / "tokenizer" / "sweep_results.json"
    if not sweep_results_path.exists():
        raise FileNotFoundError(
            f"{sweep_results_path} not found; run scripts/tokenizer_sweep.py --lang {language} first"
        )
    sweep_results = json.loads(sweep_results_path.read_text(encoding="utf-8"))
    model_path = REPO_ROOT / sweep_results["final_model_path"]
    return Tokenizer(model_path)


def _sample_sentences(language: str, n: int) -> list[str]:
    """Pulls up to n non-trivial sentences from a language's own held-out val split (falling
    back to train if val is empty, matching tokenizer_sweep.py's own fallback logic)."""
    splits_dir = REPO_ROOT / language / "data" / "splits"
    val_path = splits_dir / "val.txt"
    train_path = splits_dir / "train.txt"
    lines = []
    if val_path.exists() and val_path.stat().st_size > 0:
        lines = [l.strip() for l in val_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines and train_path.exists():
        lines = [l.strip() for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[:n]


def build_table() -> dict:
    hindi_tokenizer = _load_final_tokenizer("hindi")
    nepali_tokenizer = _load_final_tokenizer("nepali")

    hindi_sentences = _sample_sentences("hindi", N_EXAMPLES_PER_LANGUAGE)
    nepali_sentences = _sample_sentences("nepali", N_EXAMPLES_PER_LANGUAGE)
    all_sentences = hindi_sentences + nepali_sentences

    if not all_sentences:
        raise RuntimeError(
            "no sentences available from either language's splits; run scripts/build_corpus.py "
            "for both languages first"
        )

    examples = side_by_side_examples(hindi_tokenizer, nepali_tokenizer, all_sentences)

    output = {
        "hindi_tokenizer": str(hindi_tokenizer.model_path.relative_to(REPO_ROOT)),
        "nepali_tokenizer": str(nepali_tokenizer.model_path.relative_to(REPO_ROOT)),
        "examples": [
            {
                "text": ex["text"],
                "hindi_tokenizer_pieces": ex["pieces_a"],
                "hindi_tokenizer_n_tokens": ex["n_tokens_a"],
                "nepali_tokenizer_pieces": ex["pieces_b"],
                "nepali_tokenizer_n_tokens": ex["n_tokens_b"],
            }
            for ex in examples
        ],
    }
    out_path = REPO_ROOT / "report" / "tables" / "tokenizer_comparison_examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %d examples to %s", len(examples), out_path)
    return output


def main():
    result = build_table()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
