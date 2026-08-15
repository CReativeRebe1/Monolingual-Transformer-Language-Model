"""
Reads a language's current dataset/stats.json (written by build_corpus.py) and prints how many
downloaded-source tokens to pull to approach the ~500M-token target while staying safely above
the assignment's >=20%-manual floor. A thin CLI over lma/text/corpus_budget.py's pure calculation
-- does not itself pull anything (see that module's docstring for why the pull is deliberately a
separate, later, deliberate step).

Approximates tokens as word count (a whitespace-token estimate; the real tokenizer's fertility,
from scripts/tokenizer_sweep.py's sweep_results.json, refines this once a tokenizer is chosen --
see the --fertility flag).

Usage:
    python scripts/corpus_budget.py --lang hindi
    python scripts/corpus_budget.py --lang hindi --fertility 1.31  # convert words -> real tokens
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.text.corpus_budget import recommend_pull  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Recommend a downloaded-token pull size against the current manual corpus."
    )
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument(
        "--fertility", type=float, default=None,
        help="tokens/word ratio from scripts/tokenizer_sweep.py's sweep_results.json, to convert "
             "the word-count estimate to real tokenizer tokens. Default: treat words as tokens "
             "(a conservative 1:1 estimate; real fertility is typically >1, so this UNDER-counts "
             "manual tokens and is therefore the safer default in the direction of pulling less).",
    )
    parser.add_argument("--target-total-tokens", type=int, default=500_000_000)
    args = parser.parse_args()

    stats_path = REPO_ROOT / args.lang / "data" / "stats.json"
    if not stats_path.exists():
        print(f"ERROR: {stats_path} does not exist yet -- run build_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    manual_words = stats.get("corpus", {}).get("n_words")
    if manual_words is None:
        print(f"ERROR: {stats_path} has no corpus.n_words field.", file=sys.stderr)
        sys.exit(1)

    fertility = args.fertility if args.fertility is not None else 1.0
    manual_tokens = int(manual_words * fertility)

    rec = recommend_pull(manual_tokens, target_total_tokens=args.target_total_tokens)

    print(json.dumps({
        "language": args.lang,
        "manual_words": manual_words,
        "fertility_used": fertility,
        "manual_tokens_estimate": rec.manual_tokens,
        "recommended_downloaded_tokens": rec.recommended_downloaded_tokens,
        "resulting_total_tokens": rec.resulting_total_tokens,
        "resulting_manual_fraction": round(rec.resulting_manual_fraction, 4),
        "limited_by": rec.limited_by,
    }, indent=2))


if __name__ == "__main__":
    main()
