"""
Evaluation entry point for Model H (Hindi): perplexity, bits-per-byte, generation metrics, diversity.

    python hindi/eval/evaluate.py --checkpoint hindi/train/checkpoints/ckpt_best.pt
    python hindi/eval/evaluate.py --checkpoint ... --split test     # once, at the very end

Runs on CPU or a single GPU -- no Kaggle session needed after training finishes.

**The split default is `val`, deliberately.** Every number used to make a decision (which checkpoint,
which decoding settings) comes from validation. `--split test` is run exactly once, after all of
that is settled, and its output is what the report quotes. A test set consulted while tuning is a
validation set with a misleading name.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from hindi.model.config import ENCODED_DIR, MODEL_NAME, TOKENIZER_PATH  # noqa: E402
from lma.eval.diversity import diversity_report  # noqa: E402
from lma.eval.generate import DECODING_GRID, REQUIRED_GRID, continuations  # noqa: E402
from lma.eval.perplexity import evaluate_split  # noqa: E402
from lma.eval.textmetrics import all_metrics  # noqa: E402
from lma.train.checkpoint import load_model  # noqa: E402
from lma.tokenizer.train_sp import Tokenizer  # noqa: E402
from lma.viz.plots import plot_diversity_grid  # noqa: E402

RESULTS_DIR = REPO_ROOT / "hindi" / "eval" / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate Model H (Hindi).")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "hindi/train/checkpoints/ckpt_best.pt")
    p.add_argument("--data-dir", type=Path, default=ENCODED_DIR)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap the perplexity sweep (default: the whole split)")
    p.add_argument("--gen-docs", type=int, default=200, help="held-out documents to continue")
    p.add_argument("--gen-batch", type=int, default=16,
                   help="prompts generated in parallel. Lower it on a small GPU: generation runs "
                        "without a KV cache, so the activation cost grows with the sequence as it "
                        "extends, and 7 decoding settings run back to back.")
    p.add_argument("--prefix-tokens", type=int, default=128)
    p.add_argument("--continuation-tokens", type=int, default=128)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="where result JSON goes (the NoPE bonus writes to hindi/nope/results)")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--label", default=MODEL_NAME, help="model name used in results and plot titles")
    p.add_argument("--figure-prefix", default="hindi", help="figure filename prefix")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state, _ = load_model(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    tokenizer = Tokenizer(TOKENIZER_PATH)

    print(f"{args.label}: checkpoint step {state.step:,}, {model.param_count()[0]:,} params, "
          f"evaluating on {args.split}")

    bin_path = args.data_dir / f"{args.split}.bin"
    idx_path = args.data_dir / f"{args.split}.idx"

    ppl = evaluate_split(
        model, tokenizer, bin_path, idx_path,
        block_size=model.config.max_seq_len, batch_size=args.batch_size,
        device=device, max_batches=args.max_batches, split_name=args.split,
    )
    print(f"  perplexity {ppl.perplexity:9.3f} | bits/byte {ppl.bits_per_byte:.4f} "
          f"| {ppl.n_tokens:,} tokens over {ppl.n_bytes:,} bytes")

    results = {
        "model": args.label, "language": "hindi", "split": args.split,
        "checkpoint": str(args.checkpoint), "checkpoint_step": state.step,
        "params": model.param_count()[0], "vocab_size": model.config.vocab_size,
        "perplexity": ppl.to_dict(),
    }

    if not args.skip_generation:
        # Recorded because the sampled numbers are only reproducible at the same batch size:
        # `continuations()` seeds once per call, and the batch shape determines how many values
        # each multinomial draw consumes, so a different --gen-batch walks a different random
        # stream. Same model, same seed, slightly different chrF. Storing the protocol here means
        # the report's table can be regenerated exactly.
        results["generation_protocol"] = {
            "n_docs": args.gen_docs, "prefix_tokens": args.prefix_tokens,
            "continuation_tokens": args.continuation_tokens, "gen_batch": args.gen_batch,
            "seed": 42, "bleu_ngram_order": 4,
        }
        results["generation"] = {}
        for setting in DECODING_GRID:
            name = setting["name"]
            kwargs = {k: v for k, v in setting.items() if k != "name"}
            conts = continuations(
                model, tokenizer, bin_path, idx_path, n_docs=args.gen_docs,
                prefix_tokens=args.prefix_tokens, continuation_tokens=args.continuation_tokens,
                batch_size=args.gen_batch, device=device, **kwargs,
            )
            # Released between settings: seven decoding configurations run back to back, each
            # allocating progressively longer sequences, and the caching allocator will otherwise
            # hold every setting's peak at once. Costs nothing; prevents an OOM on a small GPU.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            generated = [c.generated for c in conts]
            entry = {
                "decoding": kwargs,
                # Marks the four settings the assignment names (greedy, T=0.5, 1.0, 1.5) so the
                # report can table them on their own rather than mixed with the nucleus extras.
                "assignment_required": any(s["name"] == name for s in REQUIRED_GRID),
                **all_metrics(generated, [c.reference for c in conts]),
                **diversity_report(generated),
            }
            results["generation"][name] = entry
            print(f"  {name:<16} chrF {entry['chrf']:6.2f} | BLEU {entry['bleu']:5.2f} "
                  f"| ROUGE-L {entry['rouge_l']:5.2f} | distinct-2 {entry['distinct_2']:.3f} "
                  f"| self-BLEU {entry['self_bleu']:5.2f}")

            if name == "t0.5":
                # One representative setting's raw generations are kept for the report's qualitative
                # section. Keeping all five would be five times the file for no extra insight.
                args.results_dir.mkdir(parents=True, exist_ok=True)
                (args.results_dir / "samples.json").write_text(
                    json.dumps([vars(c) for c in conts[:25]], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        names = list(results["generation"])
        plot_diversity_grid(
            names,
            [results["generation"][n]["distinct_2"] for n in names],
            [results["generation"][n]["self_bleu"] for n in names],
            args.figures_dir / f"{args.figure_prefix}_diversity_grid.png", args.label,
        )

    out = args.out or args.results_dir / f"metrics_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
