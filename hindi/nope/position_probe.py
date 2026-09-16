"""
Position diagnostics for one checkpoint: loss by position and sensitivity to context order.

    python hindi/nope/position_probe.py --checkpoint hindi/nope/checkpoints/ckpt_best.pt --label model_h_nope
    python hindi/nope/position_probe.py --checkpoint hindi/train/hindi_ckpt_best.pt --label model_h

Run on both models (the committed `results/position_probe_model_h.json` is the standard model's) and
read side by side by `compare.py`. See `lma/eval/position.py` for what each measurement means.
Uses the validation split, so the test split stays reserved for the headline metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from hindi.nope.config import ENCODED_DIR, RESULTS_DIR  # noqa: E402
from lma.eval.position import bucket_positions, loss_by_position, order_sensitivity  # noqa: E402
from lma.train.checkpoint import load_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Loss by position and order sensitivity for one checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--data-dir", type=Path, default=ENCODED_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=40,
                   help="40 x 8 x 1024 = 327,680 validation tokens, ~40%% of the split")
    p.add_argument("--order-block", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state, _ = load_model(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    bin_path, idx_path = args.data_dir / "val.bin", args.data_dir / "val.idx"
    print(f"{args.label}: step {state.step:,}, use_rope={model.config.use_rope}")

    losses = loss_by_position(model, bin_path, idx_path, model.config.max_seq_len, args.batch_size,
                              device, args.max_batches)
    order = order_sensitivity(model, bin_path, idx_path, args.order_block, args.batch_size * 4,
                              device, args.max_batches)
    result = {
        "label": args.label, "checkpoint": str(args.checkpoint), "checkpoint_step": state.step,
        "use_rope": model.config.use_rope, "split": "val",
        "windows_for_position_curve": args.max_batches * args.batch_size,
        "loss_by_position": [round(float(v), 5) for v in losses],
        "loss_by_position_buckets": bucket_positions(losses),
        "order_sensitivity": order,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"position_probe_{args.label}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for b in result["loss_by_position_buckets"]:
        print(f"  positions {b['from']:>4}-{b['to'] - 1:<4}  nll {b['mean_nll']:.4f}")
    print(f"  order: intact {order['intact_nll']:.4f}  shuffled context {order['shuffled_context_nll']:.4f}  "
          f"delta {order['delta_nll']:+.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
