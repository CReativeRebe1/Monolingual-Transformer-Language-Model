"""
Attention diagnostics for Model L (Nepali): entropy, mean distance, head taxonomy, heatmaps.

    python nepali/eval/attention_report.py --checkpoint nepali/train/checkpoints/ckpt_best.pt

Writes `nepali/eval/results/attention_stats.json` and three figures under `report/figures/`.
CPU-runnable; no Kaggle session needed.

Extraction runs at a short context (default 192 tokens) with batch 1, because
`GPT.forward(output_attentions=True)` materialises a [B, H, T, T] tensor per layer -- 25 MB at
T=1024, so 250 MB across ten layers, and that is before the backward graph. 192 tokens is ample:
head roles are visible in a couple of hundred tokens, and the heatmap labels stay legible.

Statistics are averaged over several held-out documents rather than read off one, because a "local"
verdict drawn from a single input that happens to be a list of names is a fact about the input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nepali.model.config import ENCODED_DIR, MODEL_NAME, TOKENIZER_PATH  # noqa: E402
from lma.eval.attention import aggregate, analyse, layer_means, role_counts, save_stats  # noqa: E402
from lma.eval.generate import sample_documents  # noqa: E402
from lma.train.checkpoint import load_model  # noqa: E402
from lma.tokenizer.train_sp import Tokenizer  # noqa: E402
from lma.viz.plots import (  # noqa: E402
    plot_attention_by_layer, plot_attention_heatmap, plot_head_taxonomy,
)

RESULTS_DIR = REPO_ROOT / "nepali" / "eval" / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
LANG = "nepali"


def main() -> int:
    p = argparse.ArgumentParser(description="Attention diagnostics for Model L (Nepali).")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "nepali/train/checkpoints/ckpt_best.pt")
    p.add_argument("--data-dir", type=Path, default=ENCODED_DIR)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--context", type=int, default=192, help="tokens per extraction pass")
    p.add_argument("--docs", type=int, default=8, help="held-out documents to average over")
    p.add_argument("--heatmap-context", type=int, default=48,
                   help="shorter window for the heatmaps, so token labels stay legible")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state, _ = load_model(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    tokenizer = Tokenizer(TOKENIZER_PATH)

    bin_path = args.data_dir / f"{args.split}.bin"
    idx_path = args.data_dir / f"{args.split}.idx"
    docs = sample_documents(bin_path, idx_path, args.docs, args.context, seed=42)
    print(f"{MODEL_NAME}: step {state.step:,} | {len(docs)} documents x {args.context} tokens")

    runs = []
    for _, doc in docs:
        idx = torch.from_numpy(doc[:args.context].astype(np.int64))[None].to(device)
        runs.append(analyse(model, idx))
    stats = aggregate(runs)

    per_layer = layer_means(stats)
    counts = role_counts(stats)
    save_stats(stats, RESULTS_DIR / "attention_stats.json", extra={
        "model": MODEL_NAME, "language": LANG, "checkpoint_step": state.step,
        "context_tokens": args.context, "n_documents": len(docs),
        "layer_means": {str(k): v for k, v in per_layer.items()},
    })

    print("\n  layer  entropy(norm)  distance  prev-token")
    for layer, m in per_layer.items():
        print(f"  {layer:>5}  {m['entropy_normalised']:>13.3f}  "
              f"{m['mean_distance']:>8.2f}  {m['prev_token_mass']:>10.3f}")
    print("\n  head roles:", ", ".join(f"{k}={v}" for k, v in counts.items()))

    # ---- figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_head_taxonomy(
        [s.mean_distance for s in stats], [s.entropy_normalised for s in stats],
        [s.layer for s in stats], [s.role for s in stats],
        FIGURES_DIR / f"{LANG}_head_taxonomy.png", MODEL_NAME,
    )
    layers = sorted(per_layer)
    plot_attention_by_layer(
        layers, [per_layer[l]["entropy_normalised"] for l in layers],
        [per_layer[l]["mean_distance"] for l in layers],
        FIGURES_DIR / f"{LANG}_attention_by_layer.png", MODEL_NAME,
    )

    # Heatmaps on a short window so the Devanagari token labels stay readable. An early, a middle
    # and a late layer, and within each layer SEVERAL heads rather than one: a single head cannot
    # show that heads within a layer specialise differently, which is the point of plotting them.
    # The heads chosen per layer are the most-focused, the most-diffuse, and the one with the
    # highest previous-token mass -- i.e. the extremes of the taxonomy, where the contrast is
    # visible, instead of three heads that happen to look alike.
    _, doc = docs[0]
    short = torch.from_numpy(doc[:args.heatmap_context].astype(np.int64))[None].to(device)
    labels = [tokenizer._sp.id_to_piece(int(t)).replace("▁", " ")
              for t in doc[:args.heatmap_context]]
    with torch.no_grad():
        _, _, attentions = model(short, output_attentions=True)
    n_layers = model.config.n_layers
    n_plotted = 0
    for layer in sorted({0, n_layers // 2, n_layers - 1}):
        by_head = [s for s in stats if s.layer == layer]
        chosen = {
            min(by_head, key=lambda s: s.entropy_normalised).head,
            max(by_head, key=lambda s: s.entropy_normalised).head,
            max(by_head, key=lambda s: s.prev_token_mass).head,
        }
        for head in sorted(chosen):
            s = next(h for h in by_head if h.head == head)
            plot_attention_heatmap(
                attentions[layer][0, head].float().cpu().numpy(), labels,
                FIGURES_DIR / f"{LANG}_attn_L{layer}H{head}.png",
                f"{MODEL_NAME}: layer {layer}, head {head} "
                f"({s.role.replace('_', ' ')}, distance {s.mean_distance:.1f}, "
                f"entropy {s.entropy_normalised:.2f})",
            )
            n_plotted += 1
    print(f"  plotted {n_plotted} heatmaps across layers "
          f"{sorted({0, n_layers // 2, n_layers - 1})}")
    print(f"\nwrote figures to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
