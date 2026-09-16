"""
Standard Model H vs the NoPE ablation: every Phase-2 metric side by side, plus the position probes.

    python hindi/nope/compare.py

Reads only result files -- no checkpoint, no GPU:

    standard  hindi/train/logs/train_log.jsonl, hindi/eval/results/{metrics_test,attention_stats}.json,
              hindi/nope/results/position_probe_model_h.json
    NoPE      hindi/nope/logs/train_log.jsonl,  hindi/nope/results/{metrics_test,attention_stats}.json,
              hindi/nope/results/position_probe_model_h_nope.json

Writes `hindi/nope/results/comparison.{md,json}` and comparison figures (`<prefix>_vs_standard_*.png`).
Every input path can be overridden, so the same script runs on Kaggle against /kaggle/working outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hindi.nope.config import (  # noqa: E402
    BASELINE_LOG, BASELINE_NAME, BASELINE_RESULTS_DIR, FIGURE_PREFIX, FIGURES_DIR, LOG_DIR, MODEL_NAME,
    RESULTS_DIR,
)
from lma.train.logger import loss_curves, read_log  # noqa: E402
from lma.viz.plots import (  # noqa: E402
    plot_layer_compare, plot_loss_by_position, plot_loss_curves_compare, plot_metric_bars,
)

GEN_METRICS = ("bleu", "chrf", "rouge_l", "distinct_2", "self_bleu", "repetition_rate_4gram", "mean_length_words")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    p = argparse.ArgumentParser(description="Compare standard Model H with its NoPE ablation.")
    p.add_argument("--baseline-results", type=Path, default=BASELINE_RESULTS_DIR)
    p.add_argument("--baseline-log", type=Path, default=BASELINE_LOG)
    p.add_argument("--nope-results", type=Path, default=RESULTS_DIR)
    p.add_argument("--nope-log", type=Path, default=LOG_DIR / "train_log.jsonl")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    std_m = _load(args.baseline_results / f"metrics_{args.split}.json")
    nope_m = _load(args.nope_results / f"metrics_{args.split}.json")
    if std_m is None or nope_m is None:
        raise SystemExit(f"need metrics_{args.split}.json for both models "
                         f"({args.baseline_results}, {args.nope_results})")
    std_a = _load(args.baseline_results / "attention_stats.json")
    nope_a = _load(args.nope_results / "attention_stats.json")
    std_p = _load(args.nope_results / f"position_probe_{BASELINE_NAME}.json")
    nope_p = _load(args.nope_results / f"position_probe_{MODEL_NAME}.json")
    std_log, nope_log = read_log(args.baseline_log), read_log(args.nope_log)

    fig = args.figures_dir
    labels = {"std": f"{BASELINE_NAME} (RoPE)", "nope": f"{MODEL_NAME} (no positions)"}
    out: dict = {"split": args.split, "models": labels}

    # ---- language modelling
    sp, np_ = std_m["perplexity"], nope_m["perplexity"]
    out["perplexity"] = {"standard": sp, "nope": np_,
                         "ppl_ratio": np_["perplexity"] / sp["perplexity"],
                         "bpb_delta": np_["bits_per_byte"] - sp["bits_per_byte"]}

    # ---- training
    def final(records: list[dict]) -> dict:
        steps = [r for r in records if r.get("kind") == "step"]
        vals = [r for r in steps if r.get("val_loss") is not None]
        trains = [r for r in steps if "train_loss" in r]
        return {"steps": steps[-1]["step"] if steps else 0,
                "final_train_loss": trains[-1]["train_loss"] if trains else None,
                "final_val_loss": vals[-1]["val_loss"] if vals else None,
                "val_curve": [(r["step"], r["val_loss"]) for r in vals]}
    out["training"] = {"standard": final(std_log), "nope": final(nope_log)}
    if std_log and nope_log:
        plot_loss_curves_compare({labels["std"]: loss_curves(std_log), labels["nope"]: loss_curves(nope_log)},
                                 fig / f"{FIGURE_PREFIX}_vs_standard_loss.png",
                                 "Model H: pretraining loss with and without positional embeddings")

    # ---- generation and diversity
    settings = [s for s in std_m["generation"] if s in nope_m["generation"]]
    out["generation"] = {s: {m: {"standard": std_m["generation"][s].get(m), "nope": nope_m["generation"][s].get(m)}
                             for m in GEN_METRICS} for s in settings}
    for metric, ylabel in (("chrf", "chrF"), ("distinct_2", "distinct-2"), ("repetition_rate_4gram", "4-gram repetition rate")):
        plot_metric_bars(settings, {labels["std"]: [std_m["generation"][s][metric] for s in settings],
                                    labels["nope"]: [nope_m["generation"][s][metric] for s in settings]},
                         fig / f"{FIGURE_PREFIX}_vs_standard_{metric}.png",
                         f"Model H vs NoPE: {ylabel} by decoding setting ({args.split})",
                         "Decoding setting", ylabel)

    # ---- attention
    if std_a and nope_a:
        layers = sorted(int(k) for k in std_a["layer_means"])
        out["attention"] = {"role_counts": {"standard": std_a["role_counts"], "nope": nope_a["role_counts"]},
                            "layer_means": {"standard": std_a["layer_means"], "nope": nope_a["layer_means"]}}
        for metric, ylabel in (("entropy_normalised", "Normalised attention entropy"),
                               ("mean_distance", "Mean attention distance (tokens)"),
                               ("prev_token_mass", "Previous-token attention mass")):
            plot_layer_compare(layers, {labels["std"]: [std_a["layer_means"][str(l)][metric] for l in layers],
                                        labels["nope"]: [nope_a["layer_means"][str(l)][metric] for l in layers]},
                               fig / f"{FIGURE_PREFIX}_vs_standard_{metric}.png",
                               f"Model H vs NoPE: {ylabel.lower()} by layer", ylabel)

    # ---- position probes
    if std_p and nope_p:
        out["position"] = {"buckets": {"standard": std_p["loss_by_position_buckets"],
                                       "nope": nope_p["loss_by_position_buckets"]},
                           "order_sensitivity": {"standard": std_p["order_sensitivity"],
                                                 "nope": nope_p["order_sensitivity"]}}
        plot_loss_by_position({labels["std"]: std_p["loss_by_position"], labels["nope"]: nope_p["loss_by_position"]},
                              fig / f"{FIGURE_PREFIX}_vs_standard_loss_by_position.png",
                              "Model H vs NoPE: next-token loss by position in the window (val)")
        plot_metric_bars(["intact context", "shuffled context"],
                         {labels["std"]: [std_p["order_sensitivity"]["intact_nll"], std_p["order_sensitivity"]["shuffled_context_nll"]],
                          labels["nope"]: [nope_p["order_sensitivity"]["intact_nll"], nope_p["order_sensitivity"]["shuffled_context_nll"]]},
                         fig / f"{FIGURE_PREFIX}_vs_standard_order.png",
                         "Continuation loss after an intact vs shuffled first half (val, 256-token windows)",
                         "Context condition", "Mean continuation loss (nats)")

    args.nope_results.mkdir(parents=True, exist_ok=True)
    (args.nope_results / "comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(out)
    (args.nope_results / "comparison.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def markdown(out: dict) -> str:
    """The comparison tables for the report."""
    L = ["# Bonus ablation — Model H with and without positional embeddings", "",
         f"Split: **{out['split']}**. Standard = Phase-2 Model H (RoPE); NoPE = identical model, data, "
         "tokenizer and training configuration with RoPE removed.", "",
         "## Language modelling", "", "| | standard | NoPE | change |", "|---|---:|---:|---:|"]
    s, n = out["perplexity"]["standard"], out["perplexity"]["nope"]
    L.append(f"| Perplexity | {s['perplexity']:.2f} | {n['perplexity']:.2f} | ×{out['perplexity']['ppl_ratio']:.3f} |")
    L.append(f"| Bits per byte | {s['bits_per_byte']:.4f} | {n['bits_per_byte']:.4f} | {out['perplexity']['bpb_delta']:+.4f} |")
    L.append(f"| Bits per token | {s['bits_per_token']:.3f} | {n['bits_per_token']:.3f} | {n['bits_per_token'] - s['bits_per_token']:+.3f} |")
    t = out["training"]
    if t["standard"]["final_val_loss"] is not None and t["nope"]["final_val_loss"] is not None:
        L.append(f"| Final validation loss (training log) | {t['standard']['final_val_loss']:.4f} | "
                 f"{t['nope']['final_val_loss']:.4f} | {t['nope']['final_val_loss'] - t['standard']['final_val_loss']:+.4f} |")
        L.append(f"| Steps | {t['standard']['steps']:,} | {t['nope']['steps']:,} | |")
        sv, nv = dict(t["standard"]["val_curve"]), dict(t["nope"]["val_curve"])
        common = sorted(set(sv) & set(nv))
        if common:
            L += ["", "### Validation loss during training", "", "| step | standard | NoPE | gap |", "|---:|---:|---:|---:|"]
            for step in common:
                L.append(f"| {step:,} | {sv[step]:.4f} | {nv[step]:.4f} | {nv[step] - sv[step]:+.4f} |")

    L += ["", "## Generation and diversity", "",
          "| setting | chrF std / NoPE | BLEU std / NoPE | ROUGE-L std / NoPE | distinct-2 std / NoPE | self-BLEU std / NoPE | 4-gram repetition std / NoPE |",
          "|---|---|---|---|---|---|---|"]
    for setting, m in out["generation"].items():
        def pair(k, f="{:.2f}"):
            a, b = m[k]["standard"], m[k]["nope"]
            return f"{f.format(a)} / {f.format(b)}" if a is not None and b is not None else "-"
        L.append(f"| {setting} | {pair('chrf')} | {pair('bleu')} | {pair('rouge_l')} | {pair('distinct_2', '{:.3f}')} | "
                 f"{pair('self_bleu')} | {pair('repetition_rate_4gram', '{:.3f}')} |")

    if "attention" in out:
        a = out["attention"]
        L += ["", "## Attention by layer", "",
              "| layer | entropy std | entropy NoPE | distance std | distance NoPE | prev-token std | prev-token NoPE |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for layer in sorted(a["layer_means"]["standard"], key=int):
            s, n = a["layer_means"]["standard"][layer], a["layer_means"]["nope"][layer]
            L.append(f"| {layer} | {s['entropy_normalised']:.3f} | {n['entropy_normalised']:.3f} | {s['mean_distance']:.2f} | "
                     f"{n['mean_distance']:.2f} | {s['prev_token_mass']:.3f} | {n['prev_token_mass']:.3f} |")
        L += ["", f"Head roles — standard: {a['role_counts']['standard']}; NoPE: {a['role_counts']['nope']}"]

    if "position" in out:
        pos = out["position"]
        L += ["", "## Position diagnostics (validation)", "", "| positions | loss std | loss NoPE | gap |", "|---|---:|---:|---:|"]
        for s, n in zip(pos["buckets"]["standard"], pos["buckets"]["nope"]):
            L.append(f"| {s['from']}–{s['to'] - 1} | {s['mean_nll']:.4f} | {n['mean_nll']:.4f} | {n['mean_nll'] - s['mean_nll']:+.4f} |")
        so, no = pos["order_sensitivity"]["standard"], pos["order_sensitivity"]["nope"]
        L += ["", "| order sensitivity (256-token windows) | standard | NoPE |", "|---|---:|---:|",
              f"| continuation loss, intact context | {so['intact_nll']:.4f} | {no['intact_nll']:.4f} |",
              f"| continuation loss, shuffled context | {so['shuffled_context_nll']:.4f} | {no['shuffled_context_nll']:.4f} |",
              f"| increase from shuffling | {so['delta_nll']:+.4f} | {no['delta_nll']:+.4f} |"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
