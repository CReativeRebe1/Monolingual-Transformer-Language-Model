"""
Model H vs Model L on the reasoning task -- the side-by-side table §3 asks for.

    python scripts/phase3_compare.py

Reads each language's `<lang>/reasoning/results/summary.json` (written by `evaluate.py --summarize`)
and writes `report/phase3_comparison.md` plus a grouped bar figure. Reads results only; it never loads
a checkpoint, so it cannot mix the two models' weights or tokenizers.

The two models are compared on the *same design* (task mix, split construction, template families,
seeds), not the same sentences: each language's data was generated in its own words. Differences
therefore combine model quality and how hard each language's phrasing is for its own tokenizer,
which the report discusses rather than hides.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.reasoning.build import TEST_SPLITS  # noqa: E402
from lma.viz.plots import plot_accuracy_bars  # noqa: E402

LABELS = {"hindi": "Model H (Hindi)", "nepali": "Model L (Nepali)"}


def load(lang: str) -> dict:
    return json.loads((REPO_ROOT / lang / "reasoning" / "results" / "summary.json").read_text(encoding="utf-8"))


def best_sft(table: dict) -> str:
    cands = [g for g in table if g.startswith("sft_lr") and "_n" not in g and "val" in table[g]["splits"]]
    return max(cands, key=lambda g: table[g]["splits"]["val"]["em"][0])


def ms(x) -> str:
    return f"{x[0]:.3f} ± {x[1]:.3f}" if x and x[1] else (f"{x[0]:.3f}" if x else "-")


def main() -> int:
    tables = {lang: load(lang) for lang in LABELS}
    rows = []
    for lang, t in tables.items():
        sft = best_sft(t)
        lr = sft.split("_lr")[1]
        conds = {"pretrained 0-shot": "pretrained_0shot", "pretrained 3-shot": "pretrained_3shot",
                 "SFT": sft, "SFT + replay": f"replay_lr{lr}"}
        for name, g in conds.items():
            if g in t:
                rows.append((lang, name, g, t[g]))

    lines = ["# Phase 3 — Model H vs Model L on comparative reasoning", "",
             "Mean ± std over seeds. EM = generative exact match; cand = candidate-choice accuracy; "
             "PPL = Phase-2 validation perplexity (forgetting).", "",
             "| model | condition | seeds | " + " | ".join(f"EM {s}" for s in TEST_SPLITS) +
             " | cand test_both | PPL |", "|---|---|---|" + "---|" * (len(TEST_SPLITS) + 2)]
    for lang, name, g, row in rows:
        sp = row["splits"]
        em = [ms(sp[s]["em"]) if s in sp else "-" for s in TEST_SPLITS]
        cand = ms(sp["test_both"].get("cand_acc")) if "test_both" in sp else "-"
        lines.append(f"| {LABELS[lang]} | {name} | {row['n_seeds']} | " + " | ".join(em) +
                     f" | {cand} | {ms(row.get('val_ppl'))} |")

    lines += ["", "## Exact match by task type (SFT, test_both)", "",
              "| model | cmp2 | cmp3 | ext3 | pair3 | ext4 |", "|---|---|---|---|---|---|"]
    for lang, t in tables.items():
        bt = t[best_sft(t)]["splits"]["test_both"]["by_task_em"]
        lines.append(f"| {LABELS[lang]} | " + " | ".join(ms(bt.get(k)) for k in ("cmp2", "cmp3", "ext3", "pair3", "ext4")) + " |")

    out = REPO_ROOT / "report" / "phase3_comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    series, errors = {}, {}
    for lang, t in tables.items():
        g = best_sft(t)
        series[f"{LABELS[lang]} SFT"] = [t[g]["splits"][s]["em"][0] for s in TEST_SPLITS]
        errors[f"{LABELS[lang]} SFT"] = [t[g]["splits"][s]["em"][1] for s in TEST_SPLITS]
    plot_accuracy_bars(list(TEST_SPLITS), series, REPO_ROOT / "report" / "figures" / "phase3_h_vs_l_em.png",
                       "Model H vs Model L: exact match after SFT (mean ± std over 3 seeds)",
                       errors=errors,
                       reference=tables["hindi"]["pretrained_0shot"]["splits"]["test_id"]["chance"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
