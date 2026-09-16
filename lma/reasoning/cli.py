"""
Shared command implementations behind each language's reasoning entry points.

`hindi/reasoning/{generate,finetune,evaluate,attention_compare}.py` and their Nepali twins are thin:
they build a `LangSpec` from their own config and hand it here -- the same split as Phase 2, where
`hindi/train/pretrain.py` picks Hindi's configs and calls the shared loop. Nothing in this file names
a language; everything language-specific arrives through the spec, so a Hindi run cannot read a
Nepali lexicon, tokenizer, corpus or checkpoint.

Run naming encodes the full identity of a run, so a directory name is enough to know what it was:

    sft_lr0.0001_s0            SFT, peak LR 1e-4, seed 0
    replay_lr0.0001_s1         SFT + pretraining replay, seed 1
    sft_lr0.0001_n1000_s0      data-size curve point: first 1,000 training examples

This is also what lets the learning-rate sweep double as the seed-0 main run: the best sweep run *is*
`sft_lr<best>_s0`, not a retrained copy of it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import torch

from lma.reasoning.build import (
    SPLITS, TEST_SPLITS, BuildConfig, build_dataset, chance_accuracy, dataset_stats, heuristic_accuracy,
    read_jsonl, write_jsonl,
)
from lma.reasoning.score import ERROR_TYPES, evaluate_records
from lma.reasoning.sft import SFTConfig, finetune
from lma.eval.perplexity import evaluate_split
from lma.tokenizer.train_sp import Tokenizer
from lma.train.checkpoint import load_model
from lma.train.logger import read_log


@dataclass
class LangSpec:
    """Everything one language contributes to Phase 3."""
    lang: str
    model_name: str
    lexicon: ModuleType
    templates: ModuleType
    tokenizer_path: Path
    encoded_dir: Path
    pretrained: Path
    root: Path                     # <lang>/reasoning
    figures_dir: Path
    build: BuildConfig
    sft: SFTConfig
    smoke_sft: SFTConfig
    lr_sweep: tuple[float, ...]
    seeds: tuple[int, ...]
    data_sizes: tuple[int, ...]

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def ckpt_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def known_names(self) -> set[str]:
        """Every entity name in this language's lexicon (for the out-of-prompt error class)."""
        return {n for n, _ in self.lexicon.PERSONS} | {n for n, _ in self.lexicon.OBJECTS}

    def split(self, name: str) -> list[dict]:
        """Loads one generated split; tells the user how to generate it if it is missing."""
        path = self.data_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run {self.lang}/reasoning/generate.py first")
        return read_jsonl(path)


def device_default() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_name(condition: str, lr: float, seed: int, train_size: int = 0) -> str:
    """Canonical directory name for a run (see module docstring)."""
    parts = [condition, f"lr{lr:g}"]
    if train_size:
        parts.append(f"n{train_size}")
    parts.append(f"s{seed}")
    return "_".join(parts)


# ------------------------------------------------------------------ generate

def cmd_generate(spec: LangSpec) -> dict:
    """Builds all six splits, writes them as JSONL, and writes the dataset statistics report."""
    data, pools = build_dataset(spec.lexicon, spec.templates, spec.build)
    for split, recs in data.items():
        write_jsonl(recs, spec.data_dir / f"{split}.jsonl")
    tok = Tokenizer(spec.tokenizer_path)
    stats = dataset_stats(data, pools, spec.lexicon, spec.build, tokenizer=tok)
    stats["round_trip_failures"] = sum(tok.decode(tok.encode(r["prompt"])) != r["prompt"]
                                       for recs in data.values() for r in recs)
    stats["pools_detail"] = pools
    spec.results_dir.mkdir(parents=True, exist_ok=True)
    (spec.results_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (spec.results_dir / "dataset_stats.md").write_text(stats_markdown(spec, stats, data), encoding="utf-8")
    print(stats_markdown(spec, stats, data))
    return stats


def stats_markdown(spec: LangSpec, stats: dict, data: dict[str, list[dict]]) -> str:
    """The dataset table for the report: sizes, variety, leakage, heuristic baselines, examples."""
    s = stats["splits"]
    lines = [f"# {spec.model_name} reasoning dataset ({spec.lang})", "",
             f"Generator seed {stats['config']['seed']}; tie probability {stats['config']['tie_prob']}; "
             f"entity pools split {stats['config']['seen_fraction']:.0%} seen / "
             f"{1 - stats['config']['seen_fraction']:.0%} unseen.", "",
             "| split | examples | distinct prompts | entities | wording | tie answers | "
             "prompt tokens (median / p95 / max) |", "|---|---|---|---|---|---|---|"]
    for split, st in s.items():
        pool, fam = SPLITS[split]
        tokens = st.get("prompt_tokens", {})
        lines.append(f"| {split} | {st['n']:,} | {st['distinct_prompts']:,} | {pool} | "
                     f"{'F1-F3' if fam == 'train' else 'F4'} | {st['equal_answers']:,} | "
                     f"{tokens.get('median', '-')} / {tokens.get('p95', '-')} / {tokens.get('max', '-')} |")
    p = stats["pools"]
    lines += ["", f"Entity pools: seen {p['seen']['person']} persons + {p['seen']['object']} objects; "
              f"unseen {p['unseen']['person']} persons + {p['unseen']['object']} objects; "
              f"{p['items']} countable items (shared).", "",
              "Observed sentence templates per family (entities, numbers and items abstracted): " +
              ", ".join(f"{f}: {n}" for f, n in stats["sentence_templates_total"].items()) + ".", "",
              "## Leakage checks", ""]
    lk = stats["leakage"]
    lines += [f"- prompt overlap between any two splits: {sum(lk['prompt_overlap'].values())}",
              f"- fact-level overlap between any two splits: {sum(lk['fact_overlap'].values())}",
              f"- seen/unseen pool overlap: {lk['pool_overlap']}",
              f"- unseen names in seen-entity splits: {lk['unseen_names_in_seen_splits']}",
              f"- seen names in unseen-entity splits: {lk['seen_names_in_unseen_splits']}",
              f"- template families shared by train-wording and held-out splits: {lk['family_overlap'] or 'none'}",
              f"- tokenizer round-trip failures: {stats.get('round_trip_failures', '-')}", "",
              "## Heuristic baselines (accuracy)", "",
              "| split | chance | first mentioned | last mentioned | most frequent | random candidate |",
              "|---|---|---|---|---|---|"]
    for split, st in s.items():
        h = st["heuristics"]
        lines.append(f"| {split} | {st['chance']['all']:.3f} | {h['first_mentioned']['all']:.3f} | "
                     f"{h['last_mentioned']['all']:.3f} | {h['most_frequent']['all']:.3f} | "
                     f"{h['random_candidate']['all']:.3f} |")
    lines += ["", "## Examples by task and family", ""]
    seen = set()
    for split in ("train", "test_wording"):
        for r in data[split]:
            key = (r["task"], r["family"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- `{r['task']}` {r['family']} ({r['attr']}): {r['prompt']} → **{r['answer']}**")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ finetune

def finetune_parser(spec: LangSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Reasoning finetuning for {spec.model_name} ({spec.lang}).")
    p.add_argument("--condition", choices=["sft", "replay"], default="sft")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="peak LR (default: config)")
    p.add_argument("--train-size", type=int, default=0, help="use only the first N training examples")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny run into checkpoints/smoke, logs/smoke")
    p.add_argument("--fresh", action="store_true", help="ignore an existing ckpt_last.pt")
    return p


def cmd_finetune(spec: LangSpec, argv: list[str] | None = None) -> dict:
    """One finetuning run. Resumes automatically unless --fresh."""
    args = finetune_parser(spec).parse_args(argv)
    device = device_default()
    tok = Tokenizer(spec.tokenizer_path)
    train, val = spec.split("train"), spec.split("val")

    base = spec.smoke_sft if args.smoke else spec.sft
    cfg = replace(base, seed=args.seed, lr=args.lr or base.lr, replay=args.condition == "replay",
                  train_size=args.train_size, max_steps=args.max_steps or base.max_steps)
    n = len(train[:cfg.train_size]) if cfg.train_size else len(train)
    total = cfg.max_steps or math.ceil(cfg.epochs * n / cfg.batch_size)
    # Small data-size runs have few steps; keep ~8 evaluations so early stopping means something.
    cfg = replace(cfg, eval_every=min(cfg.eval_every, max(10, total // 8)))
    name = run_name(args.condition, cfg.lr, args.seed, cfg.train_size)
    model, _, _ = load_model(spec.pretrained, map_location=device)
    out, logs = _run_dirs(spec, name, args.smoke)
    summary = finetune(model, tok, train, val, cfg, out, logs, device,
                       replay_dir=spec.encoded_dir, resume=not args.fresh,
                       meta=_meta(spec, name, spec.pretrained))
    (logs / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def _run_dirs(spec: LangSpec, name: str, smoke: bool) -> tuple[Path, Path]:
    if smoke:
        return spec.ckpt_dir / "smoke" / name, spec.log_dir / "smoke" / name
    return spec.ckpt_dir / name, spec.log_dir / name


def _meta(spec: LangSpec, name: str, init: Path) -> dict:
    return {"run": name, "language": spec.lang, "model": spec.model_name, "init_checkpoint": str(init),
            "tokenizer": str(spec.tokenizer_path), "data_seed": spec.build.seed}


# ------------------------------------------------------------------ evaluate

def evaluate_parser(spec: LangSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Reasoning evaluation for {spec.model_name} ({spec.lang}).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="a finetuning run name under checkpoints/")
    g.add_argument("--pretrained", action="store_true", help="the Phase-2 checkpoint, no finetuning")
    g.add_argument("--summarize", action="store_true", help="aggregate every results/eval/*.json")
    p.add_argument("--shots", type=int, default=0, help="in-context demonstrations (pretrained only)")
    p.add_argument("--splits", nargs="+", default=["val", *TEST_SPLITS])
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N examples per split")
    p.add_argument("--no-ppl", action="store_true", help="skip Phase-2 validation perplexity")
    p.add_argument("--batch-size", type=int, default=64)
    return p


def cmd_evaluate(spec: LangSpec, argv: list[str] | None = None) -> dict:
    """Scores one model on the reasoning splits (+ Phase-2 val perplexity), or aggregates all runs."""
    args = evaluate_parser(spec).parse_args(argv)
    if args.summarize:
        return summarize(spec)
    device = device_default()
    tok = Tokenizer(spec.tokenizer_path)
    if args.pretrained:
        ckpt, label = spec.pretrained, f"pretrained_{args.shots}shot"
    else:
        ckpt, label = spec.ckpt_dir / args.run / "ckpt_best.pt", args.run
    model, state, extra = load_model(ckpt, map_location=device)
    model = model.to(device).eval()
    train = spec.split("train") if args.shots else None

    result = {"label": label, "language": spec.lang, "model": spec.model_name, "checkpoint": str(ckpt),
              "checkpoint_step": state.step, "shots": args.shots, "splits": {},
              "train_config": extra.get("sft_config")}
    rows_dir = spec.results_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        recs = spec.split(split)
        if args.limit:
            recs = recs[:args.limit]
        summary, rows = evaluate_records(model, tok, recs, device, spec.templates.EQUAL_WORD,
                                         spec.known_names(), demos_pool=train, shots=args.shots,
                                         batch_size=args.batch_size)
        summary["heuristics"] = heuristic_accuracy(recs)
        summary["chance"] = chance_accuracy(recs)
        result["splits"][split] = summary
        write_jsonl(rows, rows_dir / f"{label}__{split}.jsonl")
        a = summary["all"]
        print(f"  {label:<34} {split:<13} EM {a['em']:.3f}  prefix {a['prefix']:.3f}  "
              f"cand {a.get('cand_acc', float('nan')):.3f}  stopped {a['stopped']:.2f}")

    if not args.no_ppl and not args.limit:
        ppl = evaluate_split(model, tok, spec.encoded_dir / "val.bin", spec.encoded_dir / "val.idx",
                             block_size=model.config.max_seq_len, batch_size=8, device=device,
                             split_name="val")
        result["phase2_val_perplexity"] = ppl.to_dict()
        print(f"  {label:<34} Phase-2 val perplexity {ppl.perplexity:.3f}  bpb {ppl.bits_per_byte:.4f}")

    out = spec.results_dir / "eval" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return result


# ------------------------------------------------------------------ summarize

def _group_of(label: str) -> str:
    """Seed-free condition label: 'replay_lr0.0001_s2' -> 'replay_lr0.0001'."""
    head, _, tail = label.rpartition("_s")
    return head if tail.isdigit() else label


def _mean_std(xs: list[float]) -> tuple[float, float]:
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarize(spec: LangSpec) -> dict:
    """Aggregates every evaluation into tables (markdown + JSON) and figures.

    Seeds of the same configuration are grouped and reported as mean ± std. Per-task, per-family and
    error-type breakdowns are kept alongside, because a single aggregate would hide the number-
    tokenisation confound (cmp tasks) and the wording gap (F4) the design exists to expose.
    """
    from lma.viz.plots import (
        plot_accuracy_bars, plot_error_breakdown, plot_scaling_curve, plot_sft_curves,
    )

    results = {}
    for path in sorted((spec.results_dir / "eval").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        results[r["label"]] = r
    groups: dict[str, list[dict]] = defaultdict(list)
    for label, r in results.items():
        groups[_group_of(label)].append(r)

    splits = ["val", *TEST_SPLITS]
    table: dict[str, dict] = {}
    for g, rs in groups.items():
        row: dict = {"n_seeds": len(rs), "splits": {}}
        for s in splits:
            if not all(s in r["splits"] for r in rs):
                continue
            entry = {}
            for metric in ("em", "prefix", "cand_acc", "stopped"):
                vals = [r["splits"][s]["all"].get(metric) for r in rs]
                if None not in vals:
                    entry[metric] = _mean_std(vals)
            tasks = rs[0]["splits"][s]["by_task"].keys()
            entry["by_task_em"] = {t: _mean_std([r["splits"][s]["by_task"][t]["em"] for r in rs]) for t in tasks}
            entry["by_task_cand"] = {t: _mean_std([r["splits"][s]["by_task"][t].get("cand_acc", 0.0) for r in rs])
                                     for t in tasks}
            entry["errors"] = {e: statistics.fmean([r["splits"][s]["errors"][e] for r in rs]) for e in ERROR_TYPES}
            entry["chance"] = rs[0]["splits"][s]["chance"]["all"]
            entry["heuristics"] = {h: v["all"] for h, v in rs[0]["splits"][s]["heuristics"].items()}
            row["splits"][s] = entry
        ppl = [r["phase2_val_perplexity"]["perplexity"] for r in rs if "phase2_val_perplexity" in r]
        if ppl:
            row["val_ppl"] = _mean_std(ppl)
        table[g] = row

    spec.results_dir.mkdir(parents=True, exist_ok=True)
    (spec.results_dir / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    md = summary_markdown(spec, table)
    (spec.results_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)

    fig = spec.figures_dir
    # Figures show the conditions the report compares: the baselines, the sweep's winning SFT
    # configuration, and every other condition. Losing sweep LRs and data-size points stay in the
    # tables (and the data-size figure) but would only crowd these bars.
    best_sft = _best_sft_group(table)
    main = [g for g in table if "_n" not in g and (not g.startswith("sft_lr") or g == best_sft)]
    for metric, ylabel in (("em", "Exact match"), ("cand_acc", "Candidate-choice accuracy")):
        series, errors = {}, {}
        for g in main:
            if all(metric in table[g]["splits"].get(s, {}) for s in TEST_SPLITS):
                series[g] = [table[g]["splits"][s][metric][0] for s in TEST_SPLITS]
                errors[g] = [table[g]["splits"][s][metric][1] for s in TEST_SPLITS]
        if series:
            chance = table[main[0]]["splits"]["test_id"]["chance"] if main else None
            plot_accuracy_bars(list(TEST_SPLITS), series, fig / f"{spec.lang}_reasoning_{metric}.png",
                               f"{spec.model_name}: {ylabel.lower()} by test split (mean ± std over seeds)",
                               ylabel=ylabel, errors=errors, reference=chance)
    err_conditions = [g for g in main if "test_both" in table[g]["splits"]]
    if err_conditions:
        plot_error_breakdown(err_conditions,
                             {e: [table[g]["splits"]["test_both"]["errors"][e] for g in err_conditions]
                              for e in ERROR_TYPES},
                             fig / f"{spec.lang}_reasoning_errors.png",
                             f"{spec.model_name}: error types on test_both (mean over seeds)")

    best = _best_sft_group(table)
    if best:
        lr = best.split("_lr")[1]
        sizes = []
        for n in sorted(spec.data_sizes):
            g = f"sft_lr{lr}_n{n}" if n != max(spec.data_sizes) else best
            if g in table and all(s in table[g]["splits"] for s in TEST_SPLITS):
                sizes.append((n, g))
        if len(sizes) > 1:
            plot_scaling_curve([n for n, _ in sizes],
                               {s: [table[g]["splits"][s]["em"][0] for _, g in sizes] for s in TEST_SPLITS},
                               fig / f"{spec.lang}_reasoning_data_size.png",
                               f"{spec.model_name}: exact match vs number of training examples",
                               "Training examples (log scale)")
        log = spec.log_dir / f"{best}_s0" / "train_log.jsonl"
        if log.exists():
            recs = read_log(log)
            st = [(r["step"], r["train_loss"]) for r in recs if r.get("kind") == "step" and "train_loss" in r]
            sv = [(r["step"], r["val_loss"], r["val_em"]) for r in recs
                  if r.get("kind") == "step" and "val_em" in r and "val_loss" in r]
            if st and sv:
                plot_sft_curves([a for a, _ in st], [b for _, b in st], [a for a, _, _ in sv],
                                [b for _, b, _ in sv], [c for _, _, c in sv],
                                fig / f"{spec.lang}_sft_curves.png", f"{spec.model_name} ({best}_s0)")
    return table


def _best_sft_group(table: dict) -> str | None:
    """The plain-SFT LR group (full data) with the highest validation EM -- the sweep's winner."""
    cands = [g for g in table if g.startswith("sft_lr") and "_n" not in g and "val" in table[g]["splits"]]
    return max(cands, key=lambda g: table[g]["splits"]["val"]["em"][0]) if cands else None


def summary_markdown(spec: LangSpec, table: dict) -> str:
    """Report tables: headline EM/candidate accuracy per split, per-task EM, errors, perplexity."""
    def ms(x):
        return f"{x[0]:.3f} ± {x[1]:.3f}" if x[1] else f"{x[0]:.3f}"
    lines = [f"# {spec.model_name} ({spec.lang}) reasoning results", "",
             "Mean ± std over seeds where more than one seed was run.", "",
             "## Exact match by split", "",
             "| condition | seeds | val | test_id | test_entity | test_wording | test_both | Phase-2 val PPL |",
             "|---|---|---|---|---|---|---|---|"]
    for g, row in table.items():
        cells = [ms(row["splits"][s]["em"]) if s in row["splits"] else "-" for s in ["val", *TEST_SPLITS]]
        ppl = ms(row["val_ppl"]) if "val_ppl" in row else "-"
        lines.append(f"| {g} | {row['n_seeds']} | " + " | ".join(cells) + f" | {ppl} |")
    lines += ["", "## Candidate-choice accuracy by split", "",
              "| condition | val | test_id | test_entity | test_wording | test_both |", "|---|---|---|---|---|---|"]
    for g, row in table.items():
        cells = [ms(row["splits"][s]["cand_acc"]) if "cand_acc" in row["splits"].get(s, {}) else "-"
                 for s in ["val", *TEST_SPLITS]]
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## Exact match by task type (test_both)", ""]
    tasks = ["cmp2", "cmp3", "ext3", "pair3", "ext4"]
    lines += ["| condition | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for g, row in table.items():
        if "test_both" in row["splits"]:
            bt = row["splits"]["test_both"]["by_task_em"]
            lines.append(f"| {g} | " + " | ".join(ms(bt[t]) if t in bt else "-" for t in tasks) + " |")
    lines += ["", "## Error types (test_both, mean count over seeds)", "",
              "| condition | " + " | ".join(ERROR_TYPES) + " |", "|---|" + "---|" * len(ERROR_TYPES)]
    for g, row in table.items():
        if "test_both" in row["splits"]:
            e = row["splits"]["test_both"]["errors"]
            lines.append(f"| {g} | " + " | ".join(f"{e[k]:.0f}" for k in ERROR_TYPES) + " |")
    first = next((r for r in table.values() if "test_both" in r["splits"]), None)
    if first:
        lines += ["", "## Heuristic baselines (test splits)", "",
                  "| split | chance | " + " | ".join(first["splits"]["test_both"]["heuristics"]) + " |",
                  "|---|---|" + "---|" * len(first["splits"]["test_both"]["heuristics"])]
        for s in TEST_SPLITS:
            if s in first["splits"]:
                h = first["splits"][s]["heuristics"]
                lines.append(f"| {s} | {first['splits'][s]['chance']:.3f} | " +
                             " | ".join(f"{v:.3f}" for v in h.values()) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ attention comparison

def attention_parser(spec: LangSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Pretrained vs finetuned attention for {spec.model_name}.")
    p.add_argument("--run", default=None, help="finetuned run (default: SFT + replay at the selected LR, seed 0)")
    p.add_argument("--layers", nargs="+", type=int, default=[0, 9], help="heatmap layers (early, late)")
    p.add_argument("--heads", nargs="+", type=int, default=[0, 3, 5], help="heatmap heads")
    p.add_argument("--reasoning-prompts", type=int, default=200, help="test_id prompts for statistics")
    p.add_argument("--docs", type=int, default=8, help="generic validation documents")
    p.add_argument("--context", type=int, default=192)
    return p


def cmd_attention(spec: LangSpec, argv: list[str] | None = None) -> dict:
    """The §3.2 analysis: does finetuning change local vs long-range attention or head roles, and
    does it create heads that point at the answer?

    Everything is computed twice -- on reasoning prompts and on ordinary validation text -- because a
    change that shows up on both is global drift from finetuning, while a change only on reasoning
    prompts is task-specific.
    """
    import numpy as np

    from lma.eval.attention import aggregate, analyse, layer_means, role_counts
    from lma.eval.generate import sample_documents
    from lma.reasoning.facts import EQUAL
    from lma.reasoning.pointer import attention_matrix, pointer_mass
    from lma.reasoning.sft_data import encode_prompt
    from lma.viz.plots import plot_attention_compare, plot_layer_compare, plot_layer_head_grid

    args = attention_parser(spec).parse_args(argv)
    device = device_default()
    tok = Tokenizer(spec.tokenizer_path)
    run = args.run
    if run is None:
        table = json.loads((spec.results_dir / "summary.json").read_text(encoding="utf-8"))
        # The recommended model, not plain SFT: plain SFT can destroy the language model (Model L,
        # perplexity 418k), and its attention changes would then mix task learning with forgetting.
        run = f"{_best_sft_group(table).replace('sft_', 'replay_', 1)}_s0"
    models = {
        "pretrained": load_model(spec.pretrained, map_location=device)[0].to(device).eval(),
        "finetuned": load_model(spec.ckpt_dir / run / "ckpt_best.pt", map_location=device)[0].to(device).eval(),
    }
    recs = spec.split("test_id")[:args.reasoning_prompts]
    docs = sample_documents(spec.encoded_dir / "val.bin", spec.encoded_dir / "val.idx", args.docs,
                            args.context, seed=42)
    fig = spec.figures_dir
    out: dict = {"run": run, "layers": {}, "roles": {}, "pointer": {}, "entropy_grid": {}}

    stats = {}
    for name, model in models.items():
        on_reasoning = aggregate([
            analyse(model, torch.tensor([encode_prompt(tok, r["prompt"])], device=device)) for r in recs])
        on_text = aggregate([
            analyse(model, torch.from_numpy(d[:args.context].astype(np.int64))[None].to(device))
            for _, d in docs])
        stats[name] = {"reasoning": on_reasoning, "text": on_text}
        for kind, st in stats[name].items():
            out["layers"][f"{name}/{kind}"] = layer_means(st)
            out["roles"][f"{name}/{kind}"] = role_counts(st)
        out["pointer"][name] = pointer_mass(model, tok, recs, device).to_dict()

    L = models["pretrained"].config.n_layers
    H = models["pretrained"].config.n_heads
    layers = list(range(L))
    for metric, ylabel in (("entropy_normalised", "Normalised attention entropy"),
                           ("mean_distance", "Mean attention distance (tokens)")):
        series = {f"{m} / {k}": [out["layers"][f"{m}/{k}"][l][metric] for l in layers]
                  for m in models for k in ("reasoning", "text")}
        plot_layer_compare(layers, series, fig / f"{spec.lang}_ft_attn_{metric}.png",
                           f"{spec.model_name}: {ylabel.lower()} by layer, pretrained vs finetuned", ylabel)
    for kind in ("reasoning", "text"):
        grid = {m: np.array([s.entropy_normalised for s in stats[m][kind]]).reshape(L, H) for m in models}
        delta = grid["finetuned"] - grid["pretrained"]
        out["entropy_grid"][kind] = {"delta": delta.round(4).tolist()}
        plot_layer_head_grid(delta, fig / f"{spec.lang}_ft_entropy_delta_{kind}.png",
                             f"{spec.model_name}: change in normalised entropy after SFT ({kind} input)",
                             "Δ normalised entropy (finetuned − pretrained)", diverging=True)

    pre = np.array(out["pointer"]["pretrained"]["score"])
    ft = np.array(out["pointer"]["finetuned"]["score"])
    plot_layer_head_grid(pre, fig / f"{spec.lang}_pointer_pretrained.png",
                         f"{spec.model_name} pretrained: answer-pointer score",
                         "Correct − distractor attention mass")
    plot_layer_head_grid(ft, fig / f"{spec.lang}_pointer_finetuned.png",
                         f"{spec.model_name} finetuned: answer-pointer score",
                         "Correct − distractor attention mass")
    plot_layer_head_grid(ft - pre, fig / f"{spec.lang}_pointer_delta.png",
                         f"{spec.model_name}: change in answer-pointer score after SFT",
                         "Δ pointer score (finetuned − pretrained)", diverging=True)
    order = np.argsort(-(ft - pre), axis=None)[:5]
    out["pointer_top_gain_heads"] = [
        {"layer": int(i // H), "head": int(i % H), "delta": round(float((ft - pre).flat[i]), 4)} for i in order]

    example = next(r for r in recs if r["task"] == "ext3" and r["answer_key"] != EQUAL)
    ids = encode_prompt(tok, example["prompt"])
    # SentencePiece's word-boundary mark U+2581 has no glyph in the Devanagari fonts and renders as
    # a box; a plain underscore keeps the boundary visible.
    labels = ["<eos>"] + [p.replace("▁", "_") for p in tok.encode_pieces(example["prompt"])]
    out["heatmap_prompt"] = {"id": example["id"], "prompt": example["prompt"], "answer": example["answer"]}
    for layer in args.layers:
        for head in args.heads:
            before = attention_matrix(models["pretrained"], ids, layer, head, device)
            after = attention_matrix(models["finetuned"], ids, layer, head, device)
            plot_attention_compare(before, after, labels, fig / f"{spec.lang}_ft_attn_L{layer}H{head}.png",
                                   f"{spec.model_name}: layer {layer}, head {head} on a comparative prompt "
                                   f"(answer: {example['answer']})")
    spec.results_dir.mkdir(parents=True, exist_ok=True)
    (spec.results_dir / "attention_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("run", "roles", "pointer_top_gain_heads")},
                     ensure_ascii=False, indent=2))
    return out
