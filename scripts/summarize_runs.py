"""
Writes result.md: a side-by-side review summary of both languages' cleaning runs.

    python scripts/summarize_runs.py

Reads each language's data/stats.json -- the file its pipeline wrote -- and renders the funnel,
thresholds, rejection reasons, dedup counts and final corpus size as Markdown. Generated rather than
hand-written on purpose: the numbers change every time a run or a re-split happens, and a summary
copied by hand goes stale silently while still looking authoritative.

Missing languages are skipped with a note, so this is safe to run when only one side has finished.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ("hindi", "nepali")
FERTILITY = 1.31          # tokens per word, from scripts/tokenizer_sweep.py's sweep_results.json

STAGE_LABELS = {
    "collected": "Collected (raw manual)",
    "language": "After language filtering",
    "repaired_and_rules": "After repair + text metric rules",
    "quality_scored": "After perplexity / quality / coherence",
    "paragraph_deduped": "After paragraph dedup",
    "exact_deduped": "After exact dedup",
    "near_deduped": "After near dedup (MinHash-LSH)",
    "topic_filtered": "After topic filtering",
    "safety_checked": "After safety filtering",
    "decontaminated": "Final (post-decontamination, incl. downloaded)",
}


def load(language: str) -> dict | None:
    """Returns the language's stats, or None if it has not completed a run under this pipeline.

    Both `funnel` and `words` are required. The previous pipeline also wrote a stats.json with a
    `funnel` in it, so checking for that alone treats a stale file from the old cascade as a finished
    run and then dies on the first missing key -- better to say "no completed run" than to render
    half a summary from numbers that describe different code.
    """
    path = REPO_ROOT / language / "data" / "stats.json"
    if not path.exists():
        return None
    stats = json.loads(path.read_text(encoding="utf-8"))
    if "funnel" not in stats or "words" not in stats:
        print(f"  note: {path} predates the current pipeline; skipping {language}")
        return None
    return stats


def fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else ("-" if n is None else str(n))


def funnel_table(stats: dict) -> list[str]:
    rows = ["| Stage | Documents | % of previous | % of collected |",
            "|---|---:|---:|---:|"]
    funnel = stats["funnel"]
    first = next(iter(funnel.values()), 0) or 1
    prev = None
    for key, value in funnel.items():
        pct_prev = "-" if prev is None else f"{100 * value / prev:.1f}%"
        rows.append(f"| {STAGE_LABELS.get(key, key)} | {fmt(value)} | {pct_prev} | "
                    f"{100 * value / first:.1f}% |")
        prev = value
    return rows


def section(language: str, stats: dict) -> list[str]:
    w = stats["words"]
    dedup = stats.get("dedup", {})
    dl = stats.get("downloaded", {})
    decon = stats.get("decontamination", {})
    thr = stats.get("thresholds", {})
    splits = stats.get("splits", {})
    holdout = decon.get("holdout", {})

    out = [f"## {language.title()}", ""]
    out += [
        f"Run took {stats.get('elapsed_s', 0) / 60:.0f} min.",
        "",
        "### Final corpus",
        "",
        "| | Words | Tokens (x1.31) |",
        "|---|---:|---:|",
        f"| Manual | {fmt(w['manual'])} | {fmt(int(w['manual'] * FERTILITY))} |",
        f"| Downloaded | {fmt(w['downloaded'])} | {fmt(int(w['downloaded'] * FERTILITY))} |",
        f"| **Total** | **{fmt(w['total'])}** | **{fmt(int(w['total'] * FERTILITY))}** |",
        "",
        f"**Manual share: {100 * w['manual_fraction']:.1f}%** "
        f"({'clears' if w['manual_fraction'] >= 0.20 else 'BELOW'} the 20% floor).",
        "",
        f"Splits: train {fmt(splits.get('train'))} / val {fmt(splits.get('val'))} / "
        f"test {fmt(splits.get('test'))} documents.",
    ]
    if holdout:
        out += ["", f"Held-out composition: val {fmt(holdout.get('val_manual'))} manual + "
                    f"{fmt(holdout.get('val', 0) - holdout.get('val_manual', 0))} downloaded; "
                    f"test {fmt(holdout.get('test_manual'))} manual + "
                    f"{fmt(holdout.get('test', 0) - holdout.get('test_manual', 0))} downloaded."]

    out += ["", "### Cleaning funnel (manual side)", ""] + funnel_table(stats)

    out += [
        "",
        "### Thresholds used (calibrated per language)",
        "",
        "| Threshold | Value |",
        "|---|---:|",
        f"| Perplexity (99th pct of trusted text) | {fmt(thr.get('perplexity'))} |",
        f"| Quality classifier minimum | {fmt(thr.get('quality_min'))} |",
        f"| Coherence minimum (1st pct of wiki) | {fmt(thr.get('coherence_min'))} |",
        f"| fastText LID minimum confidence | {fmt(thr.get('lid_min_confidence'))} |",
        "",
        "### Deduplication",
        "",
        "| Stage | Removed |",
        "|---|---:|",
        f"| Paragraphs excised (cross-document) | {fmt(dedup.get('paragraphs_removed'))} "
        f"across {fmt(dedup.get('documents_touched'))} documents |",
        f"| Exact duplicate documents | {fmt(dedup.get('exact_removed'))} |",
        f"| Near duplicate documents | {fmt(dedup.get('near_removed'))} "
        f"in {fmt(dedup.get('near_clusters'))} clusters |",
        "",
        "### Decontamination (13-gram overlap with val/test)",
        "",
        f"- Manual train documents dropped: **{fmt(decon.get('manual_dropped'))}** of "
        f"{fmt(decon.get('manual_considered'))}"
        + (f" ({100 * decon['manual_dropped'] / decon['manual_considered']:.1f}%)"
           if decon.get("manual_considered") else ""),
        f"- Downloaded train documents dropped: {fmt(decon.get('downloaded_dropped'))}",
        f"- Downloaded documents scanned for held-out candidates: {fmt(dl.get('documents_streamed'))}",
        f"- Downloaded word budget: {fmt(dl.get('words_budget'))} (counted as words WRITTEN)",
    ]

    rejected = stats.get("rejected_by_rule", {})
    if rejected:
        out += ["", "### Top rejection reasons", "", "| Rule | Documents |", "|---|---:|"]
        out += [f"| `{k}` | {fmt(v)} |" for k, v in list(rejected.items())[:15]]

    repairs = stats.get("repairs_applied", {})
    if repairs:
        out += ["", "### Repairs applied (before any filtering)", "",
                "| Repair | Documents |", "|---|---:|"]
        out += [f"| `{k}` | {fmt(v)} |" for k, v in repairs.items()]

    clusters = stats.get("clusters", {})
    if clusters:
        out += ["", f"### Semantic / topic filtering", "",
                f"{fmt(clusters.get('n'))} clusters; {len(clusters.get('ad_heavy', []))} flagged "
                f"ad-heavy; {fmt(clusters.get('topic_dropped'))} documents downsampled away."]
    return out + [""]


def main() -> int:
    lines = [
        "# Phase 1 cleaning results",
        "",
        f"_Generated by `scripts/summarize_runs.py` on {datetime.now():%Y-%m-%d %H:%M}. "
        "Regenerate after any run or re-split; do not edit by hand._",
        "",
        "Both languages run independent pipelines (`hindi/clean/`, `nepali/clean/`) implementing "
        "the Yi 5-stage cascade. Thresholds are calibrated per language on that language's own "
        "data -- see each `clean/config.py` for the measurements behind every number.",
        "",
    ]
    found = 0
    for language in LANGUAGES:
        stats = load(language)
        if stats is None:
            lines += [f"## {language.title()}", "", "_No completed run yet._", ""]
            continue
        found += 1
        lines += section(language, stats)

    out_path = REPO_ROOT / "result.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path} ({found}/{len(LANGUAGES)} languages with completed runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
