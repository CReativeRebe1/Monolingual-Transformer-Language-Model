"""
Visualization functions for the Phase-1 (and later phases') report figures.

Guideline compliance, deliberately: every function here ONLY plots -- it takes already-computed
statistics (from lma/text/stats.py, lma/tokenizer/analyze.py, lma/text/langid.py, etc.) as input
and never itself computes a statistic, per the assignment's "keep visualization and computation
logic in separate functions" requirement. Every axes gets a title, x-label, and y-label, and a
legend wherever more than one series is drawn, since the assignment states plots missing these
may score zero for that component.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# Force the non-interactive Agg backend before pyplot is ever imported. This module only ever
# saves figures to disk (report/figures/) -- it never displays a window -- but without this,
# matplotlib may auto-select an interactive backend (e.g. TkAgg) depending on what happens to be
# importable in the environment, which then fails outright on a headless machine/CI runner
# without a working Tk installation. Setting the backend explicitly makes plot generation
# environment-independent, exactly like the rest of this pipeline's scripts.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Fonts capable of shaping Devanagari, in preference order. Nirmala UI ships with Windows (verified
# in Phase 1 by rendering the tokenization-samples figure and reading it back); Noto Sans Devanagari
# is the usual Linux/Kaggle answer. The probe exists because the failure mode is silent: matplotlib
# does not warn when a glyph is missing, it draws a hollow box, and a figure full of boxes looks
# like a rendering setting rather than a missing font.
DEVANAGARI_FONTS = (
    "Nirmala UI", "Noto Sans Devanagari", "Noto Serif Devanagari",
    "Mangal", "Lohit Devanagari", "FreeSerif", "Kokila", "Utsaah",
)


def devanagari_font() -> str | None:
    """First installed font that can shape Devanagari, or None (with a warning) if there is none.

    Returning None rather than raising: a missing font should degrade the labels on one figure, not
    abort a whole evaluation run that has already spent GPU time. The warning is loud because the
    alternative is a report figure full of hollow boxes that nobody notices until it is printed.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in DEVANAGARI_FONTS:
        if name in available:
            return name
    import warnings

    warnings.warn(
        "No Devanagari-capable font found (tried: " + ", ".join(DEVANAGARI_FONTS) + "). "
        "Devanagari labels will render as missing-glyph boxes. On Kaggle/Linux install "
        "fonts-noto-devanagari; on Windows Nirmala UI ships with the OS.",
        RuntimeWarning, stacklevel=2,
    )
    return None


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_funnel_chart(stage_names: list[str], counts: list[int], out_path: Path, corpus_label: str) -> None:
    """Horizontal bar funnel: documents surviving each cleaning-cascade stage in order."""
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(stage_names) + 1.5))
    y_pos = np.arange(len(stage_names))
    ax.barh(y_pos, counts, color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stage_names)
    ax.invert_yaxis()  # first stage on top
    ax.set_xlabel("Documents surviving")
    ax.set_ylabel("Cleaning-cascade stage")
    ax.set_title(f"{corpus_label}: document survival through the cleaning cascade")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c:,}", va="center")
    _save(fig, out_path)


def plot_length_histogram(
    lengths: list[int], out_path: Path, corpus_label: str, unit: str = "words", bins: int = 50
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(lengths, bins=bins, color="#55A868", edgecolor="white")
    ax.set_xlabel(f"Document length ({unit})")
    ax.set_ylabel("Number of documents")
    ax.set_title(f"{corpus_label}: document length distribution")
    _save(fig, out_path)


def plot_domain_distribution(docs_per_source: dict[str, int], out_path: Path, corpus_label: str, top_k: int = 15) -> None:
    top = sorted(docs_per_source.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [k for k, _ in top]
    values = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(labels) + 1.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#C44E52")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Document count")
    ax.set_ylabel("Source (domain / dataset)")
    ax.set_title(f"{corpus_label}: document count by source (top {len(labels)})")
    _save(fig, out_path)


def plot_fertility_vs_vocab_size(
    results_by_language: dict[str, list[tuple[int, float]]], out_path: Path
) -> None:
    """results_by_language: {language_label: [(vocab_size, fertility), ...]} -- one line per
    language, both drawn on shared axes so the Phase-1 vocab-size sweep is directly comparable."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, points in results_by_language.items():
        points_sorted = sorted(points)
        xs = [p[0] for p in points_sorted]
        ys = [p[1] for p in points_sorted]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Fertility (tokens per word)")
    ax.set_title("Tokenizer fertility vs. vocabulary size")
    ax.legend(title="Language")
    _save(fig, out_path)


def plot_zipf(token_frequencies: list[int], out_path: Path, corpus_label: str) -> None:
    """Log-log rank-frequency plot -- the classic Zipf's-law diagnostic for a trained vocabulary."""
    sorted_freqs = sorted(token_frequencies, reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, sorted_freqs, color="#8172B2")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Token frequency (log scale)")
    ax.set_title(f"{corpus_label}: token frequency vs. rank (Zipf plot)")
    _save(fig, out_path)


def plot_langid_confusion_matrix(
    matrix: dict[str, dict[str, int]], out_path: Path, title: str = "Language-ID confusion matrix"
) -> None:
    labels = list(matrix.keys())
    data = np.array([[matrix[true].get(pred, 0) for pred in labels] for true in labels])
    fig, ax = plt.subplots(figsize=(5 + 0.3 * len(labels), 5))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted language")
    ax.set_ylabel("True language")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Document count")
    _save(fig, out_path)


def plot_manual_vs_downloaded(manual_tokens: int, downloaded_tokens: int, out_path: Path, corpus_label: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    values = [manual_tokens, downloaded_tokens]
    labels = ["Manual (crawl + OCR)", "Downloaded"]
    colors = ["#DD8452", "#4C72B0"]
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title(f"{corpus_label}: manual vs. downloaded token split")
    ax.legend(labels, title="Source type", loc="lower left", bbox_to_anchor=(1, 0))
    _save(fig, out_path)


def plot_training_loss_curve(
    steps: list[int], train_loss: list[float], val_steps: list[int], val_loss: list[float],
    out_path: Path, model_label: str,
) -> None:
    """Reused in Phase 2 (pretraining curves) and Phase 2's checkpoint kill/resume proof --
    included here since lma/viz is the shared plotting module for the whole project, not
    duplicated per phase."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_loss, label="Train loss", color="#4C72B0", alpha=0.8)
    ax.plot(val_steps, val_loss, label="Validation loss", color="#C44E52", marker="o", markersize=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{model_label}: pretraining loss curve")
    ax.legend()
    _save(fig, out_path)


# ------------------------------------------------------------------ Phase 2: attention analysis

def plot_attention_heatmap(
    weights, token_labels: list[str], out_path: Path, title: str,
) -> None:
    """Query-vs-key attention map for one head. `weights` is a [T, T] array (already computed).

    Rows are queries, columns are keys, and the upper triangle is structurally zero -- the visible
    black wedge above the diagonal IS the causal mask, which makes this figure a check on the model
    as well as a description of it.
    """
    weights = np.asarray(weights)
    font = devanagari_font()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(weights, cmap="viridis", aspect="auto", interpolation="nearest")

    n = len(token_labels)
    # Every label at T=128 is an unreadable smear, so thin them to ~24 ticks.
    stride = max(1, n // 24)
    ticks = list(range(0, n, stride))
    labels = [token_labels[i] for i in ticks]
    kw = {"fontfamily": font} if font else {}
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7, **kw)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7, **kw)
    ax.set_xlabel("Key position (attended to)")
    ax.set_ylabel("Query position (attending from)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    _save(fig, out_path)


def plot_head_taxonomy(
    distances: list[float], entropies: list[float], layers: list[int], roles: list[str],
    out_path: Path, model_label: str,
) -> None:
    """Mean attention distance vs normalised entropy, one point per head, coloured by depth.

    The single figure that summarises every head at once: local heads cluster bottom-left, diffuse
    global heads top-right, and the colour gradient shows whether specialisation tracks depth.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        distances, entropies, c=layers, cmap="viridis", s=48, edgecolor="white", linewidth=0.5
    )
    for x, y, role in zip(distances, entropies, roles):
        if role in ("previous_token", "attention_sink"):
            # Only the mechanically-identified roles are annotated. Labelling all 60 points would
            # make the figure unreadable and the other categories are threshold conventions anyway.
            ax.annotate(role.replace("_", " "), (x, y), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", alpha=0.75)
    ax.set_xlabel("Mean attention distance (tokens)")
    ax.set_ylabel("Normalised attention entropy  H / log2(i+1)")
    ax.set_title(f"{model_label}: attention head taxonomy")
    fig.colorbar(scatter, ax=ax, label="Layer (0 = first)")
    _save(fig, out_path)


def plot_attention_by_layer(
    layers: list[int], entropy: list[float], distance: list[float],
    out_path: Path, model_label: str,
) -> None:
    """Layer-mean entropy and attention distance against depth, on twin axes.

    The depth trend is the claim the report makes about the model's structure -- early layers local,
    later layers integrating over more context -- so it gets its own figure rather than being left
    for the reader to extract from a 60-row table.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, entropy, marker="o", color="#4C72B0", label="Normalised entropy")
    ax.set_xlabel("Layer (0 = first)")
    ax.set_ylabel("Normalised attention entropy", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax.twinx()
    ax2.plot(layers, distance, marker="s", color="#C44E52", label="Mean distance")
    ax2.set_ylabel("Mean attention distance (tokens)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax.set_title(f"{model_label}: attention behaviour by depth")
    _save(fig, out_path)


def plot_diversity_grid(
    setting_names: list[str], distinct_2: list[float], self_bleu: list[float],
    out_path: Path, model_label: str,
) -> None:
    """Diversity against repetitiveness over the decoding grid -- where the trade-off knee is."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(setting_names))
    width = 0.38
    ax.bar(x - width / 2, distinct_2, width, label="distinct-2 (higher = more varied)", color="#55A868")
    ax.bar(x + width / 2, self_bleu, width, label="self-BLEU (lower = more varied)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(setting_names, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("Decoding configuration")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_label}: generation diversity across decoding settings")
    ax.legend()
    _save(fig, out_path)
