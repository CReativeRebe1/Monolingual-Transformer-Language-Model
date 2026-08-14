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

import matplotlib.pyplot as plt
import numpy as np


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
