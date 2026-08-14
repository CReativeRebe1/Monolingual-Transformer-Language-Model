"""Smoke tests for lma.viz.plots -- each function must run without error and produce a non-empty
file. Visual correctness is not meaningfully unit-testable; the title/xlabel/ylabel/legend
requirement is enforced by code review of lma/viz/plots.py itself (every function sets them
unconditionally), not by inspecting rendered pixels here.
"""
from lma.viz.plots import (
    plot_domain_distribution,
    plot_fertility_vs_vocab_size,
    plot_funnel_chart,
    plot_langid_confusion_matrix,
    plot_length_histogram,
    plot_manual_vs_downloaded,
    plot_training_loss_curve,
    plot_zipf,
)


def _assert_nonempty_file(path):
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_funnel_chart(tmp_path):
    out = tmp_path / "funnel.png"
    plot_funnel_chart(["collected", "filtered", "deduped"], [1000, 700, 500], out, "Hindi")
    _assert_nonempty_file(out)


def test_plot_length_histogram(tmp_path):
    out = tmp_path / "lengths.png"
    plot_length_histogram([10, 20, 30, 40, 15, 25], out, "Hindi", unit="words", bins=5)
    _assert_nonempty_file(out)


def test_plot_domain_distribution(tmp_path):
    out = tmp_path / "domains.png"
    plot_domain_distribution({"a.com": 100, "b.com": 50, "c.com": 20}, out, "Hindi")
    _assert_nonempty_file(out)


def test_plot_fertility_vs_vocab_size(tmp_path):
    out = tmp_path / "fertility.png"
    plot_fertility_vs_vocab_size(
        {"Hindi": [(8000, 1.8), (16000, 1.5), (32000, 1.3)],
         "Nepali": [(8000, 2.0), (16000, 1.7), (32000, 1.4)]},
        out,
    )
    _assert_nonempty_file(out)


def test_plot_zipf(tmp_path):
    out = tmp_path / "zipf.png"
    plot_zipf([100, 80, 60, 40, 20, 10, 5, 3, 2, 1], out, "Hindi")
    _assert_nonempty_file(out)


def test_plot_langid_confusion_matrix(tmp_path):
    out = tmp_path / "confusion.png"
    matrix = {"hi": {"hi": 90, "ne": 10}, "ne": {"hi": 20, "ne": 80}}
    plot_langid_confusion_matrix(matrix, out)
    _assert_nonempty_file(out)


def test_plot_manual_vs_downloaded(tmp_path):
    out = tmp_path / "manual_split.png"
    plot_manual_vs_downloaded(120_000_000, 380_000_000, out, "Hindi")
    _assert_nonempty_file(out)


def test_plot_training_loss_curve(tmp_path):
    out = tmp_path / "loss.png"
    plot_training_loss_curve(
        steps=list(range(0, 1000, 10)),
        train_loss=[3.0 - i * 0.002 for i in range(100)],
        val_steps=list(range(0, 1000, 100)),
        val_loss=[3.1 - i * 0.02 for i in range(10)],
        out_path=out,
        model_label="Model H",
    )
    _assert_nonempty_file(out)
