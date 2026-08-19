"""
Builds an honest language-ID confusion matrix for the Phase-1 report.

    python scripts/langid_report.py

**Why this script exists.** The previous confusion-matrix figures were built in-sample: the
character-n-gram classifier was evaluated on the same seed documents it had been fitted on, so its
diagonal dominance was guaranteed by construction and told a reader nothing about how well language
identification actually works on this corpus.

Two things are needed for the figure to mean something, and both are arranged here:

1. **Labels that do not come from our own pipeline.** Ground truth is the Wikipedia dump the document
   came from -- `hi.wikipedia.org` is Hindi, `ne.wikipedia.org` is Nepali -- which is independent of
   every filter in `clean/`. Using cleaned splits instead would be circular in a subtler way than the
   original bug: those documents survived a fastText check, so measuring fastText on them would
   measure the filter against itself and report ~100% by construction.
2. **Raw documents, before the language filter runs.** Read from `<lang>/data/raw/wiki/`, not from
   the splits.

fastText's `lid.176` is a pretrained model we never trained, so evaluating it on held-out labelled
text is genuinely out-of-sample. The marker vote from each language's `clean/lang.py` is measured on
the same documents for comparison, since it is the tiebreaker that catches what fastText confuses.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import zlib
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lma.viz.plots import plot_langid_confusion_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.langid_report")

LANGUAGES = ("hindi", "nepali")
FASTTEXT_CODE = {"hindi": "hi", "nepali": "ne"}
DEFAULT_SAMPLE = 3_000
MIN_CHARS = 200


def read_wiki(language: str, limit: int) -> list[str]:
    """Raw Wikipedia-dump documents for one language. The dump is the ground-truth label."""
    out: list[str] = []
    d = REPO_ROOT / language / "data" / "raw" / "wiki"
    for shard in sorted(d.glob("*.jsonl.gz")):
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        text = (json.loads(line).get("text") or "").strip()
                    except json.JSONDecodeError:
                        continue
                    if len(text) >= MIN_CHARS:
                        out.append(text)
                    if len(out) >= limit:
                        return out
        except (EOFError, OSError, zlib.error, UnicodeDecodeError):
            continue
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Out-of-sample language-ID confusion matrix.")
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                   help="documents per language (default: %(default)s)")
    args = p.parse_args()

    try:
        from lma.text.langid import FastTextLangID

        lid = FastTextLangID()
    except Exception as e:                        # noqa: BLE001
        logger.error("fastText unavailable (%s); cannot build the confusion matrix", e)
        return 1

    from hindi.clean import lang as hindi_lang
    from nepali.clean import lang as nepali_lang
    from hindi.clean.norm import norm as hindi_norm
    from nepali.clean.norm import norm as nepali_norm

    markers = {"hindi": hindi_lang, "nepali": nepali_lang}
    norms = {"hindi": hindi_norm, "nepali": nepali_norm}

    # Square matrix: the plotter uses the same label list for both axes, so "other" needs a row too.
    labels = ["hindi", "nepali", "other"]
    matrix = {t: {p_: 0 for p_ in labels} for t in labels}
    summary: dict[str, dict] = {}

    for language in LANGUAGES:
        docs = read_wiki(language, args.sample)
        if not docs:
            logger.warning("no raw wiki shards for %s; skipping", language)
            continue
        raw_labels: Counter[str] = Counter()
        marker_correct = 0
        for text in docs:
            normed = norms[language](text)
            code, _conf = lid.predict(normed)
            raw_labels[code] += 1
            predicted = {"hi": "hindi", "ne": "nepali"}.get(code, "other")
            matrix[language][predicted] += 1
            # The marker vote returns (this language's hits, the other language's hits).
            own, other = markers[language].marker_vote(normed)
            marker_correct += 1 if own >= other else 0

        n = len(docs)
        correct = matrix[language][language]
        summary[language] = {
            "documents": n,
            "fasttext_correct": correct,
            "fasttext_accuracy": round(correct / n, 4),
            "confused_as_other_language": matrix[language][
                "nepali" if language == "hindi" else "hindi"
            ],
            "labelled_other_script_or_language": matrix[language]["other"],
            "marker_vote_agrees": marker_correct,
            "marker_vote_accuracy": round(marker_correct / n, 4),
            "raw_fasttext_labels": dict(raw_labels.most_common(6)),
        }
        logger.info("%s: %d documents, fastText %.1f%% correct, marker vote %.1f%% correct",
                    language, n, 100 * correct / n, 100 * marker_correct / n)

    fig_dir = REPO_ROOT / "report" / "figures"
    out_png = fig_dir / "langid_confusion.png"
    plot_langid_confusion_matrix(
        matrix, out_png,
        title="fastText lid.176 on held-out Wikipedia-dump text (labels from dump source)",
    )
    out_json = fig_dir / "langid_confusion.json"
    out_json.write_text(json.dumps({"matrix": matrix, "per_language": summary},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s and %s", out_png.name, out_json.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
