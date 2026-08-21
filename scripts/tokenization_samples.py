"""
Renders qualitative subword-tokenization samples for the Phase-1 report.

    python scripts/tokenization_samples.py

For a handful of real held-out sentences per language, this shows how the sentence is split into
subword pieces by BOTH tokenizers -- its own and the other language's -- and writes:

  report/figures/tokenization_samples.png   the segmentations, in Devanagari
  report/figures/tokenization_samples.json  the same data as text, for the Markdown report

**Why a figure and not a LaTeX table.** The report compiles with pdflatex, which cannot typeset
Devanagari without an Indic-capable font and a different engine. Rendering the samples to an image
keeps the toolchain as it is while still showing the actual script -- transliterating them instead
would defeat the purpose, since the whole point is to see where the segmenter cuts a conjunct or a
matra. Verified that the renderer shapes Devanagari correctly (conjuncts and pre-base matras form
properly) before adopting this route.

Sentences are drawn from each language's held-out validation split, so they are real corpus text the
tokenizer was not trained on.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lma.tokenizer.train_sp import Tokenizer  # noqa: E402

DEV_FONT = "Nirmala UI"          # ships with Windows; shapes Devanagari correctly
SEP = "|"                     # ASCII: Nirmala UI has no box-drawing glyphs
SENTENCE_END = re.compile(r"(?<=[।॥.!?])\s+")
MIN_WORDS, MAX_WORDS = 6, 10      # short enough that a segmentation fits on one line
MAX_LINE_CHARS = 104              # a rendered segmentation wider than this runs off the figure
WORD_START = "_"                  # SentencePiece marks word starts with U+2581, which Nirmala UI
                                  # lacks; it renders as a missing-glyph box, so display it as "_"
N_SAMPLES = 4


def pick_sentences(split_path: Path, n: int) -> list[str]:
    """Real held-out sentences, short enough to display whole."""
    out: list[str] = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            for sent in SENTENCE_END.split(line.strip()):
                sent = sent.strip()
                if MIN_WORDS <= len(sent.split()) <= MAX_WORDS and len(sent) > 30:
                    out.append(sent)
                    if len(out) >= n:
                        return out
    return out


def pieces(tok: Tokenizer, text: str) -> list[str]:
    return tok._sp.encode(text, out_type=str)


def shown(ps: list[str]) -> str:
    """Piece list as one display line, with the word-start marker made visible."""
    return SEP.join(p.replace("▁", WORD_START) for p in ps)


def main() -> int:
    p = argparse.ArgumentParser(description="Qualitative tokenization samples for the report.")
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    args = p.parse_args()

    toks = {
        "hindi": Tokenizer(REPO_ROOT / "hindi/tokenizer/hindi_unigram_16000.model"),
        "nepali": Tokenizer(REPO_ROOT / "nepali/tokenizer/nepali_unigram_16000.model"),
    }

    data: dict[str, list[dict]] = {}
    for lang in ("hindi", "nepali"):
        sents = pick_sentences(REPO_ROOT / lang / "data/splits/val.txt", args.samples * 12)
        rows = []
        for s in sents:
            hi, ne = pieces(toks["hindi"], s), pieces(toks["nepali"], s)
            if max(len(shown(hi)), len(shown(ne))) > MAX_LINE_CHARS:
                continue          # would overflow the figure and collide with the count label
            if len(rows) >= args.samples:
                break
            rows.append({
                "text": s,
                "words": len(s.split()),
                "hindi_pieces": hi,
                "nepali_pieces": ne,
                "hindi_n": len(hi),
                "nepali_n": len(ne),
            })
        data[lang] = rows
        print(f"{lang}: {len(rows)} sentences")

    out_json = REPO_ROOT / "report/figures/tokenization_samples.json"
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- figure: one block per sentence, own tokenizer above the other language's
    plt.rcParams["font.family"] = DEV_FONT
    n_rows = sum(len(v) for v in data.values())
    fig, ax = plt.subplots(figsize=(11.2, 0.98 * n_rows + 1.1))
    ax.axis("off")
    y = 1.0
    step = 1.0 / (n_rows * 3 + 4)

    for lang, rows in data.items():
        other = "nepali" if lang == "hindi" else "hindi"
        ax.text(0.0, y, f"{lang.title()} held-out sentences", fontsize=11, fontweight="bold",
                color="#1f5a96" if lang == "hindi" else "#b05414")
        y -= step * 1.5
        for r in rows:
            ax.text(0.0, y, r["text"], fontsize=11)
            y -= step
            ax.text(0.0, y, f"{lang[:2]} {r[f'{lang}_n']:>2}", fontsize=8.5, color="#1f5a96",
                    fontweight="bold")
            ax.text(0.052, y, shown(r[f"{lang}_pieces"]), fontsize=9.5, color="#1f5a96")
            y -= step
            ax.text(0.0, y, f"{other[:2]} {r[f'{other}_n']:>2}", fontsize=8.5, color="#b05414",
                    fontweight="bold")
            ax.text(0.052, y, shown(r[f"{other}_pieces"]), fontsize=9.5, color="#b05414")
            y -= step * 1.4
        y -= step * 0.6

    out_png = REPO_ROOT / "report/figures/tokenization_samples.png"
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png.name} and {out_json.name}")

    # ---- the comparative summary the report quotes
    for lang, rows in data.items():
        other = "nepali" if lang == "hindi" else "hindi"
        own_t = sum(r[f"{lang}_n"] for r in rows)
        oth_t = sum(r[f"{other}_n"] for r in rows)
        words = sum(r["words"] for r in rows)
        same = sum(1 for r in rows if r["hindi_pieces"] == r["nepali_pieces"])
        print(f"{lang}: own {own_t} tokens ({own_t/words:.3f}/word), "
              f"{other} tokenizer {oth_t} ({oth_t/words:.3f}/word), "
              f"+{100*(oth_t/own_t - 1):.1f}%; identical segmentations {same}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
