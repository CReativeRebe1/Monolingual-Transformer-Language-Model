"""
Tokenizer vocabulary-size sweep: trains BPE and Unigram tokenizers at {8k, 16k, 24k, 32k, 48k}
for one language on its train split, evaluates each on the held-out val split (fertility,
chars/token, UNK rate, vocab utilization), and selects a final tokenizer per the assignment's
stated criterion -- "fertility / unknown-token rate on held-out text".

Since byte_fallback=True makes UNK rate exactly 0% at every vocab size (see
lma/tokenizer/train_sp.py's docstring), selection in practice is driven by fertility: the
smallest vocab size past which fertility stops improving by a meaningful margin (an elbow
criterion), so the final model is not needlessly large relative to the ~25M-parameter budget
Phase 2 imposes -- vocabulary size directly sets embedding-table size at that budget.

Usage:
    python scripts/tokenizer_sweep.py --lang hindi
    python scripts/tokenizer_sweep.py --lang nepali
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from collections import Counter  # noqa: E402
from lma.tokenizer.analyze import (  # noqa: E402
    evaluate_tokenizer, token_frequency_table, token_length_distribution,
)
from lma.tokenizer.sample import build_training_sample  # noqa: E402
from lma.tokenizer.train_sp import Tokenizer, TokenizerTrainConfig, train_tokenizer  # noqa: E402
from lma.viz.plots import (  # noqa: E402
    plot_fertility_vs_vocab_size, plot_length_histogram, plot_manual_vs_downloaded, plot_zipf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lma.tokenizer_sweep")

VOCAB_SIZES = (8_000, 12_000, 16_000, 24_000, 32_000, 48_000)
ALGORITHMS = ("bpe", "unigram")
ELBOW_IMPROVEMENT_THRESHOLD = 0.03  # stop growing vocab once fertility improves <3% per step
SAMPLE_SENTENCES = 1_000_000        # sentences sampled from train for tokenizer training
SAMPLE_DOCUMENTS = 200_000          # documents those sentences are drawn from, spread by stride
# 1M sentences, not 5M: SentencePiece died with "MemoryError: bad allocation" training BPE on a
# 5M-sentence / 516M-character sample on this 7.4 GB machine. 1M sentences (~100M characters) is a
# standard size for SentencePiece training and is far more than a 48k vocabulary needs -- the merge
# statistics stop moving long before that. The cost of the smaller sample is nil; the cost of the
# larger one was the run.
PHASE2_PARAM_BUDGET = 25_000_000    # the assignment's ~25M-parameter model
PHASE2_D_MODEL = 512                # embedding width assumed when costing a vocabulary
EMBEDDING_BUDGET_SHARE = 0.35       # above this share of the budget, a vocab is flagged unaffordable


def embedding_cost(vocab_size: int) -> tuple[int, float]:
    """(embedding parameters, share of the Phase-2 budget) for a vocabulary.

    Fertility alone always prefers a bigger vocabulary, and at a 25M-parameter budget that
    preference is not free: the embedding table is vocab_size * d_model, so a 48k vocabulary would
    spend ~24.6M parameters -- essentially the entire model -- before a single Transformer block
    exists. The sweep reports this next to fertility so the trade is visible rather than implied.
    """
    params = vocab_size * PHASE2_D_MODEL
    return params, params / PHASE2_PARAM_BUDGET


def _flatten_counts(length_counts) -> list[int]:
    """Expands a {length: count} Counter into a flat list of individual lengths, suitable for
    plot_length_histogram (which expects one entry per observation, not pre-aggregated counts).
    Pulled out as its own function purely so the expansion logic itself -- easy to get backwards
    (e.g. count repeated by length instead of length repeated by count) -- is directly testable."""
    return [length for length, count in length_counts.items() for _ in range(count)]


def _elbow_vocab_size(sweep_by_vocab: dict[int, float]) -> int:
    """Smallest vocab size past which fertility improvement falls below the elbow threshold."""
    sizes = sorted(sweep_by_vocab.keys())
    if len(sizes) == 1:
        return sizes[0]
    for i in range(1, len(sizes)):
        prev_fertility = sweep_by_vocab[sizes[i - 1]]
        curr_fertility = sweep_by_vocab[sizes[i]]
        if prev_fertility <= 0:
            continue
        improvement = (prev_fertility - curr_fertility) / prev_fertility
        if improvement < ELBOW_IMPROVEMENT_THRESHOLD:
            return sizes[i - 1]
    return sizes[-1]  # fertility kept improving meaningfully all the way to the largest size


def token_stats(lang_dir: Path, tokenizer: Tokenizer) -> dict:
    """Encodes every split document to get the TRUE manual-vs-downloaded token split.

    This is the assignment's explicitly required statistic (report/phase1.md 2.1/3), and it is a
    full-corpus encoding rather than a sample because it backs a headline number.

    Streams the manifest.jsonl + .txt pair line by line. The previous version called read_text() on
    both, which is 5.1 GB for Hindi and 7.5 GB for Nepali -- it would have exhausted this machine's
    7.4 GB of RAM before producing anything, the same failure the cleaning cascade hit.
    """
    splits_dir = lang_dir / "data" / "splits"
    manual_tokens = downloaded_tokens = 0
    manual_docs = downloaded_docs = 0
    n_words = n_chars = 0
    hosts: Counter[str] = Counter()

    for split_name in ("train", "val", "test"):
        manifest_path = splits_dir / f"{split_name}.manifest.jsonl"
        text_path = splits_dir / f"{split_name}.txt"
        if not manifest_path.exists() or not text_path.exists():
            continue
        n_lines = 0
        with open(manifest_path, encoding="utf-8") as mf, open(text_path, encoding="utf-8") as tf:
            for manifest_line, text_line in zip(mf, tf):
                n_lines += 1
                meta = json.loads(manifest_line)
                text = text_line.rstrip("\n")
                n_tokens = len(tokenizer.encode(text))
                n_words += len(text.split())
                n_chars += len(text)
                hosts[meta.get("host") or "unknown"] += 1
                if meta.get("source_type") == "downloaded":
                    downloaded_tokens += n_tokens
                    downloaded_docs += 1
                else:
                    manual_tokens += n_tokens
                    manual_docs += 1
                if n_lines % 500_000 == 0:
                    logger.info("    %s: %d documents encoded", split_name, n_lines)
        # A length mismatch means the two files stopped being written together, which would make
        # every source_type attribution below wrong. Checked by exhausting both iterators rather
        # than by reading the files into memory: these splits are multi-gigabyte, and the
        # read_text() version of this function would exhaust RAM long before it could compare.
        leftover_manifest = sum(1 for _ in open(manifest_path, encoding="utf-8")) - n_lines
        leftover_text = sum(1 for _ in open(text_path, encoding="utf-8")) - n_lines
        if leftover_manifest or leftover_text:
            raise ValueError(
                f"{split_name}: manifest and text files differ in length "
                f"({leftover_manifest:+d} / {leftover_text:+d} lines beyond the {n_lines} read) -- "
                f"they are written together by the cleaning pipeline and must stay in lockstep."
            )

    total_tokens = manual_tokens + downloaded_tokens
    return {
        "n_documents": manual_docs + downloaded_docs,
        "n_words": n_words,
        "n_chars": n_chars,
        "n_tokens": total_tokens,
        "manual_documents": manual_docs,
        "downloaded_documents": downloaded_docs,
        "manual_tokens": manual_tokens,
        "downloaded_tokens": downloaded_tokens,
        "manual_fraction": manual_tokens / total_tokens if total_tokens else None,
        "n_hosts": len(hosts),
        "top_hosts": hosts.most_common(15),
    }


def run_sweep(language: str, reuse_models: bool = False,
              pin_vocab_size: int | None = None,
              pin_algorithm: str | None = None) -> dict:
    lang_dir = REPO_ROOT / language
    train_path = lang_dir / "data" / "splits" / "train.txt"
    val_path = lang_dir / "data" / "splits" / "val.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found; run scripts/build_corpus.py --lang {language} first"
        )

    val_texts = []
    if val_path.exists() and val_path.stat().st_size > 0:
        val_texts = [l.strip() for l in val_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not val_texts:
        # Corpus too small yet to have a non-empty val split (e.g. early in the crawl); fall
        # back to a sample of train text for the sweep's own metrics, and note this explicitly
        # in the results so it is never confused with a real held-out evaluation.
        logger.warning(
            "no val split available yet for %s; sweep metrics use a TRAIN sample "
            "(re-run once the corpus is large enough for a real held-out val split)",
            language,
        )
        train_lines = [l.strip() for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        val_texts = train_lines[: max(1, len(train_lines) // 10)]
        used_train_as_val = True
    else:
        used_train_as_val = False

    sweep_dir = lang_dir / "tokenizer" / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # One sentence-split sample, reused by all 10 trainings. See lma/tokenizer/sample.py for why
    # documents cannot be fed to SentencePiece directly (it silently drops lines over 4192 bytes).
    sample_path = sweep_dir / f"{language}_sentences.txt"
    if sample_path.exists() and sample_path.stat().st_size > 0:
        logger.info("reusing existing training sample %s", sample_path.name)
        sample_info = None
    else:
        # Stride so the sample spans the corpus. The splits are written manual-first, then
        # downloaded, so a head sample of 200k documents out of 3.2M would contain no downloaded
        # text at all and the tokenizer would be fitted on a quarter of its own training data.
        n_train_docs = 0
        stats_path = lang_dir / "data" / "stats.json"
        if stats_path.exists():
            n_train_docs = json.loads(stats_path.read_text(encoding="utf-8")).get(
                "splits", {}).get("train", 0)
        stride = max(1, n_train_docs // SAMPLE_DOCUMENTS) if n_train_docs else 1
        logger.info("building sentence-level training sample from %s (stride %d over %d docs)...",
                    train_path.name, stride, n_train_docs)
        sample_info = build_training_sample(train_path, sample_path,
                                            max_sentences=SAMPLE_SENTENCES, stride=stride)
        logger.info("sample: %d sentences (%.0f MB) from %d documents",
                    sample_info.n_sentences, sample_path.stat().st_size / 1e6,
                    sample_info.n_documents_read)

    results = []
    fertility_by_algo_vocab: dict[str, dict[int, float]] = {algo: {} for algo in ALGORITHMS}

    for algo in ALGORITHMS:
        for vocab_size in VOCAB_SIZES:
            out_prefix = sweep_dir / f"{language}_{algo}_{vocab_size}"
            config = TokenizerTrainConfig(algorithm=algo, vocab_size=vocab_size)
            existing = out_prefix.with_suffix(".model")
            if reuse_models and existing.exists():
                # Re-evaluating trained models against a rewritten val split, without paying for
                # training again. The sweep's expensive half is training; its metrics depend on the
                # split files, so a corpus re-split invalidates the numbers but not the models.
                logger.info("reusing %s vocab_size=%d from %s", algo, vocab_size, existing.name)
                model_path = existing
            else:
                logger.info("training %s vocab_size=%d ...", algo, vocab_size)
            try:
                if not (reuse_models and existing.exists()):
                    model_path, _ = train_tokenizer([sample_path], out_prefix, config,
                                                    input_sentence_size=SAMPLE_SENTENCES)
            except (RuntimeError, MemoryError) as e:
                logger.warning("skipping %s vocab_size=%d: %s: %s", algo, vocab_size,
                               type(e).__name__, e)
                continue
            tokenizer = Tokenizer(model_path)
            stats = evaluate_tokenizer(tokenizer, val_texts, algorithm=algo)
            fertility_by_algo_vocab[algo][vocab_size] = stats.fertility
            emb_params, emb_share = embedding_cost(vocab_size)
            results.append(
                {
                    "algorithm": algo,
                    "vocab_size": vocab_size,
                    "fertility": stats.fertility,
                    "chars_per_token": stats.chars_per_token,
                    "unk_rate": stats.unk_rate,
                    "vocab_utilization": stats.vocab_utilization,
                    "embedding_params": emb_params,
                    "embedding_share_of_budget": round(emb_share, 4),
                    "affordable_at_25M": emb_share <= EMBEDDING_BUDGET_SHARE,
                }
            )
            logger.info(
                "  fertility=%.3f chars/token=%.3f unk_rate=%.4f vocab_util=%.3f "
                "embeddings=%.1fM (%.0f%% of the 25M budget)%s",
                stats.fertility, stats.chars_per_token, stats.unk_rate, stats.vocab_utilization,
                emb_params / 1e6, 100 * emb_share,
                "" if emb_share <= EMBEDDING_BUDGET_SHARE else "  <-- unaffordable at 25M params",
            )

    # ---- selection
    #
    # Vocab size and algorithm are chosen SEPARATELY, and the algorithm comparison happens at a
    # MATCHED vocab size. Comparing each algorithm at its own elbow instead (the previous rule) is
    # apples-to-oranges: on the Nepali sweep BPE's fertility improvement never dropped below the 3%
    # threshold, so its elbow search ran off the end of the grid to 48k, and 48k-BPE then "beat"
    # 24k-unigram on raw fertility. A 48k vocabulary costs 98% of the Phase-2 parameter budget in
    # embeddings, leaving room for zero transformer layers. At every matched size, unigram wins in
    # both languages.
    if pin_vocab_size or pin_algorithm:
        best_vocab_size = pin_vocab_size or _elbow_vocab_size(
            fertility_by_algo_vocab[pin_algorithm or ALGORITHMS[0]]
        )
        candidates = [r for r in results if r["vocab_size"] == best_vocab_size
                      and (pin_algorithm is None or r["algorithm"] == pin_algorithm)]
        if not candidates:
            raise RuntimeError(
                f"no sweep result for pinned selection "
                f"(algorithm={pin_algorithm}, vocab_size={pin_vocab_size})"
            )
        chosen = min(candidates, key=lambda r: r["fertility"])
        best_algo, best_fertility = chosen["algorithm"], chosen["fertility"]
        selection_rule = (
            f"pinned by the caller (vocab_size={best_vocab_size}"
            + (f", algorithm={pin_algorithm}" if pin_algorithm else
               "; algorithm chosen at that size by fertility") + ")"
        )
    else:
        elbows = {algo: _elbow_vocab_size(sizes)
                  for algo, sizes in fertility_by_algo_vocab.items() if sizes}
        if not elbows:
            raise RuntimeError(f"tokenizer sweep produced no successful runs for {language}")
        # The vocab size is the smallest elbow across algorithms: whichever algorithm is picked,
        # that is the point past which more vocabulary stops paying for itself.
        best_vocab_size = min(elbows.values())
        at_size = [r for r in results if r["vocab_size"] == best_vocab_size]
        chosen = min(at_size, key=lambda r: r["fertility"])
        best_algo, best_fertility = chosen["algorithm"], chosen["fertility"]
        selection_rule = (
            f"fertility elbow at {ELBOW_IMPROVEMENT_THRESHOLD:.0%} per step "
            f"(per-algorithm elbows: {elbows}); algorithm compared at the matched size"
        )
        for algo, size in elbows.items():
            if size == max(VOCAB_SIZES):
                logger.warning(
                    "%s fertility was still improving by >=%.0f%% at the largest vocab size in the "
                    "grid, so its elbow never fired -- treat %d as a grid edge, not a chosen value",
                    algo, 100 * ELBOW_IMPROVEMENT_THRESHOLD, size,
                )
    logger.info("selection rule: %s", selection_rule)

    sel_params, sel_share = embedding_cost(best_vocab_size)
    logger.info("selected: algorithm=%s vocab_size=%d fertility=%.3f "
                "(embeddings %.1fM = %.0f%% of the 25M Phase-2 budget)",
                best_algo, best_vocab_size, best_fertility, sel_params / 1e6, 100 * sel_share)
    if sel_share > EMBEDDING_BUDGET_SHARE:
        # Selection follows the assignment's stated criterion (fertility on held-out text). The
        # parameter cost is reported rather than allowed to override it silently -- but a vocabulary
        # this large would leave little of a 25M-parameter model for anything but embeddings, and
        # that is a Phase-2 decision to make deliberately.
        logger.warning(
            "the fertility elbow selected a vocabulary costing %.0f%% of the Phase-2 parameter "
            "budget in embeddings alone; consider %d instead (see sweep_results.json)",
            100 * sel_share,
            max((r["vocab_size"] for r in results if r["affordable_at_25M"]), default=8_000),
        )

    # Copy the selected sweep model into the final tokenizer/ location (the actual Phase-1
    # deliverable path), rather than leaving it buried in sweep/.
    final_dir = lang_dir / "tokenizer"
    src_model = sweep_dir / f"{language}_{best_algo}_{best_vocab_size}.model"
    src_vocab = sweep_dir / f"{language}_{best_algo}_{best_vocab_size}.vocab"
    dst_model = final_dir / f"{language}_{best_algo}_{best_vocab_size}.model"
    dst_vocab = final_dir / f"{language}_{best_algo}_{best_vocab_size}.vocab"
    dst_model.write_bytes(src_model.read_bytes())
    dst_vocab.write_bytes(src_vocab.read_bytes())

    fig_dir = REPO_ROOT / "report" / "figures"
    plot_fertility_vs_vocab_size(
        {f"{language}-{algo}": list(sizes.items()) for algo, sizes in fertility_by_algo_vocab.items() if sizes},
        fig_dir / f"{language}_tokenizer_fertility_sweep.png",
    )

    # Manual-vs-downloaded token split and the Zipf plot: both were already built
    # (lma/viz/plots.py) but never actually wired into either orchestration script -- the same
    # "built but not called" gap decontamination had (see PHASE1_LOG.md). This is the natural
    # place for them: the final tokenizer needed to convert documents to real token counts only
    # exists from this point on in the pipeline.
    final_tokenizer = Tokenizer(dst_model)
    final_corpus_stats = token_stats(lang_dir, final_tokenizer)
    logger.info("final corpus: %d documents, %d tokens (manual %d / downloaded %d = %.1f%% manual)",
                final_corpus_stats["n_documents"], final_corpus_stats["n_tokens"],
                final_corpus_stats["manual_tokens"], final_corpus_stats["downloaded_tokens"],
                100 * (final_corpus_stats["manual_fraction"] or 0))
    if final_corpus_stats["manual_fraction"] is not None:
        plot_manual_vs_downloaded(
            final_corpus_stats["manual_tokens"], final_corpus_stats["downloaded_tokens"],
            fig_dir / f"{language}_manual_vs_downloaded.png", language.capitalize(),
        )
    freq_table = token_frequency_table(final_tokenizer, val_texts, top_k=1_000_000)
    if freq_table:
        plot_zipf([count for _, count in freq_table], fig_dir / f"{language}_zipf.png", language.capitalize())

    # Token piece-length distribution: lma/tokenizer/analyze.py's token_length_distribution had
    # the same "built, never called" gap as the functions above. Reuses plot_length_histogram
    # (already used for document lengths) rather than a new plot function -- it already accepts
    # a generic list of ints with a unit label, so no duplicated plotting logic is needed.
    piece_length_counts = token_length_distribution(final_tokenizer, val_texts)
    if piece_length_counts:
        piece_lengths_flat = _flatten_counts(piece_length_counts)
        plot_length_histogram(
            piece_lengths_flat, fig_dir / f"{language}_token_piece_length_hist.png",
            language.capitalize(), unit="characters per token piece",
        )

    output = {
        "language": language,
        "used_train_as_val": used_train_as_val,
        "sweep_results": results,
        "selected": {"algorithm": best_algo, "vocab_size": best_vocab_size,
                     "fertility": best_fertility, "rule": selection_rule,
                     "embedding_params": embedding_cost(best_vocab_size)[0],
                     "embedding_share_of_budget": round(embedding_cost(best_vocab_size)[1], 4)},
        "final_model_path": str(dst_model.relative_to(REPO_ROOT)),
        "final_corpus_token_stats": final_corpus_stats,
        "tokenizer_training_sample": {
            "path": str(sample_path.relative_to(REPO_ROOT)),
            "sentences": None if sample_info is None else sample_info.n_sentences,
            "documents_read": None if sample_info is None else sample_info.n_documents_read,
        },
        "phase2_budget": {"params": PHASE2_PARAM_BUDGET, "d_model": PHASE2_D_MODEL,
                          "max_embedding_share": EMBEDDING_BUDGET_SHARE},
    }
    results_path = final_dir / "sweep_results.json"
    results_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s", results_path)
    return output


def main():
    parser = argparse.ArgumentParser(description="Sweep tokenizer vocab size/algorithm for one language.")
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="pin the final vocabulary size instead of taking the fertility elbow "
                             "(the full sweep still runs and is reported)")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default=None,
                        help="pin the final algorithm instead of comparing at the chosen size")
    parser.add_argument("--reuse-models", action="store_true",
                        help="re-evaluate the sweep's existing models instead of retraining them "
                             "(use after a corpus re-split, which changes the metrics but not the "
                             "models)")
    args = parser.parse_args()
    result = run_sweep(args.lang, reuse_models=args.reuse_models,
                       pin_vocab_size=args.vocab_size, pin_algorithm=args.algorithm)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
