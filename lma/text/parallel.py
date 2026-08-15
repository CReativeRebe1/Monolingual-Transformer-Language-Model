"""
Parallel per-document cleaning: runs normalize -> language-ID -> heuristic filters -> quality
filter -> PII redaction across every CPU core.

Two design choices matter for speed here:

1. **The stages are fused into one worker call, not four passes.** Running each stage as its own
   parallel map would push every document across the process boundary four times, and pickling
   document text dominates the cost of the actual filtering work. Fusing them means one
   round-trip per document total, and a document rejected at stage 2 costs nothing at stages 3-5.

2. **Only what the parent needs comes back.** Rejected documents return a stage label and rule
   names (a few bytes); document text is returned only for survivors. On a corpus where most
   documents are rejected -- which is the normal case, the Hindi cascade currently keeps well
   under half -- this cuts return-pickling by roughly the rejection rate.

The fastText model cannot be pickled, so each worker loads its own copy once in the initializer
(the model file is memory-mapped by fastText, so this is cheap); the from-scratch char-ngram
language-ID and quality scorers are plain Python objects and are pickled to workers directly.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass

from lma.text.filters import FilterConfig, evaluate, strip_light_pii
from lma.text.langid import FastTextLangID
from lma.text.normalize import normalize

logger = logging.getLogger("lma.parallel")

# Per-worker state, populated once by _init and reused for every document that worker handles.
_STATE: dict = {}


def _init(language, fasttext_code, langid_model, quality_scorer, threshold, filter_config):
    _STATE.update(
        language=language,
        fasttext_code=fasttext_code,
        langid_model=langid_model,
        quality_scorer=quality_scorer,
        threshold=threshold,
        filter_config=filter_config,
        fasttext=FastTextLangID(),   # loaded per worker: not picklable
    )


@dataclass
class DocResult:
    """Outcome for one document. `text` is populated only when stage == 'kept'."""
    index: int
    stage: str                 # "empty" | "lang" | "heuristic" | "quality" | "kept"
    failed_rules: tuple = ()
    text: str | None = None


def _clean_one(item: tuple[int, str]) -> DocResult:
    """Worker: full per-document cascade. Returns as early as possible on rejection."""
    idx, raw_text = item
    s = _STATE

    text = normalize(raw_text)
    if not text:
        return DocResult(idx, "empty")

    ft_lang, ft_conf = s["fasttext"].predict(text)
    if ft_lang != s["fasttext_code"] or ft_conf < 0.5:
        return DocResult(idx, "lang")
    if s["langid_model"] is not None and s["langid_model"].predict(text).label != s["fasttext_code"]:
        return DocResult(idx, "lang")

    verdict = evaluate(text, language=s["language"], config=s["filter_config"])
    if not verdict.passed:
        return DocResult(idx, "heuristic", tuple(verdict.failed_rules))

    if s["quality_scorer"] is not None and s["threshold"] is not None:
        if s["quality_scorer"].perplexity(text) > s["threshold"]:
            return DocResult(idx, "quality")

    return DocResult(idx, "kept", text=strip_light_pii(text))


def clean_documents(
    records: list[dict], language: str, fasttext_code: str, langid_model, quality_scorer,
    threshold, filter_config: FilterConfig | None = None, workers: int | None = None,
    chunksize: int = 64,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Cleans every record in parallel.

    Returns (kept_records, stage_counts, rejection_counts) where stage_counts gives how many
    documents survived each stage (for the funnel chart) and rejection_counts gives per-rule
    heuristic rejection totals (for the report's rejection breakdown). Kept records are the same
    dicts, with "text" replaced by the cleaned/redacted version.
    """
    filter_config = filter_config or FilterConfig()
    workers = workers or max(1, mp.cpu_count() - 1)

    items = [(i, r["text"]) for i, r in enumerate(records)]
    survived = {"language_filtered": 0, "heuristic_filtered": 0, "quality_filtered": 0}
    rejection_counts: dict[str, int] = {}
    kept: list[dict] = []

    logger.info("cleaning %d documents across %d workers", len(items), workers)
    initargs = (language, fasttext_code, langid_model, quality_scorer, threshold, filter_config)
    with mp.Pool(workers, initializer=_init, initargs=initargs) as pool:
        for res in pool.imap_unordered(_clean_one, items, chunksize=chunksize):
            # A document that reaches stage N necessarily passed every earlier stage, so the
            # funnel is built by counting "got at least this far" rather than by tallying each
            # stage separately -- the two must agree, and this way they cannot drift.
            if res.stage in ("heuristic", "quality", "kept"):
                survived["language_filtered"] += 1
            if res.stage in ("quality", "kept"):
                survived["heuristic_filtered"] += 1
            if res.stage == "kept":
                survived["quality_filtered"] += 1
                rec = records[res.index]
                rec["text"] = res.text
                kept.append(rec)
            for rule in res.failed_rules:
                rejection_counts[rule] = rejection_counts.get(rule, 0) + 1

    return kept, survived, rejection_counts
