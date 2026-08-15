"""
Streaming ingestion of public downloaded corpora (Wikipedia, OSCAR/CulturaX, IndicCorp v2)
via the HuggingFace `datasets` library, in streaming mode so multi-hundred-GB source datasets
are never fully downloaded to disk -- only the token budget actually needed is materialized.

This is explicitly the "downloaded" side of the manual-vs-downloaded split the Phase-1 report
must report: every document ingested here is tagged source_type="downloaded" in its output
record, in contrast to the crawler (lma/crawl/async_crawler.py), whose output records are
tagged source_type="manual". Both write the same
JSONL.gz shard schema so the downstream cleaning cascade (lang-id -> heuristic filters -> quality
filter -> dedup) treats all three sources uniformly regardless of provenance.

Some of the datasets named in the Phase-1 plan (IndicCorp v2, parts of CulturaX) are gated on
HuggingFace and require the runtime to be logged in with an account that has accepted the
dataset's terms (`huggingface-cli login`, done once on the Kaggle/Colab runtime that performs the
real corpus-scale pull -- not on this development machine). `list_available_configs()` and
`ingest()`'s error handling surface a clear, actionable message rather than a raw stack trace
when a config is gated/unavailable, so a partial run against the sources that ARE open (Wikipedia
in particular, which is fully open) still succeeds and is reported honestly in the dataset stats.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("lma.downloaded_corpus")

# (dataset repo id, config/subset name, text field name, split) per source. Each entry is tried
# independently; a gated or unavailable source is skipped with a logged reason rather than
# aborting the whole ingestion run.
SOURCE_CONFIGS: dict[str, list[dict]] = {
    "hindi": [
        {"name": "wikipedia", "repo_id": "wikimedia/wikipedia", "config": "20231101.hi", "text_field": "text", "split": "train"},
        {"name": "oscar", "repo_id": "oscar-corpus/OSCAR-2301", "config": "hi", "text_field": "text", "split": "train"},
        {"name": "culturax", "repo_id": "uonlp/CulturaX", "config": "hi", "text_field": "text", "split": "train"},
        # ai4bharat/IndicCorpV2 has ONE builder config ("indiccorp_v2") and selects the
        # language via the `split` argument (e.g. "hin_Deva"/"npi_Deva"), unlike the other
        # sources here which select language via `config` -- verified directly against the
        # live dataset, not assumed from documentation.
        {"name": "indiccorp_v2", "repo_id": "ai4bharat/IndicCorpV2", "config": "indiccorp_v2", "text_field": "text", "split": "hin_Deva"},
    ],
    "nepali": [
        {"name": "wikipedia", "repo_id": "wikimedia/wikipedia", "config": "20231101.ne", "text_field": "text", "split": "train"},
        {"name": "oscar", "repo_id": "oscar-corpus/OSCAR-2301", "config": "ne", "text_field": "text", "split": "train"},
        {"name": "culturax", "repo_id": "uonlp/CulturaX", "config": "ne", "text_field": "text", "split": "train"},
        {"name": "indiccorp_v2", "repo_id": "ai4bharat/IndicCorpV2", "config": "indiccorp_v2", "text_field": "text", "split": "npi_Deva"},
    ],
}


def _doc_id(source_name: str, text: str) -> str:
    return f"{source_name}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def _rough_token_estimate(text: str) -> int:
    """Whitespace-token count -- a cheap proxy used only to track progress against a budget
    during ingestion; the authoritative token count is computed post-tokenizer in Phase 1's
    corpus statistics, once the real tokenizer (trained in lma/tokenizer/) exists."""
    return len(text.split())


@dataclass
class IngestionSourceReport:
    source_name: str
    repo_id: str
    config: str
    status: str  # "ok" | "unavailable" | "gated_or_error"
    documents_written: int
    approx_tokens_written: int
    error: str | None = None


def ingest_source(
    source_cfg: dict,
    out_dir: Path,
    token_budget: int,
    min_chars: int = 200,
) -> IngestionSourceReport:
    """Stream one HF dataset source until token_budget (whitespace-token estimate) is reached."""
    from datasets import load_dataset  # imported lazily: heavy import, only needed here

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"downloaded_{source_cfg['name']}.jsonl.gz"

    written_docs = 0
    written_tokens = 0
    try:
        ds = load_dataset(
            source_cfg["repo_id"],
            source_cfg.get("config"),
            split=source_cfg["split"],
            streaming=True,
        )
    except Exception as e:  # noqa: BLE001 - deliberately broad: many distinct failure modes
        # (gated dataset, network error, config renamed upstream) all reduce to the same
        # actionable outcome: skip this source, keep going, report why.
        logger.warning(
            "source %s (%s/%s) unavailable: %s",
            source_cfg["name"], source_cfg["repo_id"], source_cfg.get("config"), e,
        )
        return IngestionSourceReport(
            source_name=source_cfg["name"],
            repo_id=source_cfg["repo_id"],
            config=str(source_cfg.get("config")),
            status="gated_or_error",
            documents_written=0,
            approx_tokens_written=0,
            error=str(e),
        )

    with gzip.open(shard_path, "wt", encoding="utf-8") as f:
        for example in ds:
            text = (example.get(source_cfg["text_field"]) or "").strip()
            if len(text) < min_chars:
                continue
            record = {
                "url": None,
                "host": source_cfg["repo_id"],
                "title": example.get("title"),
                "text": text,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "host_source_name": source_cfg["name"],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source_type": "downloaded",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_docs += 1
            written_tokens += _rough_token_estimate(text)
            if written_tokens >= token_budget:
                break

    return IngestionSourceReport(
        source_name=source_cfg["name"],
        repo_id=source_cfg["repo_id"],
        config=str(source_cfg.get("config")),
        status="ok",
        documents_written=written_docs,
        approx_tokens_written=written_tokens,
    )


def ingest(
    language: str,
    out_dir: Path,
    token_budget_per_source: int,
    sources: list[str] | None = None,
) -> list[IngestionSourceReport]:
    """Ingest one or more downloaded sources for a language, each up to its own token budget.

    sources: subset of SOURCE_CONFIGS[language] names to attempt (default: all configured).
    Reports are returned for every attempted source, including failed/gated ones, so the
    Phase-1 dataset-statistics report can show exactly what was and was not available.
    """
    if language not in SOURCE_CONFIGS:
        raise ValueError(f"no downloaded-source config for language {language!r}")
    configs = SOURCE_CONFIGS[language]
    if sources is not None:
        configs = [c for c in configs if c["name"] in sources]

    reports = []
    for cfg in configs:
        logger.info("ingesting %s (%s/%s)...", cfg["name"], cfg["repo_id"], cfg.get("config"))
        report = ingest_source(cfg, out_dir, token_budget=token_budget_per_source)
        reports.append(report)
        logger.info(
            "%s: status=%s docs=%d approx_tokens=%d",
            cfg["name"], report.status, report.documents_written, report.approx_tokens_written,
        )
    return reports
