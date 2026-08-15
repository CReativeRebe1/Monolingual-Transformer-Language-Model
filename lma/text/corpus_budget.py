"""
Sizes how many downloaded-source tokens to pull for a language, given how many manually-collected
tokens (crawled + OCR) it already has, so the final mix stays safely above the assignment's
>=20%-manual floor while still working toward the ~500M-token target.

This is deliberately a calculation, not an action: it does not itself call
lma/crawl/downloaded_corpus.py's ingest(). The Phase-1 plan intentionally defers pulling
downloaded sources in bulk until the manual crawl's near-final size is known (see
report/phase1.md section 2.1 and PHASE1_LOG.md) -- pulling a large batch too early, before the
manual total is known, risks overshooting the downloaded share and having to discard work to get
back under a safe ratio. This module is the tool that makes that later decision a calculation
against the actual manual total at the time, rather than a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TARGET_TOTAL_TOKENS = 500_000_000
DEFAULT_MIN_MANUAL_FRACTION = 0.20
# A cushion above the strict 20% floor: the plan is to pull downloaded tokens once, not
# continuously rebalance, and the manual crawl is still running and will keep adding more manual
# tokens after this calculation is made -- but sizing purely against the *current* manual total
# with no cushion would leave zero margin for e.g. some already-collected manual documents later
# being dropped by decontamination against the Phase-3 reasoning test set. Targeting comfortably
# above the floor (25% by default) rather than exactly at it absorbs that without a second pass.
DEFAULT_SAFETY_MARGIN = 0.05


@dataclass
class DownloadedBudgetRecommendation:
    manual_tokens: int
    recommended_downloaded_tokens: int
    resulting_total_tokens: int
    resulting_manual_fraction: float
    limited_by: str  # "manual_floor" | "target_total" | "already_at_target"


def max_downloaded(
    manual_tokens: int,
    min_manual_fraction: float = DEFAULT_MIN_MANUAL_FRACTION,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    """Largest downloaded-token count D such that manual/(manual+D) stays >= the floor + margin.

    Derivation: manual / (manual + D) >= f  =>  D <= manual * (1 - f) / f, where
    f = min_manual_fraction + safety_margin. Returns 0 (not negative) if manual_tokens is 0, since
    no downloaded tokens can be safely added on top of zero manual tokens without violating any
    positive floor.
    """
    if manual_tokens <= 0:
        return 0
    target_fraction = min_manual_fraction + safety_margin
    if target_fraction <= 0:
        raise ValueError("min_manual_fraction + safety_margin must be > 0")
    if target_fraction >= 1:
        raise ValueError("min_manual_fraction + safety_margin must be < 1 (cannot require >100% manual)")
    return int(manual_tokens * (1 - target_fraction) / target_fraction)


def recommend_pull(
    manual_tokens: int,
    target_total_tokens: int = DEFAULT_TARGET_TOTAL_TOKENS,
    min_manual_fraction: float = DEFAULT_MIN_MANUAL_FRACTION,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> DownloadedBudgetRecommendation:
    """Recommends how many downloaded tokens to pull, respecting both the manual-floor
    constraint and the overall ~500M-token target -- whichever is more restrictive."""
    if manual_tokens < 0:
        raise ValueError("manual_tokens must be >= 0")
    if target_total_tokens <= 0:
        raise ValueError("target_total_tokens must be > 0")

    if manual_tokens >= target_total_tokens:
        # Manual collection alone has already reached (or exceeded) the target; no downloaded
        # tokens are needed at all.
        return DownloadedBudgetRecommendation(
            manual_tokens=manual_tokens,
            recommended_downloaded_tokens=0,
            resulting_total_tokens=manual_tokens,
            resulting_manual_fraction=1.0,
            limited_by="already_at_target",
        )

    floor_limited_downloaded = max_downloaded(
        manual_tokens, min_manual_fraction=min_manual_fraction, safety_margin=safety_margin
    )
    target_limited_downloaded = target_total_tokens - manual_tokens

    if floor_limited_downloaded <= target_limited_downloaded:
        downloaded = floor_limited_downloaded
        limited_by = "manual_floor"
    else:
        downloaded = target_limited_downloaded
        limited_by = "target_total"

    total = manual_tokens + downloaded
    fraction = manual_tokens / total if total > 0 else 1.0
    return DownloadedBudgetRecommendation(
        manual_tokens=manual_tokens,
        recommended_downloaded_tokens=downloaded,
        resulting_total_tokens=total,
        resulting_manual_fraction=fraction,
        limited_by=limited_by,
    )
