"""Tests for lma.text.corpus_budget -- sizing how many downloaded tokens to pull against the
>=20%-manual floor and the ~500M-token target. Pure arithmetic, no I/O."""
import pytest

from lma.text.corpus_budget import (
    DEFAULT_MIN_MANUAL_FRACTION,
    DEFAULT_SAFETY_MARGIN,
    max_downloaded_tokens_for_floor,
    recommend_downloaded_pull,
)


def test_max_downloaded_tokens_for_floor_basic_math():
    # manual=1,000,000, target fraction = 0.20+0.05 = 0.25 -> D = manual*(1-0.25)/0.25 = 3x manual
    d = max_downloaded_tokens_for_floor(1_000_000, min_manual_fraction=0.20, safety_margin=0.05)
    assert d == 3_000_000


def test_max_downloaded_tokens_for_floor_result_actually_satisfies_the_floor():
    manual = 2_345_678
    d = max_downloaded_tokens_for_floor(manual, min_manual_fraction=0.20, safety_margin=0.05)
    resulting_fraction = manual / (manual + d)
    assert resulting_fraction >= 0.20  # the real constraint we actually care about
    assert resulting_fraction == pytest.approx(0.25, abs=1e-3)  # target fraction, up to integer rounding


def test_max_downloaded_tokens_for_floor_zero_manual_returns_zero():
    assert max_downloaded_tokens_for_floor(0) == 0


def test_max_downloaded_tokens_for_floor_rejects_impossible_fraction():
    with pytest.raises(ValueError):
        max_downloaded_tokens_for_floor(1000, min_manual_fraction=0.9, safety_margin=0.2)  # sums to 1.1


def test_recommend_downloaded_pull_small_manual_is_floor_limited():
    # Small manual total relative to the 500M target -> the manual-floor constraint is what
    # actually restricts the pull, not the overall target ceiling.
    rec = recommend_downloaded_pull(manual_tokens=1_000_000)
    assert rec.limited_by == "manual_floor"
    assert rec.recommended_downloaded_tokens == max_downloaded_tokens_for_floor(1_000_000)
    assert rec.resulting_manual_fraction >= DEFAULT_MIN_MANUAL_FRACTION
    assert rec.resulting_total_tokens < 500_000_000  # nowhere near the target with only 1M manual


def test_recommend_downloaded_pull_large_manual_is_target_limited():
    # Manual total is already close to the 500M target -> the target ceiling, not the floor,
    # is what restricts how much more can be added.
    rec = recommend_downloaded_pull(manual_tokens=450_000_000)
    assert rec.limited_by == "target_total"
    assert rec.resulting_total_tokens == 500_000_000
    assert rec.recommended_downloaded_tokens == 50_000_000
    # Manual fraction (90%) is far above the floor -- confirms the floor was NOT the constraint.
    assert rec.resulting_manual_fraction == pytest.approx(0.9, abs=1e-6)


def test_recommend_downloaded_pull_manual_already_at_or_above_target():
    rec = recommend_downloaded_pull(manual_tokens=600_000_000, target_total_tokens=500_000_000)
    assert rec.limited_by == "already_at_target"
    assert rec.recommended_downloaded_tokens == 0
    assert rec.resulting_total_tokens == 600_000_000
    assert rec.resulting_manual_fraction == 1.0


def test_recommend_downloaded_pull_exactly_at_target():
    rec = recommend_downloaded_pull(manual_tokens=500_000_000, target_total_tokens=500_000_000)
    assert rec.limited_by == "already_at_target"
    assert rec.recommended_downloaded_tokens == 0


def test_recommend_downloaded_pull_zero_manual():
    rec = recommend_downloaded_pull(manual_tokens=0, target_total_tokens=500_000_000)
    assert rec.recommended_downloaded_tokens == 0  # can't add downloaded on top of zero manual
    assert rec.resulting_total_tokens == 0


def test_recommend_downloaded_pull_rejects_negative_manual():
    with pytest.raises(ValueError):
        recommend_downloaded_pull(manual_tokens=-5)


def test_recommend_downloaded_pull_rejects_nonpositive_target():
    with pytest.raises(ValueError):
        recommend_downloaded_pull(manual_tokens=1000, target_total_tokens=0)


def test_recommend_downloaded_pull_result_never_violates_the_floor_regardless_of_scale():
    # Property-style check across a spread of realistic manual totals: the recommendation must
    # NEVER produce a resulting_manual_fraction below the true floor (0.20), regardless of which
    # constraint (manual_floor vs target_total) ended up binding.
    for manual in [1, 100, 10_000, 1_000_000, 50_000_000, 200_000_000, 499_999_999]:
        rec = recommend_downloaded_pull(manual_tokens=manual)
        assert rec.resulting_manual_fraction >= DEFAULT_MIN_MANUAL_FRACTION - 1e-9, (
            f"manual={manual} produced fraction={rec.resulting_manual_fraction}, "
            f"below the floor of {DEFAULT_MIN_MANUAL_FRACTION}"
        )


def test_default_safety_margin_and_floor_constants_are_reasonable():
    assert 0 < DEFAULT_MIN_MANUAL_FRACTION < 1
    assert 0 < DEFAULT_SAFETY_MARGIN < 1
    assert DEFAULT_MIN_MANUAL_FRACTION + DEFAULT_SAFETY_MARGIN < 1
