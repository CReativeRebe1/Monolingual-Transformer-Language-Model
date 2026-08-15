"""Tests for lma.text.quality.QualityScorer -- the char-n-gram learned quality filter."""
import pytest

from lma.text.quality import QualityScorer, choose_threshold

TRUSTED_HINDI = [
    "यह एक सामान्य वाक्य है जो अच्छी तरह से लिखा गया है और इसमें कोई त्रुटि नहीं है।",
    "भारत एक विशाल देश है जिसमें अनेक भाषाएँ और संस्कृतियाँ पाई जाती हैं।",
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं।",
    "इस लेख में हम भारत के इतिहास और संस्कृति के बारे में विस्तार से चर्चा करेंगे।",
] * 3


def test_perplexity_raises_before_fit():
    scorer = QualityScorer()
    with pytest.raises(RuntimeError):
        scorer.perplexity("कुछ पाठ")


def test_perplexity_of_trusted_text_is_lower_than_gibberish():
    scorer = QualityScorer(n=4)
    scorer.fit(TRUSTED_HINDI)
    trusted_ppl = scorer.perplexity("दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है।")
    gibberish_ppl = scorer.perplexity("xzqwkj09 ###$$$ 一二三四五六 @@@@ %%%^^^")
    assert trusted_ppl < gibberish_ppl


def test_perplexity_of_seen_text_is_lower_than_novel_but_related_text():
    scorer = QualityScorer(n=4)
    scorer.fit(TRUSTED_HINDI)
    seen_ppl = scorer.perplexity(TRUSTED_HINDI[0])
    # A related-domain but unseen sentence should still score reasonably (finite, not huge).
    novel_ppl = scorer.perplexity("मुंबई भी भारत का एक बड़ा और महत्वपूर्ण शहर है।")
    assert seen_ppl < novel_ppl
    assert novel_ppl < float("inf")


def test_empty_text_has_infinite_perplexity():
    scorer = QualityScorer(n=4)
    scorer.fit(TRUSTED_HINDI)
    assert scorer.perplexity("") == float("inf")


def test_fit_perplexity_distribution_returns_one_value_per_text():
    scorer = QualityScorer(n=4)
    scorer.fit(TRUSTED_HINDI)
    ppls = scorer.fit_ppl_dist(TRUSTED_HINDI[:2])
    assert len(ppls) == 2
    assert all(p > 0 for p in ppls)


def test_choose_threshold_picks_a_percentile():
    values = list(range(1, 101))  # 1..100
    threshold = choose_threshold([float(v) for v in values], percentile=99.0)
    assert 95 <= threshold <= 100


def test_choose_threshold_empty_raises():
    with pytest.raises(ValueError):
        choose_threshold([])


def test_larger_n_still_produces_finite_scores_on_short_text():
    scorer = QualityScorer(n=6)
    scorer.fit(TRUSTED_HINDI)
    ppl = scorer.perplexity("यह छोटा वाक्य है।")
    assert ppl > 0
    assert ppl < float("inf")
