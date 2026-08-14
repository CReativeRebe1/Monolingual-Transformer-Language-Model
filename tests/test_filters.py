"""Tests for lma.text.filters -- the Gopher/C4-style heuristic quality cascade."""
import pytest

from lma.text.filters import FilterConfig, evaluate, strip_light_pii

GOOD_HINDI_TEXT = (
    "यह एक सामान्य समाचार लेख है और इसमें कई वाक्य हैं। "
    "यह लेख भारत की राजधानी दिल्ली के बारे में है और यह बताता है कि वहाँ का मौसम कैसा है। "
    "इस लेख को पढ़ने वाले लोग यह जान पाएंगे कि दिल्ली में सर्दियों में क्या होता है और गर्मियों में क्या होता है। "
    "यह जानकारी उन लोगों के लिए उपयोगी है जो दिल्ली आना चाहते हैं और वहाँ के बारे में अधिक जानना चाहते हैं।"
) * 2

GOOD_NEPALI_TEXT = (
    "यो एउटा सामान्य समाचार लेख हो र यसमा धेरै वाक्यहरू छन्। "
    "यो लेख नेपालको राजधानी काठमाडौंको बारेमा हो र यसले बताउँछ कि त्यहाँको मौसम कस्तो छ। "
    "यो लेख पढ्ने मानिसहरूले थाहा पाउनेछन् कि काठमाडौंमा जाडोमा के हुन्छ र गर्मीमा के हुन्छ। "
    "यो जानकारी ती मानिसहरूका लागि उपयोगी छ जो काठमाडौं आउन चाहन्छन्।"
) * 2


def test_good_article_passes_all_rules():
    verdict = evaluate(GOOD_HINDI_TEXT, language="hindi")
    assert verdict.passed, verdict.failed_rules


def test_good_nepali_article_passes():
    verdict = evaluate(GOOD_NEPALI_TEXT, language="nepali")
    assert verdict.passed, verdict.failed_rules


def test_too_short_document_fails():
    verdict = evaluate("यह छोटा है।", language="hindi")
    assert not verdict.passed
    assert "too_short" in verdict.failed_rules


def test_too_few_words_fails_even_if_char_count_ok():
    # A handful of long words: enough characters to clear min_chars, but nowhere near enough
    # distinct words to look like connected prose.
    text = "जानकारीपूर्ण " * 8  # 8 words, each long enough that min_chars=50 is cleared
    verdict = evaluate(text, language="hindi", config=FilterConfig(min_chars=50, min_words=40))
    assert "too_few_words" in verdict.failed_rules


def test_degenerate_single_char_words_fail_mean_word_length_rule():
    text = "क " * 150  # 150 one-char "words": passes the word-count check, fails word-length
    verdict = evaluate(text, language="hindi", config=FilterConfig(min_chars=50))
    assert "mean_word_length_too_low" in verdict.failed_rules


def test_digit_dump_fails_digit_ratio_rule():
    digit_dump = ("१२३४५६७८९० " * 60) + GOOD_HINDI_TEXT
    verdict = evaluate(digit_dump, language="hindi")
    assert "digit_char_ratio_too_high" in verdict.failed_rules


def test_bullet_heavy_document_fails():
    bullets = "\n".join(f"- आइटम {i}" for i in range(40))
    verdict = evaluate(bullets, language="hindi", config=FilterConfig(min_chars=10, min_words=5))
    assert "bullet_or_ellipsis_line_fraction_too_high" in verdict.failed_rules


def test_repeated_lines_fail_the_repetition_rule():
    text = "\n".join(["यह एक पुनरावृत्ति रेखा है।"] * 30)
    verdict = evaluate(text, language="hindi", config=FilterConfig(min_chars=10, min_words=5))
    assert "repeated_line_fraction_too_high" in verdict.failed_rules


def test_boilerplate_phrase_triggers_rejection():
    text = GOOD_HINDI_TEXT + " Cookie Policy: This website uses cookie policy for tracking."
    # note: exact-substring match is case sensitive by design; use exact phrase
    text = GOOD_HINDI_TEXT + " Please review our cookie policy before continuing to browse this site further today."
    verdict = evaluate(text, language="hindi")
    assert "boilerplate_phrase_present" in verdict.failed_rules


def test_no_stopwords_fails_language_evidence_rule():
    # Devanagari-scripted but built from non-function-word tokens only (proper nouns/numbers).
    text = " ".join(["दिल्ली", "मुंबई", "कोलकाता", "चेन्नई"] * 20)
    verdict = evaluate(text, language="hindi", config=FilterConfig(min_chars=10, min_words=5))
    assert "insufficient_stopword_evidence" in verdict.failed_rules


def test_wrong_language_stopword_list_is_used_per_language():
    # Nepali stopwords evaluated against the Hindi list should not satisfy the Hindi check,
    # demonstrating the two languages' filters are genuinely independent (no shared lexicon file).
    nepali_only_functional = " ".join(["छ", "छन", "थियो", "भएको"] * 10 + ["सामग्री"] * 10)
    verdict_as_hindi = evaluate(
        nepali_only_functional, language="hindi", config=FilterConfig(min_chars=10, min_words=5)
    )
    verdict_as_nepali = evaluate(
        nepali_only_functional, language="nepali", config=FilterConfig(min_chars=10, min_words=5)
    )
    assert "insufficient_stopword_evidence" in verdict_as_hindi.failed_rules
    assert "insufficient_stopword_evidence" not in verdict_as_nepali.failed_rules


def test_all_applicable_rules_are_reported_not_just_the_first():
    # A document that is simultaneously too short AND has no stopwords AND has repeated lines.
    text = "अ\nअ\nअ"
    verdict = evaluate(text, language="hindi")
    assert not verdict.passed
    assert len(verdict.failed_rules) >= 2


def test_strip_light_pii_redacts_email_and_phone():
    text = "संपर्क करें: test@example.com या 9876543210 पर।"
    redacted = strip_light_pii(text)
    assert "test@example.com" not in redacted
    assert "9876543210" not in redacted
    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted


def test_strip_light_pii_leaves_normal_text_untouched():
    assert strip_light_pii(GOOD_HINDI_TEXT) == GOOD_HINDI_TEXT
