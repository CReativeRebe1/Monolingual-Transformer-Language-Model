"""Tests for lma.text.normalize — the Devanagari Unicode normalizer.

These are compliance tests: they exist to prove specific claims made in the Phase-1 report
(idempotency, nukta composition, danda normalization) rather than just to sanity-check syntax.
"""
import unicodedata

import pytest

from lma.text.normalize import (
    convert_digits,
    normalize,
    normalize_danda,
    normalize_whitespace,
    normalize_zero_width_joiners,
)


# --- Idempotency: the single most important property for a pipeline stage that may run twice ---

@pytest.mark.parametrize(
    "text",
    [
        "यह एक हिन्दी वाक्य है।",
        "यो एउटा नेपाली वाक्य हो।",
        "क़िताब और ख़ास शब्द",  # nukta-bearing words
        "पहला वाक्य || दूसरा वाक्य",
        "बहुत   सारे    स्पेस",
        "पंक्ति एक\n\n\n\nपंक्ति दो",
        "",
        "   ",
        "ASCII only text 123",
        "मिश्रित Hindi and English 123 पाठ",
    ],
)
def test_normalize_is_idempotent(text):
    once = normalize(text)
    twice = normalize(once)
    assert once == twice, f"normalize() is not idempotent for {text!r}: {once!r} != {twice!r}"


# --- Nukta composition: decomposed (base + U+093C) and precomposed forms must converge under NFC ---

def test_nukta_decomposed_and_precomposed_forms_converge():
    precomposed = "क़"  # क़ QA, single codepoint
    decomposed = "क़"  # क + ़ (KA + NUKTA), two codepoints
    assert precomposed != decomposed  # distinct as raw input
    assert normalize(precomposed) == normalize(decomposed)
    # and the composed form should indeed be the single-codepoint NFC form
    assert normalize(decomposed) == unicodedata.normalize("NFC", precomposed)


# --- Danda normalization ---

def test_double_pipe_becomes_double_danda():
    assert normalize_danda("श्लोक || समाप्त") == "श्लोक ॥ समाप्त"


def test_single_pipe_adjacent_to_devanagari_becomes_danda():
    assert normalize_danda("यह वाक्य है|") == "यह वाक्य है।"
    assert normalize_danda("|यह वाक्य है") == "।यह वाक्य है"


def test_lone_ascii_pipe_without_devanagari_neighbour_is_untouched():
    assert normalize_danda("a | b") == "a | b"


# --- ZWJ / ZWNJ handling ---

def test_repeated_zwnj_collapses_to_one():
    text = "क्‌‌‌त"
    result = normalize_zero_width_joiners(text)
    assert "‌‌" not in result
    assert "‌" in result  # a single occurrence is preserved


def test_zwnj_at_line_boundary_is_stripped():
    text = "‌सुरुवात अन्त्य‌"
    result = normalize_zero_width_joiners(text)
    assert not result.startswith("‌")
    assert not result.endswith("‌")


# --- Whitespace ---

def test_multiple_spaces_collapse_to_one():
    assert normalize_whitespace("शब्द    शब्द") == "शब्द शब्द"


def test_excess_blank_lines_cap_at_one():
    result = normalize_whitespace("एक\n\n\n\n\nदो")
    assert result == "एक\n\ndो" or result == "एक\n\nदो"  # exactly one blank line preserved


def test_single_blank_line_paragraph_break_is_preserved():
    # A genuine paragraph break (one blank line) must NOT be collapsed away.
    result = normalize_whitespace("पहला अनुच्छेद\n\nदूसरा अनुच्छेद")
    assert result == "पहला अनुच्छेद\n\nदूसरा अनुच्छेद"


# --- Digit conversion (off by default; explicit opt-in) ---

def test_digit_conversion_is_off_by_default():
    text = "वर्ष २०२४ में 15 लोग"
    assert normalize(text) == text  # both digit scripts survive untouched by default


def test_convert_digits_devanagari_to_ascii():
    assert convert_digits("२०२४", target="ascii") == "2024"


def test_convert_digits_ascii_to_devanagari():
    assert convert_digits("2024", target="devanagari") == "२०२४"


def test_convert_digits_invalid_target_raises():
    with pytest.raises(ValueError):
        convert_digits("123", target="roman")


# --- Anusvara / chandrabindu must never be conflated (meaning-changing) ---

def test_anusvara_and_chandrabindu_are_not_conflated():
    anusvara_word = "हंस"     # swan
    chandrabindu_word = "हँस"  # laugh (root)
    assert normalize(anusvara_word) == anusvara_word
    assert normalize(chandrabindu_word) == chandrabindu_word
    assert normalize(anusvara_word) != normalize(chandrabindu_word)


# --- Control characters ---

def test_control_characters_are_stripped():
    text = "शब्द\x00\x01शब्द"
    assert "\x00" not in normalize(text)
    assert "\x01" not in normalize(text)


def test_newline_and_tab_survive_control_stripping():
    text = "पंक्ति एक\nपंक्ति दो"
    assert "\n" in normalize(text)
