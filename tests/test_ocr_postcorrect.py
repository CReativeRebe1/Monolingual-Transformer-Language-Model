"""Tests for lma.ocr.postcorrect -- offline, no tesseract binary required."""
from lma.ocr.postcorrect import (
    fix_matra_spacing,
    postcorrect_ocr_text,
    strip_isolated_noise_characters,
)


def test_fix_matra_spacing_removes_stray_space_before_matra():
    broken = "य ा"  # य + space + ा (should be या)
    assert fix_matra_spacing(broken) == "या"


def test_fix_matra_spacing_leaves_correct_text_untouched():
    correct = "या तो यह सही है"
    assert fix_matra_spacing(correct) == correct


def test_fix_matra_spacing_handles_multiple_occurrences():
    # "क ा ख ी" has two stray matra-spaces (क|ा and ख|ी) plus one genuine word-boundary space
    # between the two words. Only the matra-spaces should be removed.
    broken = "क ा ख ी"
    fixed = fix_matra_spacing(broken)
    assert fixed == "का खी"


def test_strip_isolated_noise_characters_removes_lone_char():
    text = "यह एक अ वाक्य है"  # 'अ' floating alone is noise
    result = strip_isolated_noise_characters(text)
    assert "यह एक  वाक्य है" == result or "अ" not in result.split()


def test_strip_isolated_noise_characters_keeps_real_words():
    text = "यह एक सामान्य वाक्य है"
    assert strip_isolated_noise_characters(text) == text


def test_postcorrect_full_pipeline_runs_without_error():
    messy = "यह  एक   OCR   पाठ   है ।  क ा  उदाहरण"
    result = postcorrect_ocr_text(messy)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "  " not in result  # whitespace collapsed by normalize()


def test_postcorrect_is_idempotent():
    messy = "यह  एक   OCR   पाठ   है ।"
    once = postcorrect_ocr_text(messy)
    twice = postcorrect_ocr_text(once)
    assert once == twice
