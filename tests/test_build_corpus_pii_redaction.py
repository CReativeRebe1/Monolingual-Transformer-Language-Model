"""Tests for scripts/build_corpus.py's redact_pii -- another "built but never called" gap:
lma/text/filters.py's strip_light_pii was implemented and unit-tested in isolation
(tests/test_filters.py) but never actually invoked anywhere in the pipeline, meaning a real email
address or phone number appearing in crawled text would have flowed straight through to the
committed train/val/test splits verbatim.
"""
from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_corpus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_redacts_email_addresses():
    module = _load_module()
    records = [{"text": "संपर्क करें: contact@example.com के लिए अधिक जानकारी के लिए।"}]
    result = module.redact_pii(records)
    assert "contact@example.com" not in result[0]["text"]
    assert "[EMAIL]" in result[0]["text"]


def test_redacts_phone_numbers():
    module = _load_module()
    records = [{"text": "मुझे 9876543210 पर कॉल करें कृपया धन्यवाद।"}]
    result = module.redact_pii(records)
    assert "9876543210" not in result[0]["text"]
    assert "[PHONE]" in result[0]["text"]


def test_returns_same_list_object_mutated_in_place():
    module = _load_module()
    records = [{"text": "clean text with nothing to redact at all"}]
    result = module.redact_pii(records)
    assert result is records  # same list object, per the function's documented semantics


def test_does_not_touch_text_with_no_pii():
    module = _load_module()
    original = "यह एक सामान्य वाक्य है जिसमें कोई व्यक्तिगत जानकारी नहीं है।"
    records = [{"text": original}]
    result = module.redact_pii(records)
    assert result[0]["text"] == original


def test_handles_multiple_records_independently():
    module = _load_module()
    records = [
        {"text": "email me at test@test.com"},
        {"text": "completely clean document text"},
        {"text": "call 9998887770 now"},
    ]
    result = module.redact_pii(records)
    assert "[EMAIL]" in result[0]["text"]
    assert result[1]["text"] == "completely clean document text"
    assert "[PHONE]" in result[2]["text"]


def test_empty_records_list():
    module = _load_module()
    assert module.redact_pii([]) == []
