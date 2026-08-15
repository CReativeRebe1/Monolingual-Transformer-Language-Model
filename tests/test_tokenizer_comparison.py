"""Tests for scripts/tokenizer_comparison.py -- generates the Phase-1 report's cross-language
tokenization example table. Another "built (lma/tokenizer/analyze.py's side_by_side_examples)
but never called by any orchestration script" gap: nothing previously generated the ~10
side-by-side tokenization examples the plan explicitly calls for.

Uses small real trained tokenizers (fast, same pattern as tests/test_encode_corpus.py) rather
than mocks, since the whole point is verifying genuinely different vocabularies produce visibly
different tokenizations of the same text.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from lma.tokenizer.train_sp import TokenizerTrainConfig, train_tokenizer_from_texts

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tokenizer_comparison.py"

HINDI_CORPUS = [
    "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में बताता है।",
    "भारत एक विशाल देश है जिसमें अनेक भाषाएँ और संस्कृतियाँ पाई जाती हैं।",
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं।",
] * 15

NEPALI_CORPUS = [
    "काठमाडौं उपत्यकामा आज बिहान हल्का वर्षा भएको छ र तापक्रममा केही कमी आएको छ।",
    "सगरमाथा विश्वको सबैभन्दा अग्लो हिमशिखर हो जुन नेपाल र चीनको सीमामा अवस्थित छ।",
    "नेपालमा दशैं सबैभन्दा ठूलो र महत्त्वपूर्ण चाड मानिन्छ र परिवारका सदस्यहरू भेटघाट गर्छन्।",
] * 15


def _load_module():
    spec = importlib.util.spec_from_file_location("tokenizer_comparison", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup_language(repo_root: Path, language: str, corpus: list[str]):
    lang_dir = repo_root / language
    tok_dir = lang_dir / "tokenizer"
    config = TokenizerTrainConfig(algorithm="bpe", vocab_size=350, character_coverage=1.0)
    model_path, _ = train_tokenizer_from_texts(corpus, tok_dir / f"{language}_bpe_350", config)

    sweep_results = {
        "language": language,
        "selected": {"algorithm": "bpe", "vocab_size": 350},
        "final_model_path": str(model_path.relative_to(repo_root)),
    }
    (tok_dir / "sweep_results.json").write_text(json.dumps(sweep_results), encoding="utf-8")

    splits_dir = lang_dir / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "val.txt").write_text("\n".join(corpus[:5]), encoding="utf-8")
    (splits_dir / "train.txt").write_text("\n".join(corpus), encoding="utf-8")


def test_generate_comparison_table_produces_examples_from_both_languages(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    _setup_language(tmp_path, "hindi", HINDI_CORPUS)
    _setup_language(tmp_path, "nepali", NEPALI_CORPUS)

    result = module.generate_comparison_table()

    assert len(result["examples"]) == 10  # 5 Hindi + 5 Nepali, per N_EXAMPLES_PER_LANGUAGE
    texts = [ex["text"] for ex in result["examples"]]
    assert any(t in HINDI_CORPUS for t in texts)
    assert any(t in NEPALI_CORPUS for t in texts)


def test_every_example_has_both_tokenizations(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    _setup_language(tmp_path, "hindi", HINDI_CORPUS)
    _setup_language(tmp_path, "nepali", NEPALI_CORPUS)

    result = module.generate_comparison_table()

    for ex in result["examples"]:
        assert ex["hindi_tokenizer_pieces"]
        assert ex["nepali_tokenizer_pieces"]
        assert ex["hindi_tokenizer_n_tokens"] > 0
        assert ex["nepali_tokenizer_n_tokens"] > 0


def test_writes_output_file(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    _setup_language(tmp_path, "hindi", HINDI_CORPUS)
    _setup_language(tmp_path, "nepali", NEPALI_CORPUS)

    module.generate_comparison_table()

    out_path = tmp_path / "report" / "tables" / "tokenizer_comparison_examples.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(written["examples"]) == 10


def test_missing_tokenizer_raises_clear_error(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    _setup_language(tmp_path, "nepali", NEPALI_CORPUS)  # hindi deliberately not set up

    with pytest.raises(FileNotFoundError, match="hindi"):
        module.generate_comparison_table()


def test_falls_back_to_train_when_val_is_empty(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    _setup_language(tmp_path, "hindi", HINDI_CORPUS)
    _setup_language(tmp_path, "nepali", NEPALI_CORPUS)
    (tmp_path / "hindi" / "data" / "splits" / "val.txt").write_text("", encoding="utf-8")  # empty val

    result = module.generate_comparison_table()

    assert len(result["examples"]) == 10  # still got 5 Hindi examples, from train fallback
