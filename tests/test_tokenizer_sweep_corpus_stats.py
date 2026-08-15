"""Tests for scripts/tokenizer_sweep.py's compute_final_corpus_token_stats -- another "built but
never wired in" gap found by reviewing the pipeline: lma/viz/plots.py's plot_manual_vs_downloaded
and plot_zipf existed and were tested only indirectly (their own module has no dedicated caller
anywhere), despite the assignment explicitly requiring the manual-vs-downloaded TOKEN split (not
just a document count) to be reported. Computing it requires a trained tokenizer, which only
exists once tokenizer_sweep.py selects one -- this is the function that closes that gap.

Uses a small real SentencePiece tokenizer (fast to train, same pattern as
tests/test_encode_corpus.py) rather than a mock, since the whole point is verifying real
per-document token counts split correctly by source_type.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from lma.tokenizer.train_sp import Tokenizer, TokenizerTrainConfig, train_tokenizer_from_texts

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tokenizer_sweep.py"

CORPUS = [
    "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में बताता है।",
    "भारत एक विशाल देश है जिसमें अनेक भाषाएँ और संस्कृतियाँ पाई जाती हैं।",
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं।",
] * 15


def _load_module():
    spec = importlib.util.spec_from_file_location("tokenizer_sweep", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("tok")
    config = TokenizerTrainConfig(algorithm="bpe", vocab_size=350, character_coverage=1.0)
    model_path, _ = train_tokenizer_from_texts(CORPUS, out_dir / "t", config)
    return Tokenizer(model_path)


def _write_split(splits_dir: Path, split_name: str, records: list[dict]):
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / f"{split_name}.manifest.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"source_type": r["source_type"], "host": r["host"]}, ensure_ascii=False) + "\n")
    with open(splits_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
        for r in records:
            f.write(r["text"] + "\n")


def test_computes_manual_and_downloaded_token_totals_correctly(tmp_path, tiny_tokenizer):
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    splits_dir = lang_dir / "data" / "splits"

    manual_docs = [{"text": CORPUS[0], "source_type": "manual", "host": "amarujala.com"},
                   {"text": CORPUS[1], "source_type": "manual", "host": "amarujala.com"}]
    downloaded_docs = [{"text": CORPUS[2], "source_type": "downloaded", "host": "wikipedia"}]
    _write_split(splits_dir, "train", manual_docs + downloaded_docs)
    _write_split(splits_dir, "val", [])
    _write_split(splits_dir, "test", [])

    stats = module.compute_final_corpus_token_stats(lang_dir, tiny_tokenizer)

    expected_manual_tokens = sum(len(tiny_tokenizer.encode(d["text"])) for d in manual_docs)
    expected_downloaded_tokens = len(tiny_tokenizer.encode(downloaded_docs[0]["text"]))

    assert stats.manual_tokens == expected_manual_tokens
    assert stats.downloaded_tokens == expected_downloaded_tokens
    assert stats.n_tokens == expected_manual_tokens + expected_downloaded_tokens
    assert stats.manual_fraction == pytest.approx(
        expected_manual_tokens / (expected_manual_tokens + expected_downloaded_tokens)
    )


def test_combines_all_three_splits(tmp_path, tiny_tokenizer):
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    splits_dir = lang_dir / "data" / "splits"

    _write_split(splits_dir, "train", [{"text": CORPUS[0], "source_type": "manual", "host": "a.com"}])
    _write_split(splits_dir, "val", [{"text": CORPUS[1], "source_type": "manual", "host": "a.com"}])
    _write_split(splits_dir, "test", [{"text": CORPUS[2], "source_type": "downloaded", "host": "wiki"}])

    stats = module.compute_final_corpus_token_stats(lang_dir, tiny_tokenizer)

    assert stats.n_documents == 3


def test_100_percent_manual_when_no_downloaded_documents(tmp_path, tiny_tokenizer):
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    splits_dir = lang_dir / "data" / "splits"
    _write_split(splits_dir, "train", [{"text": CORPUS[0], "source_type": "manual", "host": "a.com"}])
    _write_split(splits_dir, "val", [])
    _write_split(splits_dir, "test", [])

    stats = module.compute_final_corpus_token_stats(lang_dir, tiny_tokenizer)

    assert stats.downloaded_tokens == 0
    assert stats.manual_fraction == 1.0


def test_missing_splits_are_skipped_not_errored(tmp_path, tiny_tokenizer):
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    splits_dir = lang_dir / "data" / "splits"
    _write_split(splits_dir, "train", [{"text": CORPUS[0], "source_type": "manual", "host": "a.com"}])
    # val.txt/manifest deliberately not written at all -- must not raise.

    stats = module.compute_final_corpus_token_stats(lang_dir, tiny_tokenizer)

    assert stats.n_documents == 1


def test_mismatched_manifest_and_text_line_counts_raises(tmp_path, tiny_tokenizer):
    module = _load_module()
    lang_dir = tmp_path / "hindi"
    splits_dir = lang_dir / "data" / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "train.manifest.jsonl").write_text(
        json.dumps({"source_type": "manual", "host": "a.com"}) + "\n"
        + json.dumps({"source_type": "manual", "host": "a.com"}) + "\n",
        encoding="utf-8",
    )
    (splits_dir / "train.txt").write_text(CORPUS[0] + "\n", encoding="utf-8")  # only 1 line, not 2

    with pytest.raises(ValueError, match="lockstep"):
        module.compute_final_corpus_token_stats(lang_dir, tiny_tokenizer)
