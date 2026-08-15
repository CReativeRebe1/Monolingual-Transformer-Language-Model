"""Tests for lma.tokenizer.encode_corpus -- trains a small real tokenizer and encodes a small
real text file end to end (memmap format correctness matters for Phase 2 training)."""
import numpy as np
import pytest

from lma.tokenizer.encode_corpus import document_slice, encode_split, load_encoded_split
from lma.tokenizer.train_sp import Tokenizer, TokenizerTrainConfig, train_from_texts

CORPUS = [
    "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में बताता है।",
    "भारत एक विशाल देश है जिसमें अनेक भाषाएँ और संस्कृतियाँ पाई जाती हैं।",
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं।",
] * 15


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("tok")
    config = TokenizerTrainConfig(algorithm="bpe", vocab_size=350, character_coverage=1.0)
    model_path, _ = train_from_texts(CORPUS, out_dir / "t", config)
    return Tokenizer(model_path)


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("\n".join(CORPUS), encoding="utf-8")
    return path


def test_encode_split_produces_bin_and_idx_files(tokenizer, text_file, tmp_path):
    out_dir = tmp_path / "encoded"
    info = encode_split(tokenizer, text_file, out_dir, "train")
    assert info.bin_path.exists()
    assert info.idx_path.exists()
    assert info.n_documents == len(CORPUS)
    assert info.n_tokens > 0


def test_idx_offsets_are_monotonically_increasing(tokenizer, text_file, tmp_path):
    out_dir = tmp_path / "encoded"
    info = encode_split(tokenizer, text_file, out_dir, "train")
    offsets = np.fromfile(info.idx_path, dtype=np.uint64)
    assert len(offsets) == info.n_documents + 1  # N docs -> N+1 boundary offsets
    assert offsets[0] == 0
    assert offsets[-1] == info.n_tokens
    assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))


def test_load_encoded_split_and_document_slice_roundtrip(tokenizer, text_file, tmp_path):
    out_dir = tmp_path / "encoded"
    encode_split(tokenizer, text_file, out_dir, "train")
    tokens, offsets = load_encoded_split(out_dir / "train.bin", out_dir / "train.idx")

    # Document 0's slice, decoded, should recover (close to) the original first line -- "close
    # to" because an <eos> token was appended and SentencePiece may normalize whitespace.
    doc0_ids = document_slice(tokens, offsets, 0)
    decoded = tokenizer.decode([int(i) for i in doc0_ids if i != tokenizer._sp.eos_id()])
    assert decoded.replace(" ", "") == CORPUS[0].replace(" ", "")


def test_encoded_tokens_match_direct_encoding(tokenizer, text_file, tmp_path):
    out_dir = tmp_path / "encoded"
    encode_split(tokenizer, text_file, out_dir, "train", append_eos=False)
    tokens, offsets = load_encoded_split(out_dir / "train.bin", out_dir / "train.idx")
    doc0_ids = [int(i) for i in document_slice(tokens, offsets, 0)]
    assert doc0_ids == tokenizer.encode(CORPUS[0])


def test_eos_appended_when_requested(tokenizer, text_file, tmp_path):
    out_dir = tmp_path / "encoded"
    info_with_eos = encode_split(tokenizer, text_file, out_dir / "with_eos", "train", append_eos=True)
    info_without_eos = encode_split(tokenizer, text_file, out_dir / "without_eos", "train", append_eos=False)
    assert info_with_eos.n_tokens == info_without_eos.n_tokens + info_with_eos.n_documents


def test_memmap_is_not_fully_loaded_into_memory(tokenizer, text_file, tmp_path):
    out_dir = tmp_path / "encoded"
    encode_split(tokenizer, text_file, out_dir, "train")
    tokens, _ = load_encoded_split(out_dir / "train.bin", out_dir / "train.idx")
    assert isinstance(tokens, np.memmap)


def test_empty_text_file_produces_empty_but_valid_files(tokenizer, tmp_path):
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    out_dir = tmp_path / "encoded"
    info = encode_split(tokenizer, empty_path, out_dir, "empty")
    assert info.n_documents == 0
    assert info.n_tokens == 0
    offsets = np.fromfile(info.idx_path, dtype=np.uint64)
    assert list(offsets) == [0]


def test_vocab_too_large_for_uint16_raises(tmp_path):
    class FakeTokenizer:
        vocab_size = 70_000

    with pytest.raises(ValueError):
        encode_split(FakeTokenizer(), tmp_path / "x.txt", tmp_path / "out", "train")
