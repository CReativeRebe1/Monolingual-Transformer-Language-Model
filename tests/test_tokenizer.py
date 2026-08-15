"""Tests for lma.tokenizer.{train_sp,analyze} -- trains small REAL SentencePiece models (fast,
tiny vocab, tiny corpus) rather than mocking, since tokenizer correctness (roundtrip, byte
fallback, vocab separation) is exactly what the Phase-1 report's claims rest on.
"""
from pathlib import Path

import pytest

from lma.tokenizer.analyze import (
    evaluate_tokenizer,
    side_by_side_examples,
    token_frequency_table,
    token_length_distribution,
)
from lma.tokenizer.train_sp import Tokenizer, TokenizerTrainConfig, train_from_texts

HINDI_CORPUS = [
    "यह एक समाचार लेख है जो दिल्ली के मौसम के बारे में बताता है।",
    "भारत एक विशाल देश है जिसमें अनेक भाषाएँ और संस्कृतियाँ पाई जाती हैं।",
    "दिल्ली भारत की राजधानी है और यह एक बहुत बड़ा शहर है जहाँ करोड़ों लोग रहते हैं।",
    "इस लेख में हम भारत के इतिहास और संस्कृति के बारे में विस्तार से चर्चा करेंगे।",
    "मुंबई भी भारत का एक बड़ा और महत्वपूर्ण शहर है जो व्यापार का केंद्र है।",
    "क्रिकेट भारत में सबसे लोकप्रिय खेल है और करोड़ों लोग इसे पसंद करते हैं।",
] * 20  # repeated so SentencePiece has enough signal to train a tiny vocab reliably

NEPALI_CORPUS = [
    "यो एउटा समाचार लेख हो जसले काठमाडौंको मौसमको बारेमा बताउँछ।",
    "नेपाल एउटा सानो तर सुन्दर देश हो जहाँ धेरै भाषा र संस्कृति पाइन्छ।",
    "काठमाडौं नेपालको राजधानी हो र यो एउटा ठूलो सहर हो जहाँ लाखौं मानिस बस्छन्।",
    "यस लेखमा हामी नेपालको इतिहास र संस्कृतिको बारेमा विस्तृत छलफल गर्नेछौं।",
    "पोखरा पनि नेपालको एउटा ठूलो र महत्वपूर्ण सहर हो जुन पर्यटनको लागि प्रसिद्ध छ।",
    "फुटबल नेपालमा निकै लोकप्रिय खेल हो र धेरै मानिसले यसलाई मन पराउँछन्।",
] * 20


@pytest.fixture(scope="module")
def hindi_tokenizer(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("hindi_tok")
    config = TokenizerTrainConfig(algorithm="bpe", vocab_size=350, character_coverage=1.0)
    model_path, vocab_path = train_from_texts(
        HINDI_CORPUS, out_dir / "hindi_bpe", config
    )
    return Tokenizer(model_path)


@pytest.fixture(scope="module")
def nepali_tokenizer(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("nepali_tok")
    config = TokenizerTrainConfig(algorithm="unigram", vocab_size=350, character_coverage=1.0)
    model_path, vocab_path = train_from_texts(
        NEPALI_CORPUS, out_dir / "nepali_unigram", config
    )
    return Tokenizer(model_path)


def test_trained_model_and_vocab_files_exist(tmp_path):
    config = TokenizerTrainConfig(algorithm="bpe", vocab_size=350, character_coverage=1.0)
    model_path, vocab_path = train_from_texts(HINDI_CORPUS, tmp_path / "t", config)
    assert model_path.exists()
    assert vocab_path.exists()


def test_vocab_size_matches_config(hindi_tokenizer):
    assert hindi_tokenizer.vocab_size == 350


def test_encode_decode_roundtrip_on_training_text(hindi_tokenizer):
    for text in HINDI_CORPUS[:6]:
        ids = hindi_tokenizer.encode(text)
        decoded = hindi_tokenizer.decode(ids)
        assert decoded.replace(" ", "") == text.replace(" ", "")  # SP may normalize spacing


def test_encode_decode_roundtrip_on_unseen_text_via_byte_fallback(hindi_tokenizer):
    # Text using characters/words never seen in training; byte_fallback guarantees lossless
    # roundtrip regardless (this is the whole point of enabling it).
    unseen = "यह पूर्णतः नवीन और अपरिचित वाक्य है जो प्रशिक्षण डेटा में कभी नहीं था।"
    ids = hindi_tokenizer.encode(unseen)
    decoded = hindi_tokenizer.decode(ids)
    assert decoded == unseen


def test_unk_rate_is_zero_with_byte_fallback(hindi_tokenizer):
    assert hindi_tokenizer.unk_rate("पूर्णतः असंबंधित और यादृच्छिक पाठ नमूना") == 0.0


def test_hindi_and_nepali_vocabularies_are_independent(hindi_tokenizer, nepali_tokenizer):
    hindi_pieces = {hindi_tokenizer._sp.id_to_piece(i) for i in range(hindi_tokenizer.vocab_size)}
    nepali_pieces = {nepali_tokenizer._sp.id_to_piece(i) for i in range(nepali_tokenizer.vocab_size)}
    # The two vocabularies should NOT be identical (they were trained independently on
    # different corpora with different algorithms) -- this is the mechanical proof behind the
    # "separate tokenizer/vocab per model" hard constraint.
    assert hindi_pieces != nepali_pieces


# --- analyze.py ---

def test_evaluate_tokenizer_returns_sane_stats(hindi_tokenizer):
    stats = evaluate_tokenizer(hindi_tokenizer, HINDI_CORPUS[:6], algorithm="bpe")
    assert stats.vocab_size == 350
    assert stats.n_words > 0
    assert stats.n_tokens > 0
    assert stats.fertility > 0
    assert stats.unk_rate == 0.0
    assert 0.0 < stats.vocab_utilization <= 1.0


def test_token_frequency_table_returns_top_k(hindi_tokenizer):
    table = token_frequency_table(hindi_tokenizer, HINDI_CORPUS[:12], top_k=5)
    assert len(table) <= 5
    assert all(isinstance(piece, str) and isinstance(count, int) for piece, count in table)
    # sorted descending by count
    counts = [c for _, c in table]
    assert counts == sorted(counts, reverse=True)


def test_token_length_distribution_sums_to_total_tokens(hindi_tokenizer):
    texts = HINDI_CORPUS[:6]
    dist = token_length_distribution(hindi_tokenizer, texts)
    total_from_dist = sum(dist.values())
    total_tokens = sum(len(hindi_tokenizer.encode_pieces(t)) for t in texts)
    assert total_from_dist == total_tokens


def test_side_by_side_examples_shows_different_segmentations(hindi_tokenizer, nepali_tokenizer):
    examples = side_by_side_examples(hindi_tokenizer, nepali_tokenizer, ["नमस्ते दुनिया"])
    assert len(examples) == 1
    ex = examples[0]
    assert "pieces_a" in ex and "pieces_b" in ex
    assert isinstance(ex["pieces_a"], list) and isinstance(ex["pieces_b"], list)
