"""Tests for the tokenizer pipeline -- validated against HuggingFace `tokenizers` as an independent
reference, in the same spirit as tests/test_minhash.py validating our MinHash against `datasketch`.

Two different things get checked, and the distinction matters:

* **Our metrics** (fertility, chars/token, UNK rate) are recomputed here from first principles and
  compared against `evaluate_tokenizer`. A metric that is quietly wrong would silently pick the wrong
  vocabulary size, and no amount of internal consistency would reveal it.
* **Our trained model** is compared against an independently trained HuggingFace BPE on the same
  text at the same vocabulary size. Two BPE implementations will not produce identical merges --
  SentencePiece works on a unigram-initialised character stream with byte fallback, HuggingFace on
  pre-tokenised words -- so the assertion is that fertility lands in the same neighbourhood, not that
  the outputs match. A gross divergence would mean our training call is misconfigured.

The regression that motivated the sentence-splitting tests is real and was silent: SentencePiece
defaults `max_sentence_length` to 4192 BYTES and discards longer lines without warning. Devanagari
costs 3 bytes per character, so whole documents were being dropped from tokenizer training.
"""
import pytest

from lma.tokenizer.sample import (
    MAX_SENTENCE_CHARS,
    build_training_sample,
    split_sentences,
)
from lma.tokenizer.analyze import evaluate_tokenizer
from lma.tokenizer.train_sp import Tokenizer, TokenizerTrainConfig, train_from_texts

SENTENCES = [
    "दिल्ली भारत की राजधानी है और यहाँ करोड़ों लोग रहते हैं।",
    "यह शहर अपनी ऐतिहासिक इमारतों के लिए जाना जाता है।",
    "सरकार ने प्रदूषण कम करने के लिए कई कदम उठाए हैं।",
    "मेट्रो का विस्तार किया गया है और यातायात सुधरा है।",
    "विशेषज्ञों का मानना है कि सार्वजनिक परिवहन बढ़ाना चाहिए।",
    "अदालत ने इस मामले में सख्त रुख अपनाया है।",
    "नगर निगम ने निर्माण कार्य पर निगरानी रखने का फैसला किया।",
    "हर साल लाखों पर्यटक इस शहर में आते हैं।",
]
CORPUS = [" ".join(SENTENCES[i % len(SENTENCES)] for i in range(j, j + 4)) for j in range(200)]


# ------------------------------------------------------------------ sentence sampling

def test_documents_are_split_on_danda():
    parts = split_sentences("पहला वाक्य है। दूसरा वाक्य है। तीसरा वाक्य है।")
    assert len(parts) == 3
    assert all(p.endswith("।") for p in parts)


def test_a_document_with_no_danda_is_hard_wrapped():
    """A table or a broken extraction has no sentence boundary at all. Left whole it would exceed
    SentencePiece's line limit and be dropped from training in silence."""
    parts = split_sentences("शब्द " * 2000)
    assert len(parts) > 1
    assert all(len(p) <= MAX_SENTENCE_CHARS for p in parts)


def test_sample_file_has_one_sentence_per_line(tmp_path):
    split = tmp_path / "train.txt"
    split.write_text("\n".join(CORPUS[:20]), encoding="utf-8")
    out = tmp_path / "sample.txt"
    info = build_training_sample(split, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert info.n_sentences == len(lines) > 20        # more sentences than documents
    assert all(len(line.encode("utf-8")) < 4192 for line in lines), \
        "every line must fit under SentencePiece's default 4192-byte limit"


def test_sample_stride_spans_the_whole_corpus(tmp_path):
    """Splits are ordered manual-then-downloaded, so taking a prefix would train the tokenizer on
    manual text only. Stride sampling must reach the end of the file."""
    split = tmp_path / "train.txt"
    split.write_text("\n".join(f"दस्तावेज़ संख्या {i} यहाँ है और यह एक वाक्य है।" for i in range(100)),
                     encoding="utf-8")
    out = tmp_path / "sample.txt"
    build_training_sample(split, out, stride=10)
    text = out.read_text(encoding="utf-8")
    assert "संख्या 0 " in text and "संख्या 90 " in text


# ------------------------------------------------------------------ metrics vs first principles

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok") / "hi_test"
    model_path, _ = train_from_texts(
        # 600, not a smaller round number: byte_fallback adds 256 byte pieces to the vocabulary
        # before any merges, so the floor here is 256 + distinct characters + 4 special tokens.
        # SentencePiece rejects anything below that with "Vocabulary size is smaller than
        # required_chars", which is what a 300-token test vocabulary hit.
        SENTENCES * 40, out, TokenizerTrainConfig(algorithm="bpe", vocab_size=600,
                                                  character_coverage=1.0)
    )
    return Tokenizer(model_path)


def test_fertility_matches_a_first_principles_recount(trained):
    stats = evaluate_tokenizer(trained, CORPUS[:50], algorithm="bpe")
    n_words = sum(len(t.split()) for t in CORPUS[:50])
    n_tokens = sum(len(trained.encode(t)) for t in CORPUS[:50])
    assert stats.fertility == pytest.approx(n_tokens / n_words)
    assert stats.chars_per_token == pytest.approx(
        sum(len(t) for t in CORPUS[:50]) / n_tokens
    )


def test_byte_fallback_gives_a_zero_unk_rate_even_on_unseen_script(trained):
    """byte_fallback is why UNK is 0% at every vocab size, which is why the sweep selects on
    fertility instead. Checked on text the tokenizer never saw, including a different script."""
    stats = evaluate_tokenizer(trained, ["完全に未知のテキスト ✨ unseen text"], algorithm="bpe")
    assert stats.unk_rate == 0.0


def test_encode_decode_round_trips(trained):
    for text in CORPUS[:5]:
        assert trained.decode(trained.encode(text)) == text


# ------------------------------------------------------------------ independent reference

def test_fertility_agrees_with_an_independently_trained_huggingface_bpe(trained):
    """Cross-check against HuggingFace `tokenizers`, trained on the same text at the same vocab size.

    The two implementations differ by construction -- SentencePiece uses byte fallback over a raw
    character stream, HuggingFace pre-tokenises on whitespace -- so the check is that fertility lands
    in the same neighbourhood, not that the segmentations match. A large gap would mean our training
    call is misconfigured (which is exactly what the silent max_sentence_length truncation was).
    """
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    hf = HFTokenizer(BPE(unk_token="[UNK]"))
    hf.pre_tokenizer = Whitespace()
    hf.train_from_iterator(SENTENCES * 40, BpeTrainer(vocab_size=600, special_tokens=["[UNK]"]))

    ours = evaluate_tokenizer(trained, CORPUS[:50], algorithm="bpe").fertility
    theirs = sum(len(hf.encode(t).ids) for t in CORPUS[:50]) / sum(
        len(t.split()) for t in CORPUS[:50]
    )
    assert ours == pytest.approx(theirs, rel=0.5), (
        f"our fertility {ours:.3f} vs HuggingFace BPE {theirs:.3f} at the same vocab size -- "
        "a gap this large suggests a misconfigured trainer, not an implementation difference"
    )
