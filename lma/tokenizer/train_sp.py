"""
Tokenizer training: SentencePiece BPE and Unigram, trained from scratch, one vocabulary per
language, never shared between Model H and Model L. No pretrained tokenizer is used anywhere in
this project -- every .model/.vocab file this module produces is trained on this project's own
corpus, satisfying the assignment's hard constraint on tokenizers.

`byte_fallback=True` is used throughout: any character sequence not covered by the trained
vocabulary at character_coverage=0.9995 falls back to raw UTF-8 byte tokens rather than emitting
<unk>. This is a deliberate, reported design choice (Phase-1 report states this explicitly) --
it is *why* the corpus's UNK rate is exactly 0%, not an accident of a lucky vocabulary, and it
guarantees every string is losslessly encodable/decodable regardless of vocabulary size.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


@dataclass
class TokenizerTrainConfig:
    algorithm: str  # "bpe" or "unigram"
    vocab_size: int
    character_coverage: float = 0.9995
    byte_fallback: bool = True
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3


def train_tokenizer(
    corpus_paths: list[Path],
    out_prefix: Path,
    config: TokenizerTrainConfig,
    input_sentence_size: int = 5_000_000,
    max_sentence_length: int = 8_192,
) -> tuple[Path, Path]:
    """Trains a SentencePiece model from one or more plaintext corpus files (one doc per line).

    Returns (model_path, vocab_path) -- SentencePiece writes <out_prefix>.model/.vocab itself.
    input_sentence_size caps how many lines SentencePiece samples for vocabulary estimation
    (not a hard cap on corpus size used downstream); this keeps tokenizer training tractable on
    a >100M-token corpus without needing the entire corpus in memory at once, which is exactly
    the reason SentencePiece exposes this knob.

    max_sentence_length is passed EXPLICITLY because SentencePiece's default is 4192 bytes and it
    discards longer lines in silence. Devanagari costs 3 bytes per character in UTF-8, so the
    default drops anything past ~1,400 characters -- and this project's splits are written one
    document per line, averaging ~1.6 KB. A large share of the corpus was therefore never seen by
    the trainer, with no error and no warning. Callers should feed sentence-split input
    (lma/tokenizer/sample.py) and leave this at the raised default as a second line of defence.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_type = "bpe" if config.algorithm == "bpe" else "unigram"

    spm.SentencePieceTrainer.train(
        input=",".join(str(p) for p in corpus_paths),
        model_prefix=str(out_prefix),
        vocab_size=config.vocab_size,
        model_type=model_type,
        character_coverage=config.character_coverage,
        byte_fallback=config.byte_fallback,
        pad_id=config.pad_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        unk_id=config.unk_id,
        input_sentence_size=input_sentence_size,
        max_sentence_length=max_sentence_length,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )
    return out_prefix.with_suffix(".model"), out_prefix.with_suffix(".vocab")


def train_from_texts(
    texts: list[str], out_prefix: Path, config: TokenizerTrainConfig
) -> tuple[Path, Path]:
    """Convenience wrapper for training directly from an in-memory list of documents (used in
    tests and small-scale experiments; the real Phase-1 pipeline uses train_tokenizer() against
    the on-disk corpus shards so the whole corpus never needs to fit in memory at once)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")
        tmp_path = Path(f.name)
    try:
        return train_tokenizer([tmp_path], out_prefix, config)
    finally:
        tmp_path.unlink(missing_ok=True)


class Tokenizer:
    """Thin, explicit wrapper around a trained SentencePiece model for encode/decode use downstream
    (corpus encoding, evaluation, generation) -- keeps the raw sentencepiece API out of caller code."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @property
    def vocab_size(self) -> int:
        return self._sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)

    def encode_pieces(self, text: str) -> list[str]:
        return self._sp.encode(text, out_type=str)

    def unk_rate(self, text: str) -> float:
        ids = self.encode(text)
        if not ids:
            return 0.0
        unk_id = self._sp.unk_id()
        return sum(1 for i in ids if i == unk_id) / len(ids)
