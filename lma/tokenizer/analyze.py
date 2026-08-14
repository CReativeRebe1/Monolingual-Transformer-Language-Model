"""
Tokenizer analysis: the metrics Phase-1 uses to choose a vocabulary size and algorithm, per
language, on held-out text. Every function here operates on an already-trained Tokenizer
(lma/tokenizer/train_sp.py) -- this module never trains anything, only measures.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from lma.tokenizer.train_sp import Tokenizer


@dataclass
class TokenizerStats:
    vocab_size: int
    algorithm: str
    n_documents: int
    n_words: int
    n_tokens: int
    n_chars: int
    fertility: float          # tokens per word -- lower is more "word-like" segmentation
    chars_per_token: float    # compression: higher means fewer tokens needed per character
    unk_rate: float           # fraction of tokens that are <unk> (expected ~0 with byte_fallback)
    vocab_utilization: float  # fraction of the vocabulary that appears at least once in the sample


def evaluate_tokenizer(
    tokenizer: Tokenizer, held_out_texts: list[str], algorithm: str
) -> TokenizerStats:
    """Computes the full metric set used to pick a vocab size (see Phase-1 report's sweep table)."""
    n_words = 0
    n_tokens = 0
    n_chars = 0
    n_unk = 0
    token_id_counts: Counter[int] = Counter()
    unk_id = tokenizer._sp.unk_id()

    for text in held_out_texts:
        n_words += len(text.split())
        n_chars += len(text)
        ids = tokenizer.encode(text)
        n_tokens += len(ids)
        n_unk += sum(1 for i in ids if i == unk_id)
        token_id_counts.update(ids)

    fertility = n_tokens / n_words if n_words else float("nan")
    chars_per_token = n_chars / n_tokens if n_tokens else float("nan")
    unk_rate = n_unk / n_tokens if n_tokens else float("nan")
    vocab_utilization = len(token_id_counts) / tokenizer.vocab_size

    return TokenizerStats(
        vocab_size=tokenizer.vocab_size,
        algorithm=algorithm,
        n_documents=len(held_out_texts),
        n_words=n_words,
        n_tokens=n_tokens,
        n_chars=n_chars,
        fertility=fertility,
        chars_per_token=chars_per_token,
        unk_rate=unk_rate,
        vocab_utilization=vocab_utilization,
    )


def token_frequency_table(
    tokenizer: Tokenizer, texts: list[str], top_k: int = 50
) -> list[tuple[str, int]]:
    """Top-k most frequent token pieces (by surface form, not id) over a text sample."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenizer.encode_pieces(text))
    return counts.most_common(top_k)


def token_length_distribution(tokenizer: Tokenizer, texts: list[str]) -> Counter[int]:
    """Histogram of token piece lengths (in characters), for the Zipf/length-distribution plots."""
    lengths: Counter[int] = Counter()
    for text in texts:
        for piece in tokenizer.encode_pieces(text):
            lengths[len(piece)] += 1
    return lengths


def side_by_side_examples(
    tokenizer_a: Tokenizer, tokenizer_b: Tokenizer, texts: list[str]
) -> list[dict]:
    """Tokenizes the same sentences under two tokenizers (e.g. Hindi vs Nepali) -- visible proof
    the two vocabularies are genuinely different, for the Phase-1 report's example table."""
    examples = []
    for text in texts:
        examples.append(
            {
                "text": text,
                "pieces_a": tokenizer_a.encode_pieces(text),
                "pieces_b": tokenizer_b.encode_pieces(text),
                "n_tokens_a": len(tokenizer_a.encode(text)),
                "n_tokens_b": len(tokenizer_b.encode(text)),
            }
        )
    return examples
