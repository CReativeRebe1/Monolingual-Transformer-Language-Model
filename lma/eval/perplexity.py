"""
Perplexity and bits-per-byte on a held-out split.

**Why both, and why bits-per-byte is the one that matters here.** Perplexity is per *token*, and
Model H and Model L have different tokenizers -- trained separately on different corpora, as the
assignment requires. A token is a different amount of language in each, so comparing Hindi's
perplexity to Nepali's compares two different units and means nothing. It is still reported, because
it is the standard number and it is comparable *across checkpoints of the same model*, which is what
the training curve needs.

Bits-per-byte divides the same total surprisal by the UTF-8 byte length of the underlying text
instead of by the token count. Bytes are the same unit in both languages, so BPB is the number that
can be put side by side. It is also the number that cannot be gamed by a tokenizer: a vocabulary
that merges more aggressively lowers perplexity by making each token carry more text, and leaves BPB
untouched.

    BPB = (total NLL in nats / ln 2) / total UTF-8 bytes

The byte count is measured by decoding the evaluated token ids back to text, not estimated from a
fertility constant. Decoding is exact and cheap; a constant would silently absorb any tokenizer
difference into the metric that exists to be immune to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from lma.train.data import iter_eval_windows, to_device


@dataclass
class PerplexityResult:
    split: str
    n_tokens: int
    n_bytes: int
    mean_nll_nats: float
    perplexity: float
    bits_per_byte: float
    bits_per_token: float

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def evaluate_split(
    model, tokenizer, bin_path: Path, idx_path: Path, block_size: int, batch_size: int = 8,
    device: torch.device | None = None, max_batches: int | None = None, split_name: str = "val",
) -> PerplexityResult:
    """Sweeps a split in order and returns token- and byte-level surprisal.

    The sweep is deterministic and complete (see `iter_eval_windows`): every token of the split is
    scored exactly once, in order. Sampling instead would make the reported test number depend on a
    seed, which is not something a final result should do.
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0

    for x, y in iter_eval_windows(bin_path, idx_path, block_size, batch_size, max_batches):
        x, y = to_device(x, y, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits, _, _ = model(x)
        # Summed, not averaged: batches can be different sizes (the final one is short), so
        # averaging per batch and then averaging the averages would weight the tail wrongly.
        # float() before the sum because fp16 accumulation over millions of tokens loses precision
        # long before the sweep finishes.
        nll = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_nll += float(nll)
        total_tokens += int(y.numel())
        for row in y.tolist():
            total_bytes += len(tokenizer.decode(row).encode("utf-8"))

    if was_training:
        model.train()

    mean_nll = total_nll / max(total_tokens, 1)
    return PerplexityResult(
        split=split_name,
        n_tokens=total_tokens,
        n_bytes=total_bytes,
        mean_nll_nats=mean_nll,
        perplexity=math.exp(mean_nll),
        bits_per_byte=(total_nll / math.log(2)) / max(total_bytes, 1),
        bits_per_token=mean_nll / math.log(2),
    )


def perplexity_from_nll(mean_nll_nats: float) -> float:
    """PPL = exp(mean NLL). Separated out so the identity is testable on its own."""
    return math.exp(mean_nll_nats)


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    """Total surprisal in nats -> bits, divided by the bytes it describes."""
    return (total_nll_nats / math.log(2)) / max(n_bytes, 1)
