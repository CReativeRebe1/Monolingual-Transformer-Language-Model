"""
The continuation protocol: how held-out documents become (prompt, generated, reference) triples.

Stated explicitly because BLEU/chrF/ROUGE-L numbers are meaningless without it -- the same model
scores very differently at a 32-token prompt and a 256-token one, and reporting a score without the
protocol invites comparison against numbers produced under a different one.

    take a held-out document with at least prefix_tokens + continuation_tokens tokens
    feed the first `prefix_tokens` as the prompt
    generate `continuation_tokens`
    score against the document's own next `continuation_tokens` as the reference

Documents come from `train.idx`'s document boundaries -- the offsets `encode_corpus.py` wrote for
exactly this purpose -- so a prompt is a real document opening rather than an arbitrary window that
begins mid-sentence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lma.tokenizer.encode_corpus import document_slice, load_encoded_split


@dataclass
class Continuation:
    doc_index: int
    prompt: str
    generated: str
    reference: str


def sample_documents(
    bin_path: Path, idx_path: Path, n_docs: int, min_tokens: int, seed: int = 42
) -> list[tuple[int, np.ndarray]]:
    """Long-enough held-out documents, chosen deterministically.

    Deterministic because the report's generation table must be reproducible: a fresh random draw
    each run would change the sampled documents and therefore the BLEU score, and "the score moved"
    would be indistinguishable from "the model changed".
    """
    tokens, offsets = load_encoded_split(Path(bin_path), Path(idx_path))
    n_documents = len(offsets) - 1
    lengths = np.diff(offsets.astype(np.int64))
    eligible = np.nonzero(lengths >= min_tokens)[0]
    if len(eligible) == 0:
        raise ValueError(f"no held-out document has {min_tokens} tokens")
    rng = np.random.default_rng(seed)
    picked = rng.choice(eligible, size=min(n_docs, len(eligible)), replace=False)
    return [(int(i), np.asarray(document_slice(tokens, offsets, int(i)))) for i in sorted(picked)]


@torch.no_grad()
def continuations(
    model, tokenizer, bin_path: Path, idx_path: Path, n_docs: int = 200,
    prefix_tokens: int = 128, continuation_tokens: int = 128, batch_size: int = 16,
    temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
    seed: int = 42, device: torch.device | None = None,
) -> list[Continuation]:
    """Generates one continuation per sampled document under one decoding configuration."""
    device = device or next(model.parameters()).device
    docs = sample_documents(bin_path, idx_path, n_docs, prefix_tokens + continuation_tokens, seed)
    torch.manual_seed(seed)

    out: list[Continuation] = []
    for start in range(0, len(docs), batch_size):
        chunk = docs[start:start + batch_size]
        prompts = np.stack([d[:prefix_tokens] for _, d in chunk]).astype(np.int64)
        idx = torch.from_numpy(prompts).to(device)
        generated = model.generate(
            idx, max_new_tokens=continuation_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        for (doc_index, doc), row in zip(chunk, generated.tolist()):
            out.append(Continuation(
                doc_index=doc_index,
                prompt=tokenizer.decode(row[:prefix_tokens]),
                generated=tokenizer.decode(row[prefix_tokens:]),
                reference=tokenizer.decode(
                    doc[prefix_tokens:prefix_tokens + continuation_tokens].tolist()
                ),
            ))
    return out


# The four settings the assignment names -- greedy decoding plus temperatures 0.5, 1.0 and 1.5 --
# as PURE temperature sweeps with no top-k or top-p. That purity is the point: mixing nucleus
# truncation into a "temperature 1.0" row would mean the reported number describes the truncation as
# much as the temperature, and the three rows would no longer be comparable to each other.
REQUIRED_GRID = [
    {"name": "greedy", "temperature": 0.0},
    {"name": "t0.5", "temperature": 0.5},
    {"name": "t1.0", "temperature": 1.0},
    {"name": "t1.5", "temperature": 1.5},
]

# Supplementary nucleus settings, kept because the diversity analysis is more informative with them:
# pure temperature trades fluency against diversity along a single axis, while top-k/top-p cut the
# distribution's tail instead, and the two produce visibly different degeneration behaviour.
# Reported separately from the four above so the assignment's grid stays unmixed.
EXTRA_GRID = [
    {"name": "t0.7_top50", "temperature": 0.7, "top_k": 50},
    {"name": "t0.8_top_p0.9", "temperature": 0.8, "top_p": 0.9},
    {"name": "t1.0_top_p0.95", "temperature": 1.0, "top_p": 0.95},
]

DECODING_GRID = REQUIRED_GRID + EXTRA_GRID
