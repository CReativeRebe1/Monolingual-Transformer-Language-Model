"""
Supervised-finetuning examples: tokenise, mask the prompt, pad, batch.

One example is one sequence

    [EOS] प्रश्न: <premises> <question> उत्तर: ▁<answer> [EOS]
    |------------------ prompt ------------------| |- target -|

and the loss is taken **only on the target** -- the answer tokens and the closing EOS (L9 slide 8).
Every other label is -100, which `GPT.forward` already passes to `F.cross_entropy(ignore_index=-100)`.
Training on the prompt tokens as well would spend most of each gradient on learning to reproduce
templated premises, which is exactly the part that must *not* be memorised.

Three details that are decisions, not accidents:

* **The leading EOS.** Pretraining concatenated documents with EOS between them, so to the model
  "start of a document" is "right after an EOS". Prefixing one puts every prompt in the position the
  model already treats as a fresh start. The tokenizer's BOS id was never used in pretraining.
* **The closing EOS is a target.** It is how the model learns to stop after the name (L9 slide 9);
  without it greedy decoding runs on, and strict exact match fails for a reason that has nothing to
  do with reasoning.
* **Prompt and answer are encoded separately** and concatenated. SentencePiece's dummy prefix makes
  `encode(answer)` start with "▁", which is the space after "उत्तर:". Encoding the pair jointly
  could merge across the boundary differently for different answers; encoding separately makes the
  prompt tokens identical for every candidate, which candidate scoring relies on.

Padding is on the right with pad id 0 and label -100. Under a causal mask, padding after the last
real token cannot influence any real position, so no attention mask is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

IGNORE = -100


@dataclass
class Encoded:
    """One tokenised example. `input_ids` and `labels` are aligned (labels[i] is the target at i)."""
    input_ids: list[int]
    labels: list[int]
    prompt_len: int                    # tokens of [EOS] + prompt, i.e. where generation starts


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """[EOS] + prompt tokens: the context a model generates its answer from."""
    return [tokenizer._sp.eos_id()] + tokenizer.encode(prompt)


def encode_answer(tokenizer, answer: str, add_eos: bool = True) -> list[int]:
    """Answer tokens (leading "▁" from SentencePiece's dummy prefix), plus EOS by default."""
    ids = tokenizer.encode(answer)
    return ids + [tokenizer._sp.eos_id()] if add_eos else ids


def encode_example(tokenizer, prompt: str, answer: str) -> Encoded:
    """Input/label pair with the loss masked to the answer and its EOS."""
    p = encode_prompt(tokenizer, prompt)
    a = encode_answer(tokenizer, answer)
    seq = p + a
    input_ids = seq[:-1]
    # Position i predicts seq[i+1]. The first target that belongs to the answer is seq[len(p)],
    # predicted from position len(p)-1 -- the last prompt token.
    labels = [IGNORE] * (len(p) - 1) + a
    return Encoded(input_ids, labels, len(p))


def collate(batch: list[Encoded], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pads a list of examples into (x, y) LongTensors."""
    T = max(len(e.input_ids) for e in batch)
    x = torch.full((len(batch), T), pad_id, dtype=torch.long)
    y = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e.input_ids)
        x[i, :n] = torch.tensor(e.input_ids)
        y[i, :n] = torch.tensor(e.labels)
    return x, y


class SFTDataset:
    """Pre-tokenised records, with the same resumable position semantics as `WindowSampler`:
    an epoch is a seeded permutation, and the whole sampler state is one integer."""

    def __init__(self, records: list[dict], tokenizer, seed: int = 0):
        self.records = records
        self.examples = [encode_example(tokenizer, r["prompt"], r["answer"]) for r in records]
        self.seed = seed
        self._epoch = -1
        self._perm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.examples)

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(len(self.examples))
            self._epoch = epoch
        return self._perm

    def batch(self, position: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The batch starting at global `position` (examples consumed so far). Wraps into the next
        epoch's permutation, so a batch spanning an epoch boundary is still `batch_size` long."""
        items = []
        for k in range(position, position + batch_size):
            epoch, offset = divmod(k, len(self.examples))
            items.append(self.examples[int(self._permutation(epoch)[offset])])
        return collate(items)

    def eval_batches(self, batch_size: int):
        """In-order batches over every example, for validation loss."""
        for i in range(0, len(self.examples), batch_size):
            yield collate(self.examples[i:i + batch_size])
