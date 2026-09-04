"""
Batch sampling over the Phase-1 encoded corpora (`train.bin` + `train.idx`).

The corpus is a flat `uint16` token stream with documents already EOS-separated by
`lma/tokenizer/encode_corpus.py`, so training windows are cut straight out of the stream and are
allowed to cross document boundaries -- the EOS token is what teaches the model where a document
ends, and re-packing each document into its own padded window would waste a large fraction of every
batch on padding for no benefit.

Two design decisions here are worth stating, because the obvious alternatives are both wrong:

**Windows are drawn from a shuffled permutation, not sampled with replacement.** Sampling uniformly
at random from 504k windows and drawing 504k of them covers only ~63% of the corpus -- a third of
the data is never seen and some windows are seen three times, in a run that is nominally one epoch.
A permutation gives exact coverage, is reproducible from a seed, and makes resume trivial: the
training position is a single integer.

**Ranks are interleaved, not split into contiguous halves.** The splits are written manual-first,
downloaded-second (see `lma/tokenizer/sample.py`, where the same ordering forced stride sampling for
tokenizer training). Handing rank 0 the first half of the stream would train it almost entirely on
manual crawl and rank 1 almost entirely on downloaded text. DDP averages the gradients so the model
would survive, but the per-rank loss curves would diverge for reasons that have nothing to do with
optimisation, and the single-GPU fallback path would silently train on a skewed subset. Interleaving
by window index makes every rank's slice a uniform sample of the whole corpus and still guarantees
no two ranks ever touch the same window.

The token array is memory-mapped throughout and never materialised: `train.bin` is 1.0-1.1 GB per
language, and the machine this was developed on has 7.4 GB of RAM -- the same constraint that shaped
every Phase-1 pipeline stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lma.tokenizer.encode_corpus import load_encoded_split


@dataclass
class SplitInfo:
    name: str
    n_tokens: int
    n_documents: int
    n_windows: int


class WindowSampler:
    """Deterministic, rank-sharded, shuffled pass over non-overlapping windows of the token stream.

    `position` counts optimizer micro-steps taken *by this rank*. It is the only state, which is
    what makes resume exact: a checkpoint stores the position, and the resumed run draws the very
    next window the killed run would have drawn.
    """

    def __init__(
        self, n_tokens: int, block_size: int, rank: int = 0, world_size: int = 1, seed: int = 42
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"bad rank/world_size: {rank}/{world_size}")
        # -1 because a window of `block_size` inputs needs one extra token to form its targets.
        self.n_windows = (n_tokens - 1) // block_size
        if self.n_windows < world_size:
            raise ValueError(
                f"{n_tokens:,} tokens gives only {self.n_windows} windows of {block_size}, "
                f"fewer than world_size={world_size}"
            )
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        # Trailing windows that do not divide evenly across ranks are dropped, so every rank runs the
        # same number of steps per epoch. DDP requires this: a rank that finishes early stops
        # participating in the gradient all-reduce and every other rank blocks forever.
        self.per_rank = self.n_windows // world_size
        self._epoch = -1
        self._perm: np.ndarray | None = None

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            # Seeded per epoch, not once: a second epoch must not repeat the first epoch's order.
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n_windows)
            self._epoch = epoch
        return self._perm

    def window_start(self, position: int) -> int:
        """Token offset of the window this rank draws at `position`."""
        epoch, offset = divmod(position, self.per_rank)
        perm = self._permutation(epoch)
        return int(perm[offset * self.world_size + self.rank]) * self.block_size


class TokenBatcher:
    """Turns a WindowSampler's offsets into (x, y) LongTensor batches over a memmapped split."""

    def __init__(
        self, bin_path: Path, idx_path: Path, block_size: int, batch_size: int,
        rank: int = 0, world_size: int = 1, seed: int = 42,
    ):
        self.tokens, self.offsets = load_encoded_split(Path(bin_path), Path(idx_path))
        self.block_size = block_size
        self.batch_size = batch_size
        self.sampler = WindowSampler(len(self.tokens), block_size, rank, world_size, seed)
        self.position = 0

    @property
    def info(self) -> SplitInfo:
        return SplitInfo(
            name=Path(self.tokens.filename).stem if self.tokens.filename else "?",
            n_tokens=len(self.tokens),
            n_documents=max(len(self.offsets) - 1, 0),
            n_windows=self.sampler.n_windows,
        )

    def batch(self, position: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One batch of `batch_size` windows. Advances `self.position` unless one is passed in."""
        if position is None:
            position = self.position
            self.position += self.batch_size
        xs = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        ys = np.empty((self.batch_size, self.block_size), dtype=np.int64)
        for i in range(self.batch_size):
            start = self.sampler.window_start(position + i)
            # astype copies exactly block_size+1 tokens out of the memmap. torch has no uint16
            # dtype, so the cast to int64 is required, not merely convenient.
            window = np.asarray(self.tokens[start:start + self.block_size + 1]).astype(np.int64)
            xs[i] = window[:-1]
            ys[i] = window[1:]
        return torch.from_numpy(xs), torch.from_numpy(ys)

    def tokens_per_epoch(self) -> int:
        return self.sampler.per_rank * self.sampler.world_size * self.block_size


def iter_eval_windows(
    bin_path: Path, idx_path: Path, block_size: int, batch_size: int, max_batches: int | None = None
):
    """Sequential, non-overlapping, deterministic sweep -- for validation and test.

    Deliberately NOT the training sampler: an evaluation number that changes between two runs of the
    same checkpoint is useless for comparing checkpoints, and a shuffled sampler with a different
    seed would do exactly that. Every token of the split is scored exactly once, in order.
    """
    tokens, _ = load_encoded_split(Path(bin_path), Path(idx_path))
    n_windows = (len(tokens) - 1) // block_size
    batch, n_yielded = [], 0
    for w in range(n_windows):
        start = w * block_size
        window = np.asarray(tokens[start:start + block_size + 1]).astype(np.int64)
        batch.append(window)
        if len(batch) == batch_size:
            stacked = np.stack(batch)
            yield torch.from_numpy(stacked[:, :-1]), torch.from_numpy(stacked[:, 1:])
            batch = []
            n_yielded += 1
            if max_batches is not None and n_yielded >= max_batches:
                return
    if batch:                                   # final short batch, so no tokens are skipped
        stacked = np.stack(batch)
        yield torch.from_numpy(stacked[:, :-1]), torch.from_numpy(stacked[:, 1:])


def to_device(x: torch.Tensor, y: torch.Tensor, device: torch.device):
    """Non-blocking H2D copy when the target is CUDA, so the transfer overlaps the previous step."""
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
