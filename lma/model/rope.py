"""
Rotary position embeddings (RoPE, Su et al. 2021), implemented from scratch.

Chosen over a learned positional table for two reasons, both measurable rather than stylistic:

* **Cost.** A learned table at 1024 positions x 384 dimensions is 393,216 parameters -- 1.6% of a
  ~24M budget spent on something rotary embeddings do with zero parameters. In a project whose
  parameter count is a hard constraint, that 1.6% buys most of another transformer layer.
* **Behaviour.** A learned table has no meaning past the longest position it saw in training; RoPE
  is a fixed rotation, so positions beyond the training context degrade smoothly instead of
  indexing into random initialisation.

The formulation here is the paper's Eq. (34) directly: the head dimension is read as head_dim/2
two-dimensional planes, and the vector in plane p at position m is rotated by angle m*theta_p with
theta_p = base^(-2p/head_dim). Consecutive dimensions (0,1), (2,3), ... form the pairs, which is the
interleaved convention of the original paper rather than the split-half `rotate_half` variant used by
some implementations. The two are related by a fixed permutation of the head dimension and are
equivalent in expressiveness -- but they are NOT interchangeable across a checkpoint, so this file is
the single definition and both the training path and the attention-analysis path call it.

The property that makes RoPE work, and the one tests/test_model.py asserts numerically, is that
after rotation the dot product q_m . k_n depends on m and n only through (m - n): position enters
attention as relative distance without any relative-position bias table.
"""
from __future__ import annotations

import torch


def rope_cache(
    seq_len: int, head_dim: int, base: float = 10_000.0,
    device: torch.device | None = None, dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputed cos/sin tables of shape [seq_len, head_dim // 2].

    Built once and sliced per forward pass rather than recomputed: the angles depend only on
    position and dimension, never on the data, so recomputing them every step would be pure waste.
    Always built in float32 even when the model runs in fp16 -- the angles span several orders of
    magnitude (theta_p ranges from 1.0 down to 1e-4 here) and fp16 has ~3 decimal digits, so
    computing them in half precision loses the high-frequency planes entirely.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)          # [seq_len, head_dim // 2]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates x in place-of-position. x is [B, H, T, head_dim]; cos/sin are [T, head_dim // 2].

    Done in float32 regardless of the incoming dtype, for the same reason the cache is: under fp16
    autocast q and k arrive as half, and rotating in half precision compounds rounding error into
    every attention score. The result is cast back so the caller's dtype is unchanged.
    """
    dtype = x.dtype
    x = x.float()
    x1, x2 = x[..., 0::2], x[..., 1::2]                # the two axes of each rotation plane
    cos = cos.float()[None, None, :, :]
    sin = sin.float()[None, None, :, :]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2).to(dtype)               # re-interleave back to head_dim
