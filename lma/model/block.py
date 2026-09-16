"""
The three pieces a transformer block is made of: RMSNorm, a SwiGLU feed-forward, and the pre-LN
residual wiring that joins them to attention.

All three are written out rather than assembled from `nn.TransformerEncoderLayer`, which the
assignment requires -- and which is just as well, since that layer is post-LN, uses ReLU/GELU rather
than a gated activation, and has no hook for rotary embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lma.model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """Root-mean-square normalisation (Zhang & Sennrich 2019): rescale by RMS, no mean subtraction.

    Cheaper than LayerNorm (one reduction instead of two, and no bias parameter) and empirically
    just as stable for decoder LMs, which is why every recent open decoder uses it.

    The `.float()` is not decoration. Under fp16 autocast, x arrives as half, and x.pow(2) on
    activations of magnitude ~300 overflows fp16's 65504 ceiling -- the sum of squares across 384
    dimensions goes to inf, rsqrt(inf) is 0, and the whole layer silently outputs zeros. Normalising
    in fp32 and casting back is the standard fix and costs nothing measurable.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x32.to(dtype) * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x)).

    Three matrices instead of a plain MLP's two, so the hidden width is set to (8/3)*d_model rather
    than 4*d_model to keep the SAME parameter count as the 4x non-gated block it replaces:
    3 * d_model * (8/3 * d_model) = 8 * d_model^2 = 2 * d_model * (4 * d_model). At d_model=384 that
    is exactly 1024, which is also a clean multiple of 64 for tensor-core alignment -- a rare case
    where the principled number and the hardware-friendly number coincide.

    Writing 4*d_model here instead would push each layer from 1.77M to 2.36M parameters and the whole
    model from 23.8M to 29.7M, breaking the ~25M budget. The config asserts the total for exactly
    this reason.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(norm(x)), then x + mlp(norm(x)).

    Pre-LN, not post-LN: with the norm inside the residual branch the identity path from input to
    output is unnormalised, so gradients reach layer 0 without passing through ten normalisations.
    Post-LN needs a learning-rate warmup long enough to be its own tuning problem at this depth.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden)
        # Residual (sublayer-output) dropout, the second of the two places a decoder applies it --
        # the first being the attention probabilities, inside CausalSelfAttention. Both are present
        # and wired to the same rate; both are inert at this project's dropout=0.0, which is a
        # deliberate choice (22-24 tokens per parameter, one epoch, nothing seen twice, so there is
        # no memorisation to regularise against) rather than a missing feature. nn.Dropout holds no
        # parameters, so these modules do not change the state_dict.
        self.drop_attn = nn.Dropout(dropout)
        self.drop_mlp = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.attn(self.norm_attn(x), cos, sin, output_attentions)
        x = x + self.drop_attn(attn_out)
        x = x + self.drop_mlp(self.mlp(self.norm_mlp(x)))
        return x, attn_weights
