"""
Causal multi-head self-attention, written from scratch -- and written TWICE, on purpose.

`forward(..., output_attentions=False)` routes to `F.scaled_dot_product_attention`, which fuses the
matmul-softmax-matmul into one kernel and never materialises the [B, H, T, T] score matrix. That is
what makes pretraining affordable: at B=16, H=6, T=1024 the score matrix alone is 1.5 GB in fp32 per
layer, which does not fit on a 16 GB T4 alongside the model, the optimizer state and the activations.

`forward(..., output_attentions=True)` runs the explicit definition instead --

    A = softmax(Q K^T / sqrt(d_k) + M),   out = A V

-- and returns A. The fused kernel gives no way to read A back out (that is precisely how it saves
the memory), so extracting attention weights for analysis requires the unfused path to exist. Both
paths compute the same function; only one of them can be observed.

**They must stay identical.** tests/test_model.py asserts agreement to atol=1e-5 on the same inputs.
Without that test, a divergence between the two -- a mask off by one row, a missing scale, RoPE
applied in a different order -- would produce attention figures describing a model that was never
trained, and nothing downstream would flag it: the heatmaps would still look like plausible
attention.

Note on hardware: on Kaggle's T4 (Turing, sm75) PyTorch has no FlashAttention backend -- flash
requires sm80+. SDPA still selects its memory-efficient backend there, which is the win that
matters (it is the O(T^2) memory that hurts, not the FLOPs).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lma.model.rope import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must divide by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        # One fused projection for Q, K and V. Mathematically identical to three separate
        # nn.Linear layers, but a single [d_model, 3*d_model] matmul instead of three smaller ones
        # -- fewer kernel launches per layer, which at 10 layers and thousands of steps is real time.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # No biases anywhere: with RMSNorm ahead of every projection the bias is redundant, and
        # dropping it removes 4*d_model parameters per layer for no measured loss (the Llama/PaLM
        # convention).

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_heads, T, head_dim]: heads become a batch dimension so every head
        # attends independently in one batched matmul.
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE on Q and K only -- V carries content, not position, and rotating it would make the
        # attention OUTPUT position-dependent rather than just the attention weights.
        # cos/sin are None in the NoPE ablation (GPTConfig.use_rope=False): q and k are then pure
        # content, and two identical tokens at different positions produce identical keys.
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if not output_attentions:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
            attn = None
        else:
            # Scaling by 1/sqrt(head_dim): q.k is a sum of head_dim products of unit-variance
            # terms, so its variance grows linearly with head_dim. Unscaled, the scores at
            # head_dim=64 have standard deviation 8, softmax saturates into a near one-hot, and its
            # gradient vanishes -- the model would stop learning where to attend almost immediately.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Strictly-upper-triangular positions are the future: query i may not see key j > i.
            future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
            attn = scores.softmax(dim=-1)              # [B, H, T, T] -- the object of study
            # Dropout on the attention probabilities, matching what SDPA's dropout_p does on the
            # fast path. Without this line the two paths compute different functions whenever
            # dropout > 0 and the module is in train mode -- the equivalence test would not catch
            # it, because it (correctly) runs in eval mode where both are dropout-free.
            attn_dropped = F.dropout(attn, p=self.dropout, training=self.training)
            y = attn_dropped @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), attn
