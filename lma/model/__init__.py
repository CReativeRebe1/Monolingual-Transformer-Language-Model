"""Shared from-scratch decoder-only Transformer architecture (code only -- no language content).

Both Model H and Model L instantiate `GPT` from their own `<lang>/model/config.py`; the weights,
vocabularies and checkpoints stay entirely separate.
"""
from lma.model.attention import CausalSelfAttention
from lma.model.block import Block, RMSNorm, SwiGLU
from lma.model.rope import apply_rope, rope_cache
from lma.model.transformer import GPT, GPTConfig

__all__ = [
    "CausalSelfAttention", "Block", "RMSNorm", "SwiGLU",
    "apply_rope", "rope_cache", "GPT", "GPTConfig",
]
