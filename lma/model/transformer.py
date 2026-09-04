"""
The decoder-only Transformer language model: `GPTConfig` + `GPT`.

Shared between Model H (Hindi) and Model L (Nepali) as *code only*. The two models are separate
objects with separately initialised, separately trained weights and separate vocabularies -- sharing
the class no more couples them than sharing `torch.nn.Linear` would. `scripts/verify_independence.py`
proves this mechanically on the finished checkpoints.

Built from `nn.Linear`, `nn.Embedding` and explicit tensor maths only. No `nn.Transformer`, no
`nn.MultiheadAttention`, no HuggingFace `modeling_*`, and no pretrained weights of any kind.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lma.model.block import Block, RMSNorm
from lma.model.rope import rope_cache


def analytic_param_count(config: "GPTConfig") -> dict[str, int]:
    """Parameter counts derived from the config arithmetically, without building the model.

    Exists so `<lang>/model/config.py` can assert its budget at import time for free instead of
    allocating 24M floats to count them. It is also the parameter table the report prints, which is
    the better reason: a breakdown that says WHERE the budget goes is worth more than one number.

    tests/test_model.py asserts these figures equal `GPT(config).param_count()` exactly -- so the
    arithmetic is checked against the real module rather than trusted.
    """
    d, v, L, h = config.d_model, config.vocab_size, config.n_layers, config.mlp_hidden
    embedding = v * d                        # tied: counted once, serves as lm_head too
    attn_per_layer = 4 * d * d               # fused qkv (3 d x d) + output projection
    mlp_per_layer = 3 * d * h                # SwiGLU: gate, up, down
    norms_per_layer = 2 * d                  # RMSNorm gains, one before attention, one before MLP
    per_layer = attn_per_layer + mlp_per_layer + norms_per_layer
    total = embedding + L * per_layer + d    # + final norm
    return {
        "embedding": embedding,
        "attention_per_layer": attn_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "norms_per_layer": norms_per_layer,
        "per_layer": per_layer,
        "all_layers": L * per_layer,
        "final_norm": d,
        "total": total,
        "non_embedding": total - embedding,
    }


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6
    mlp_hidden: int = 1024          # (8/3)*d_model -- see SwiGLU's docstring for why not 4x
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_base: float = 10_000.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            Block(config.d_model, config.n_heads, config.mlp_hidden, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm_final = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Weight tying (Press & Wolf 2017): the output projection IS the embedding matrix,
            # transposed. Saves 6,144,000 parameters here -- 26% of the whole budget -- which at a
            # fixed ~25M cap buys roughly three more transformer layers. It also regularises: an
            # input embedding and an output embedding for the same token are forced to agree.
            self.lm_head.weight = self.tok_emb.weight

        # Non-persistent: rebuilt from the config on load rather than stored, so a checkpoint does
        # not carry 1024x32 floats of derived data and a longer context can be loaded into a model
        # trained at a shorter one.
        cos, sin = rope_cache(config.max_seq_len, config.head_dim, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2's residual-projection scaling. Every layer adds its output into the residual stream,
        # so with n_layers independent contributions the stream's variance grows linearly with depth
        # and the final activations blow up before training starts. Scaling the two projections that
        # WRITE to the residual stream by 1/sqrt(2*n_layers) (two per block: attention out and MLP
        # down) keeps the variance flat regardless of depth.
        residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=config.init_std * residual_scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # ------------------------------------------------------------------ forward

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Returns (logits, loss, attentions).

        `attentions` is None unless output_attentions=True, in which case it is a list of
        [B, n_heads, T, T] tensors, one per layer. Those are large -- 25 MB per layer at T=1024,
        H=6, B=1 -- so callers extracting them should use short sequences and small batches; see
        lma/eval/attention.py, which consumes one layer at a time for exactly this reason.
        """
        B, T = idx.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]

        attentions: list[torch.Tensor] | None = [] if output_attentions else None
        for block in self.blocks:
            x, attn = block(x, cos, sin, output_attentions)
            if attentions is not None:
                attentions.append(attn)

        x = self.norm_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss, attentions

    # ------------------------------------------------------------------ introspection

    def param_count(self) -> tuple[int, int]:
        """(total, non-embedding). Reported as both because they answer different questions.

        The assignment's ~25M budget is on the total. But scaling behaviour -- how much *compute*
        the model does per token -- depends on the non-embedding count, since the embedding lookup
        is a gather, not a matmul. With tied weights the lm_head shares storage with tok_emb, so it
        is counted once, which `set()` on the storage pointers below guarantees.
        """
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += p.numel()
        embedding = self.tok_emb.weight.numel()
        return total, total - embedding

    def flops_per_token(self) -> float:
        """Forward+backward FLOPs per token, as 6N + attention (Kaplan et al. 2020, App. B).

        Used to sanity-check measured throughput against the hardware's ceiling: if the smoke run
        reports a tokens/s implying more FLOPS than a T4 can physically do, the measurement is
        wrong, not the GPU.
        """
        _, non_embedding = self.param_count()
        cfg = self.config
        attn = 12 * cfg.n_layers * cfg.d_model * cfg.max_seq_len
        return 6 * non_embedding + attn

    # ------------------------------------------------------------------ optimizer groups

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Weight decay on matrices only -- not on RMSNorm gains or biases.

        Decaying a normalisation gain pulls it toward 0, which scales that layer's whole output
        toward 0; there is no regularisation story that wants this, and it is the standard split
        every GPT-style trainer uses. The rule is dimensionality: anything with >=2 dims is a matrix
        that maps between spaces and can overfit; anything 1-D is a per-feature scale that cannot.
        """
        decay, no_decay = [], []
        seen: set[int] = set()
        for p in self.parameters():
            if not p.requires_grad or p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None, top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. idx is [B, T] of context; returns [B, T + max_new_tokens].

        temperature=0 means greedy (argmax). top_k and top_p may be combined; top_k is applied first.

        Deliberately no KV cache. Without one this is O(T^2) recompute, but the evaluation protocol
        generates 128 tokens from a 128-token prefix over a few hundred prompts -- seconds of GPU
        time either way -- and a KV cache is a stateful path that would need its own correctness
        tests to be trusted. Recomputing is the version that is obviously correct.
        """
        was_training = self.training
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # Crop to the context window: past max_seq_len the RoPE cache has no entries.
            window = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _, _ = self(window)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    # Keep the smallest set whose mass reaches top_p. The shift keeps the first
                    # token that crosses the threshold -- without it, a single token with
                    # probability > top_p would leave the candidate set empty.
                    drop = cumulative - sorted_logits.softmax(dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                next_id = torch.multinomial(logits.softmax(dim=-1), num_samples=1)

            if eos_id is not None:
                # Once a sequence has emitted EOS, keep emitting EOS so the tensor stays rectangular
                # and the caller can truncate at the first one.
                next_id = torch.where(finished.unsqueeze(1), torch.full_like(next_id, eos_id), next_id)
                finished |= next_id.squeeze(1) == eos_id
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        if was_training:
            self.train()
        return idx
