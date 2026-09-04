"""
Attention diagnostics: entropy, mean attention distance, head taxonomy, and heatmaps.

Built in Phase 2 rather than Phase 3, where the assignment schedules the analysis, because the
machinery belongs with the model and the eval harness -- Phase 3 then becomes reporting rather than
engineering.

Everything here consumes the `[B, H, T, T]` tensors returned by `GPT.forward(output_attentions=True)`
(the explicit `softmax(QK^T/sqrt(d_k) + M)V` path, which a test proves numerically identical to the
fused path the model was trained with -- otherwise these figures would describe a different model).

**The normalisation that makes entropy readable.** Under causal masking, query position *i* can only
attend over *i+1* keys, so the maximum achievable entropy at position *i* is log2(i+1), not log2(T).
Raw mean entropy therefore rises with position for purely arithmetic reasons, and a head that
behaves identically everywhere looks like it "spreads out over the sequence". Both forms are
reported: raw, in bits, for comparison against the literature, and H / log2(i+1) in [0, 1], which
is the one that supports a claim about behaviour.

**Memory.** A `[1, 6, 1024, 1024]` fp32 tensor is 25 MB, so all ten layers at full context is 250 MB
-- and that is per batch element. Statistics are therefore accumulated layer by layer and the
extraction context is short (128-256 tokens) with batch 1, which is ample: head roles are visible in
a couple of hundred tokens.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

EPS = 1e-12


# ------------------------------------------------------------------ per-head statistics

def head_entropy(attn: torch.Tensor, normalise: bool = False) -> torch.Tensor:
    """Mean attention entropy in bits, per head. `attn` is [B, H, T, T]; returns [H].

        H(A_h) = -sum_j A[h,i,j] * log2(A[h,i,j])     averaged over query positions i

    With `normalise=True`, each query's entropy is divided by log2(i+1) -- the maximum achievable
    at that position under causal masking -- giving a number in [0, 1] that is comparable across
    positions. Query 0 attends to exactly one key, so its entropy is 0 and its maximum is 0; it is
    excluded from the normalised mean rather than contributing a 0/0.
    """
    a = attn.float().clamp_min(0.0)
    per_query = -(a * torch.log2(a + EPS)).sum(dim=-1)          # [B, H, T]
    if not normalise:
        return per_query.mean(dim=(0, 2))
    T = a.size(-1)
    max_entropy = torch.log2(torch.arange(1, T + 1, device=a.device, dtype=torch.float32))
    return (per_query[:, :, 1:] / max_entropy[1:]).mean(dim=(0, 2))


def head_distance(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention distance, per head. `attn` is [B, H, T, T]; returns [H].

        d(A_h) = sum_j (i - j) * A[h,i,j]             averaged over query positions i

    Non-negative by construction: causal masking puts zero mass on j > i. A head attending only to
    the immediately preceding token scores exactly 1; a head attending uniformly over its history
    scores about i/2, so the mean over positions grows with context length -- which is why the
    taxonomy below compares heads within a layer rather than against an absolute threshold.

    Query 0 has nowhere to look but itself and contributes a forced 0, so it is excluded.
    """
    a = attn.float()
    T = a.size(-1)
    positions = torch.arange(T, device=a.device, dtype=torch.float32)
    offsets = positions[:, None] - positions[None, :]            # [T, T], (i - j)
    per_query = (a * offsets).sum(dim=-1)                        # [B, H, T]
    return per_query[:, :, 1:].mean(dim=(0, 2))


def previous_token_mass(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention mass on the immediately preceding token (j = i-1), per head. Returns [H].

    Isolated because the previous-token head is the one well-documented, unambiguous head role in
    small decoders, and it is worth naming directly rather than inferring from a distance near 1 --
    a head splitting its mass evenly between j=i-2 and j=i also averages to distance 1.
    """
    a = attn.float()
    T = a.size(-1)
    idx = torch.arange(1, T, device=a.device)
    return a[:, :, idx, idx - 1].mean(dim=(0, 2))


def first_token_mass(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention mass on position 0, per head. Returns [H].

    Position 0 is the well-known "attention sink": with softmax forced to sum to 1, a head with
    nothing informative to attend to has to put its mass somewhere, and the first token is the one
    every query can see. A high value here usually means a head is idle, not that position 0 is
    important -- a distinction that matters when reading the heatmaps.
    """
    return attn.float()[:, :, :, 0].mean(dim=(0, 2))


# ------------------------------------------------------------------ taxonomy

@dataclass
class HeadStats:
    layer: int
    head: int
    entropy_bits: float
    entropy_normalised: float
    mean_distance: float
    prev_token_mass: float
    first_token_mass: float
    role: str

    def to_dict(self) -> dict:
        return asdict(self)


def classify(entropy_norm: float, distance: float, prev_mass: float, first_mass: float) -> str:
    """A coarse, explicitly-thresholded head role.

    The thresholds are conventions, not discoveries, and the report says so -- their job is to make
    a 60-head table readable, not to establish that these are the categories attention "really" has.
    Order matters: the specific, mechanically-checkable roles are tested before the diffuse ones.
    """
    if prev_mass > 0.5:
        return "previous_token"
    if first_mass > 0.5:
        return "attention_sink"
    if entropy_norm < 0.3 and distance < 5.0:
        return "local"
    if entropy_norm > 0.7:
        return "diffuse_global"
    if distance > 20.0:
        return "long_range"
    return "mixed"


@torch.no_grad()
def analyse(model, input_ids: torch.Tensor) -> list[HeadStats]:
    """Runs one forward pass with attention extraction and returns per-layer, per-head statistics.

    `input_ids` is [B, T] and should be short: see the module docstring on why 128-256 tokens with
    batch 1 is the intended scale. The attention list is consumed layer by layer and each layer's
    tensor is released before the next is read, so peak memory is one layer rather than all of them.
    """
    was_training = model.training
    model.eval()
    _, _, attentions = model(input_ids, output_attentions=True)

    stats: list[HeadStats] = []
    for layer, attn in enumerate(attentions):
        ent = head_entropy(attn).cpu()
        ent_n = head_entropy(attn, normalise=True).cpu()
        dist = head_distance(attn).cpu()
        prev = previous_token_mass(attn).cpu()
        first = first_token_mass(attn).cpu()
        for h in range(attn.size(1)):
            stats.append(HeadStats(
                layer=layer, head=h,
                entropy_bits=round(float(ent[h]), 4),
                entropy_normalised=round(float(ent_n[h]), 4),
                mean_distance=round(float(dist[h]), 4),
                prev_token_mass=round(float(prev[h]), 4),
                first_token_mass=round(float(first[h]), 4),
                role=classify(float(ent_n[h]), float(dist[h]), float(prev[h]), float(first[h])),
            ))
        del attn
        attentions[layer] = None
    if was_training:
        model.train()
    return stats


def aggregate(runs: list[list[HeadStats]]) -> list[HeadStats]:
    """Averages per-head statistics across several input sequences.

    One sequence is one sample of a head's behaviour and a head's role should not be inferred from
    a single paragraph -- a "local" verdict drawn from one input that happens to be a list of names
    would be an artefact of that input, not a property of the head.
    """
    if not runs:
        return []
    n = len(runs)
    out: list[HeadStats] = []
    for i in range(len(runs[0])):
        entries = [r[i] for r in runs]
        ent_n = sum(e.entropy_normalised for e in entries) / n
        dist = sum(e.mean_distance for e in entries) / n
        prev = sum(e.prev_token_mass for e in entries) / n
        first = sum(e.first_token_mass for e in entries) / n
        out.append(HeadStats(
            layer=entries[0].layer, head=entries[0].head,
            entropy_bits=round(sum(e.entropy_bits for e in entries) / n, 4),
            entropy_normalised=round(ent_n, 4),
            mean_distance=round(dist, 4),
            prev_token_mass=round(prev, 4),
            first_token_mass=round(first, 4),
            role=classify(ent_n, dist, prev, first),
        ))
    return out


def role_counts(stats: list[HeadStats]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in stats:
        counts[s.role] = counts.get(s.role, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def save_stats(stats: list[HeadStats], path: Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "heads": [s.to_dict() for s in stats],
        "role_counts": role_counts(stats),
        "n_layers": max((s.layer for s in stats), default=-1) + 1,
        "n_heads": max((s.head for s in stats), default=-1) + 1,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def layer_means(stats: list[HeadStats]) -> dict[int, dict[str, float]]:
    """Per-layer means -- the table the report prints, and the depth trend it discusses."""
    by_layer: dict[int, list[HeadStats]] = {}
    for s in stats:
        by_layer.setdefault(s.layer, []).append(s)
    return {
        layer: {
            "entropy_bits": round(float(np.mean([s.entropy_bits for s in group])), 4),
            "entropy_normalised": round(float(np.mean([s.entropy_normalised for s in group])), 4),
            "mean_distance": round(float(np.mean([s.mean_distance for s in group])), 4),
            "prev_token_mass": round(float(np.mean([s.prev_token_mass for s in group])), 4),
        }
        for layer, group in sorted(by_layer.items())
    }
