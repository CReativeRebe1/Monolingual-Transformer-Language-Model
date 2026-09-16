"""
Position diagnostics for the bonus NoPE ablation: where in the context, and how much, a model uses
token order.

Perplexity says *that* a model without positional embeddings is worse; these two measurements say
*where* and *why*, which is what the assignment's "short explanation of what breaks without position
information" needs evidence for.

**Loss by position.** Mean next-token loss at each position 0..T-1 of a 1024-token window. Every
position of a decoder sees a different amount of context, so this curve normally falls steeply over
the first few dozen tokens and then flattens. A model with no positional signal can still count --
the causal mask makes position i attend over exactly i+1 keys, so a uniform head's output shrinks as
1/(i+1), and deeper layers can read that as a crude position estimate (Haviv et al. 2022,
"Transformer Language Models without Positional Encodings Still Learn Positional Information";
Kazemnejad et al. 2023, NoPE). That estimate gets coarser the further into the window a token is, so
the expected failure is a gap that *grows* with position, rather than a uniform penalty.

**Order sensitivity.** For each window, score the second half (the continuation) twice: once after
the intact first half, and once after the same first-half tokens in a random order. The increase in
continuation loss measures how much the model relies on the order of its context, as opposed to the
bag of tokens in it. A model that uses order well is hurt badly by the shuffle; a model that cannot
represent order is hurt less, because it was not using that information in the first place. The
continuation itself is left intact, so local next-token prediction within it is unaffected in both
conditions -- only the *context's* order changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lma.train.data import iter_eval_windows


@torch.no_grad()
def loss_by_position(model, bin_path: Path, idx_path: Path, block_size: int, batch_size: int,
                     device, max_batches: int | None = None) -> np.ndarray:
    """[block_size] mean next-token NLL (nats) at each window position, over a deterministic sweep."""
    model.eval()
    total = torch.zeros(block_size, dtype=torch.float64, device=device)
    count = 0
    for x, y in iter_eval_windows(bin_path, idx_path, block_size, batch_size, max_batches):
        if x.size(1) != block_size:
            continue
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _, _ = model(x)
        nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")   # [B, T]
        total += nll.sum(dim=0).double()
        count += nll.size(0)
    return (total / max(count, 1)).cpu().numpy()


@torch.no_grad()
def order_sensitivity(model, bin_path: Path, idx_path: Path, block_size: int, batch_size: int,
                      device, max_batches: int | None = None, seed: int = 0) -> dict:
    """Continuation loss after an intact vs a shuffled first half, averaged over windows."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    half = block_size // 2
    intact_sum = shuffled_sum = 0.0
    n = 0
    for x, y in iter_eval_windows(bin_path, idx_path, block_size, batch_size, max_batches):
        if x.size(1) != block_size:
            continue
        xs = x.clone()
        for i in range(xs.size(0)):
            perm = torch.randperm(half, generator=gen)
            xs[i, :half] = x[i, perm]
        results = []
        for inp in (x, xs):
            inp, tgt = inp.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _, _ = model(inp)
            nll = F.cross_entropy(logits[:, half:].float().transpose(1, 2), tgt[:, half:], reduction="none")
            results.append(float(nll.mean(dim=1).sum()))
        intact_sum += results[0]
        shuffled_sum += results[1]
        n += x.size(0)
    intact, shuffled = intact_sum / max(n, 1), shuffled_sum / max(n, 1)
    return {"windows": n, "block_size": block_size, "continuation_tokens": block_size - half,
            "intact_nll": intact, "shuffled_context_nll": shuffled, "delta_nll": shuffled - intact,
            "intact_ppl": float(np.exp(intact)), "shuffled_context_ppl": float(np.exp(shuffled))}


def bucket_positions(losses: np.ndarray, edges: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
                     ) -> list[dict]:
    """Mean loss over position ranges [edge_k, edge_{k+1}) -- log-spaced, since the curve's shape is."""
    out, lo = [], 0
    for hi in edges:
        hi = min(hi, len(losses))
        if hi > lo:
            out.append({"from": lo, "to": hi, "mean_nll": float(losses[lo:hi].mean())})
        lo = hi
    return out
