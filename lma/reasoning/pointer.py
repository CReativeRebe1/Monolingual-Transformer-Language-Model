"""
Answer-pointer attention mass: which heads look at the right entity when the answer is produced.

The Phase-2 statistics (entropy, distance, previous-token mass) describe *how* a head spreads its
attention, averaged over every position. Finetuning on reasoning, if it changes anything in
attention, should change something narrower: what the model attends to at the single position where
it commits to an answer. That position is the last prompt token (after "उत्तर:"), whose output
distribution is the first answer token.

For each layer and head, from that query position:

    correct mass     = sum of attention on the tokens of the correct entity (all its mentions)
    distractor mass  = mean, over the other mentioned entities, of the same sum
    pointer score    = correct mass - distractor mass

A head with a high pointer score is reading the answer off the premises. Comparing the
[layer x head] pointer-score grid before and after SFT (the Δ heatmap) shows whether finetuning
created such heads or reused existing ones. Only examples whose answer is an entity are used -- a tie
("बराबर") has no correct entity to point at.

Token-to-entity alignment is done by decoding growing prefixes of the token sequence, which gives
exact character spans for every token without relying on SentencePiece offset conventions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lma.reasoning.build import mentions
from lma.reasoning.facts import EQUAL
from lma.reasoning.sft_data import encode_prompt


@dataclass
class PointerResult:
    """[L, H] grids averaged over the examples scored."""
    correct: np.ndarray
    distractor: np.ndarray
    n: int

    @property
    def score(self) -> np.ndarray:
        """Correct minus mean-distractor mass, per layer and head."""
        return self.correct - self.distractor

    def to_dict(self) -> dict:
        return {"n": self.n, "correct": self.correct.round(5).tolist(),
                "distractor": self.distractor.round(5).tolist(), "score": self.score.round(5).tolist()}


def token_char_spans(tokenizer, ids: list[int]) -> list[tuple[int, int]]:
    """(start, end) character span of each token within `tokenizer.decode(ids)`."""
    spans, prev = [], 0
    for i in range(len(ids)):
        end = len(tokenizer.decode(ids[:i + 1]))
        spans.append((min(prev, end), end))
        prev = end
    return spans


def entity_token_positions(tokenizer, ids: list[int], names: list[str]) -> dict[str, list[int]]:
    """Token indices overlapping each name's character occurrences in the decoded sequence."""
    text = tokenizer.decode(ids)
    spans = token_char_spans(tokenizer, ids)
    out: dict[str, list[int]] = {}
    for name, starts in mentions(text, names).items():
        pos = set()
        for s in starts:
            e = s + len(name)
            pos.update(i for i, (a, b) in enumerate(spans) if a < e and b > s)
        out[name] = sorted(pos)
    return out


@torch.no_grad()
def pointer_mass(model, tokenizer, records: list[dict], device) -> PointerResult:
    """Averages correct/distractor attention mass from the answer position over `records`."""
    model.eval()
    L, H = model.config.n_layers, model.config.n_heads
    correct = np.zeros((L, H))
    distract = np.zeros((L, H))
    n = 0
    for r in records:
        if r["answer_key"] == EQUAL:
            continue
        ids = encode_prompt(tokenizer, r["prompt"])
        pos = entity_token_positions(tokenizer, ids, r["entities"])
        gold = pos.get(r["answer"], [])
        others = [pos[e] for e in r["entities"] if e != r["answer"] and pos.get(e)]
        if not gold or not others:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, _, atts = model(x, output_attentions=True)
        for layer, a in enumerate(atts):
            row = a[0, :, -1, :].float().cpu().numpy()          # [H, T] from the answer position
            correct[layer] += row[:, gold].sum(axis=1)
            distract[layer] += np.mean([row[:, o].sum(axis=1) for o in others], axis=0)
        n += 1
    if n:
        correct /= n
        distract /= n
    return PointerResult(correct, distract, n)


@torch.no_grad()
def attention_matrix(model, ids: list[int], layer: int, head: int, device) -> np.ndarray:
    """[T, T] attention weights of one head for one sequence (for heatmaps)."""
    model.eval()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, atts = model(x, output_attentions=True)
    return atts[layer][0, head].float().cpu().numpy()
