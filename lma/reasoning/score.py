"""
Scoring a model on the reasoning splits: exact match, candidate choice, error taxonomy.

Two metrics, because they answer different questions (L10: SFT "changes reliably: format, task
behaviour; unreliably: reasoning the base model cannot do"):

* **Generative exact match** -- greedy-decode from the prompt and compare the completion, cut at EOS,
  with the gold answer. This is the assignment's metric. It needs the model to produce the name
  *and stop*, so a pretrained model scores ~0 whether or not it "knows" the answer (L9 slide 4: the
  pretrained model got 0/150 on this very task).
* **Candidate-choice accuracy** -- score every candidate answer by its log-likelihood after the prompt
  and pick the highest. No generation, no stopping, no format: it asks only which answer the model
  finds most probable. If the pretrained model is above chance here but ~0 on exact match, what it
  lacked was the format, not the comparison. If finetuning lifts exact match but not candidate
  choice, finetuning taught the format and little else.

Plus a lenient **prefix match** (first word of the completion equals the answer, whatever follows),
which separates "right answer, did not stop" from "wrong answer".

**Batching generation.** `GPT.generate` has no KV cache and no padding mask, so a batch must share one
prompt length or the shorter prompts would be generating from pad tokens. Prompts are bucketed by
exact token length; each bucket is one batched greedy decode.

**Error taxonomy.** Every miss is assigned exactly one cause, from the first word of the completion:
    no_stop             right answer, but the completion continued instead of ending
    reversed_extreme    answered the opposite question (the smallest when asked the largest)
    middle_entity       a named entity that is neither the answer nor the reversed answer
    equal_case_miss     answered a name when the values were tied, or "equal" when they were not
    out_of_prompt_name  a real name that is not in this prompt (L9 slide 24: memorised train names)
    format_failure      not a name and not the equal word at all
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import replace

import torch

from lma.reasoning.build import problem_from_record
from lma.reasoning.facts import EQUAL, solve
from lma.reasoning.sft_data import IGNORE, Encoded, collate, encode_answer, encode_prompt

FLIP = {"max": "min", "min": "max", "more": "less", "less": "more"}
ERROR_TYPES = ("no_stop", "reversed_extreme", "middle_entity", "equal_case_miss",
               "out_of_prompt_name", "format_failure")
_PUNCT = "।॥?.,!:;\"'()[]"


# ------------------------------------------------------------------ prompts

def few_shot_prompt(record: dict, demos: list[dict]) -> str:
    """k solved demonstrations followed by the test prompt, on one line.

    Each demonstration is closed with a danda so the model sees where one answer ends and the next
    question begins. L8: in-context examples "steer a capability the weights already hold; they do
    not add one" -- the 3-shot condition is what tests that claim here.
    """
    parts = [f"{d['prompt']} {d['answer']}।" for d in demos]
    return " ".join(parts + [record["prompt"]])


def pick_demos(record: dict, pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    """k demonstrations of the same task type from the training split (never from a test split)."""
    same = [r for r in pool if r["task"] == record["task"]]
    return rng.sample(same, k)


# ------------------------------------------------------------------ generation

def _first_word(text: str) -> str:
    words = text.strip().split()
    return words[0].strip(_PUNCT) if words else ""


@torch.no_grad()
def generate_answers(model, tokenizer, prompts: list[str], device, max_new_tokens: int = 10,
                     batch_size: int = 64) -> list[dict]:
    """Greedy completions for every prompt: {"text", "stopped"} in input order.

    `text` is the decoded completion up to (not including) the first EOS; `stopped` is whether an EOS
    was produced within `max_new_tokens`. Names are 1-4 tokens, so 10 new tokens is ample room to see
    whether a model stops.
    """
    eos = tokenizer._sp.eos_id()
    encoded = [encode_prompt(tokenizer, p) for p in prompts]
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, ids in enumerate(encoded):
        buckets[len(ids)].append(i)
    out: list[dict | None] = [None] * len(prompts)
    model.eval()
    for length, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            x = torch.tensor([encoded[i] for i in chunk], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0, eos_id=eos)
            for row, i in zip(y[:, length:].tolist(), chunk):
                stopped = eos in row
                new = row[:row.index(eos)] if stopped else row
                out[i] = {"text": tokenizer.decode(new).strip(), "stopped": stopped}
    return out  # type: ignore[return-value]


# ------------------------------------------------------------------ candidate choice

@torch.no_grad()
def candidate_logprobs(model, tokenizer, prompts: list[str], options: list[list[str]], device,
                       batch_size: int = 64, with_eos: bool = False) -> list[list[float]]:
    """Summed log-probability of each option's tokens after its prompt.

    EOS is excluded by default: the pretrained model never saw an answer followed by EOS, so
    including it would penalise the baseline for the format again -- the very confound this metric
    exists to remove. Sums, not means: a name is a unit, and the comparison is between whole
    answers. Every option shares the prompt tokens exactly (see sft_data), so only answer tokens
    differ between the sequences being compared.
    """
    flat: list[tuple[int, int]] = []
    seqs = []
    for i, (p, opts) in enumerate(zip(prompts, options)):
        pid = encode_prompt(tokenizer, p)
        for j, o in enumerate(opts):
            aid = encode_answer(tokenizer, o, add_eos=with_eos)
            seq = pid + aid
            labels = [IGNORE] * (len(pid) - 1) + aid
            seqs.append((seq[:-1], labels))
            flat.append((i, j))
    scores = [[0.0] * len(o) for o in options]
    model.eval()
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]))   # similar lengths per batch
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        x, y = collate([Encoded(seqs[k][0], seqs[k][1], 0) for k in chunk])
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits, _, _ = model(x)
        logp = torch.log_softmax(logits.float(), dim=-1)
        mask = y != IGNORE
        tok = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        for k, v in zip(chunk, tok.sum(dim=1).tolist()):
            i, j = flat[k]
            scores[i][j] = v
    return scores


# ------------------------------------------------------------------ per-example judgement

def classify(record: dict, completion: dict, equal_word: str, known_names: set[str]) -> str:
    """"correct" or one of ERROR_TYPES, for a generated completion. See the module docstring."""
    gold = record["answer"]
    word = _first_word(completion["text"])
    strict = completion["text"].strip().strip("।") == gold and completion["stopped"]
    if strict:
        return "correct"
    if word == gold:
        return "no_stop"
    in_prompt = set(record["entities"])
    if word == equal_word or record["answer_key"] == EQUAL:
        if word == equal_word or word in in_prompt:
            return "equal_case_miss"
    if word in in_prompt:
        p = problem_from_record(record)
        flipped = solve(replace(p, question=FLIP[p.question]))
        return "reversed_extreme" if word == flipped else "middle_entity"
    if word in known_names:
        return "out_of_prompt_name"
    return "format_failure"


def score_records(records: list[dict], completions: list[dict], cand_scores: list[list[float]] | None,
                  equal_word: str, known_names: set[str]) -> list[dict]:
    """Per-example results: what was generated, whether it matched, which candidate won, why it failed."""
    rows = []
    for i, (r, c) in enumerate(zip(records, completions)):
        verdict = classify(r, c, equal_word, known_names)
        row = {
            "id": r["id"], "task": r["task"], "attr": r["attr"], "family": r["family"],
            "answer": r["answer"], "completion": c["text"], "stopped": c["stopped"],
            "em": int(verdict == "correct"),
            "prefix": int(_first_word(c["text"]) == r["answer"]),
            "verdict": verdict,
        }
        if cand_scores is not None:
            best = max(range(len(r["candidates"])), key=lambda j: cand_scores[i][j])
            row["choice"] = r["candidates"][best]
            row["cand_acc"] = int(r["candidates"][best] == r["answer"])
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    """Aggregate metrics overall, per task and per family, plus the error-type counts."""
    def agg(sub: list[dict]) -> dict:
        n = len(sub)
        out = {"n": n, "em": sum(r["em"] for r in sub) / n, "prefix": sum(r["prefix"] for r in sub) / n,
               "stopped": sum(r["stopped"] for r in sub) / n}
        if sub and "cand_acc" in sub[0]:
            out["cand_acc"] = sum(r["cand_acc"] for r in sub) / n
        return out

    res = {"all": agg(rows), "by_task": {}, "by_family": {}, "by_attr": {}, "errors": {}}
    for key, field in (("by_task", "task"), ("by_family", "family"), ("by_attr", "attr")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r)
        res[key] = {g: agg(v) for g, v in sorted(groups.items())}
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["verdict"] != "correct":
            counts[r["verdict"]] += 1
    res["errors"] = {e: counts.get(e, 0) for e in ERROR_TYPES}
    return res


def evaluate_records(model, tokenizer, records: list[dict], device, equal_word: str,
                     known_names: set[str], demos_pool: list[dict] | None = None, shots: int = 0,
                     candidates: bool = True, batch_size: int = 64, seed: int = 0) -> tuple[dict, list[dict]]:
    """Full evaluation of one split: returns (summary, per-example rows)."""
    rng = random.Random(seed)
    if shots:
        prompts = [few_shot_prompt(r, pick_demos(r, demos_pool, shots, rng)) for r in records]
    else:
        prompts = [r["prompt"] for r in records]
    comps = generate_answers(model, tokenizer, prompts, device, batch_size=batch_size)
    cands = (candidate_logprobs(model, tokenizer, prompts, [r["candidates"] for r in records], device,
                                batch_size=batch_size) if candidates else None)
    rows = score_records(records, comps, cands, equal_word, known_names)
    for row, p in zip(rows, prompts):
        row["prompt_tokens"] = len(encode_prompt(tokenizer, p))
    return summarise(rows), rows


def em_only(model, tokenizer, records: list[dict], device, batch_size: int = 64) -> float:
    """Strict exact match alone -- the cheap number the training loop selects checkpoints on."""
    comps = generate_answers(model, tokenizer, [r["prompt"] for r in records], device,
                             batch_size=batch_size)
    hits = [c["stopped"] and c["text"].strip().strip("।") == r["answer"] for r, c in zip(records, comps)]
    return sum(hits) / max(len(hits), 1)


def normalise_answer(text: str) -> str:
    """Whitespace- and punctuation-trimmed completion (used when printing qualitative examples)."""
    return re.sub(r"\s+", " ", text).strip().strip(_PUNCT)
