"""
Dataset assembly: entity pools, the 2x2 split design, rendering, deduplication, statistics.

Language-neutral. A language contributes a `lexicon` module (names, objects, items) and a
`templates` module (how a Problem is worded); everything about *which* problems go into *which*
split, and how leakage is ruled out, lives here once, so Hindi and Nepali are built by identical
rules and differ only in their words.

**Split design** -- a 2x2 factorial over the two things a model could memorise instead of learning
the task (L9 slide 34 asks for three of these cells; the fourth isolates the interaction for free):

                         seen wording (F1-F3)     unseen wording (F4)
    seen entities        test_id                  test_wording
    unseen entities      test_entity              test_both

`train` and `val` use seen entities and F1-F3. `val` is the only split used for any selection
(learning rate, early stopping); the four test splits are read once, at the end.

**Leakage is excluded by construction and then checked** (`leakage_report`):
* entities: the person and object pools are split 70/30 per gender before anything is sampled, so
  an unseen-entity split cannot mention a training name
* wording: F4 is never rendered into train/val/test_id/test_entity
* examples: every problem's facts are reduced to a canonical key, and a key already used by any
  split is rejected, so the same comparison cannot appear in train and test under different wording
  -- a stricter test than comparing prompt strings, which a reworded copy would pass
* labels: every answer is recomputed by `facts.solve`, which shares no code with the generator

**Balance.** Task type x attribute combinations are cycled rather than sampled, so every split has
the same mix (per-task accuracy is reported, and a split that happened to be 30% cmp2 would make
its aggregate incomparable). Families F1-F3 are cycled the same way.

Fully seeded: the same seed produces byte-identical JSONL (asserted by a test).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType

from lma.reasoning.facts import EQUAL, TASKS, Problem, candidates, sample_problem, solve

# attribute -> (entity kind, inclusive value range). Ranges are chosen to be realistic for the
# quantity and to keep most values at two digits, where number tokenisation is least erratic.
ATTRIBUTES: dict[str, tuple[str, tuple[int, int]]] = {
    "age":        ("person", (8, 80)),
    "height":     ("person", (120, 195)),
    "weight":     ("person", (30, 95)),
    "price":      ("object", (20, 950)),
    "obj_weight": ("object", (2, 40)),
    "quantity":   ("person", (2, 30)),
}

# split -> (entity pool, family set)
SPLITS: dict[str, tuple[str, str]] = {
    "train":        ("seen", "train"),
    "val":          ("seen", "train"),
    "test_id":      ("seen", "train"),
    "test_entity":  ("unseen", "train"),
    "test_wording": ("seen", "heldout"),
    "test_both":    ("unseen", "heldout"),
}
TEST_SPLITS = ("test_id", "test_entity", "test_wording", "test_both")

# Evaluation splits are generated before train, so they claim their fact keys first and train is
# the split that has to avoid them -- never the other way round.
GENERATION_ORDER = ("test_id", "test_entity", "test_wording", "test_both", "val", "train")


@dataclass
class BuildConfig:
    """Everything that determines the dataset. Saved next to it so a copy can be regenerated."""
    seed: int = 1234
    sizes: dict[str, int] = field(default_factory=lambda: {
        "train": 20_000, "val": 1_000, "test_id": 1_000, "test_entity": 1_000,
        "test_wording": 1_000, "test_both": 1_000,
    })
    seen_fraction: float = 0.7
    tie_prob: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ pools

def split_pool(entries: tuple[tuple[str, str], ...], seen_fraction: float,
               rng: random.Random) -> tuple[list[str], list[str]]:
    """Splits (name, gender) entries into disjoint seen/unseen name lists, stratified by gender so
    both pools keep the same male/female mix (agreement is part of what the templates test)."""
    seen, unseen = [], []
    by_gender: dict[str, list[str]] = defaultdict(list)
    for name, g in entries:
        by_gender[g].append(name)
    for g in sorted(by_gender):
        names = sorted(by_gender[g])
        rng.shuffle(names)
        k = round(len(names) * seen_fraction)
        seen += names[:k]
        unseen += names[k:]
    return sorted(seen), sorted(unseen)


def make_pools(lexicon: ModuleType, cfg: BuildConfig) -> dict[str, dict[str, list[str]]]:
    """{"seen"|"unseen": {"person": [...], "object": [...]}}, seeded independently of sampling."""
    rng = random.Random(f"{cfg.seed}:pools")
    p_seen, p_unseen = split_pool(lexicon.PERSONS, cfg.seen_fraction, rng)
    o_seen, o_unseen = split_pool(lexicon.OBJECTS, cfg.seen_fraction, rng)
    return {"seen": {"person": p_seen, "object": o_seen},
            "unseen": {"person": p_unseen, "object": o_unseen}}


# ------------------------------------------------------------------ examples

def fact_key(p: Problem) -> str:
    """Canonical identity of a problem's content, independent of wording and premise order."""
    return json.dumps({
        "task": p.task, "attr": p.attr, "q": p.question, "item": p.item,
        "values": sorted(p.values.items()), "rel": sorted(map(list, p.relations)),
        "pair": list(p.pair) if p.pair else None,
    }, ensure_ascii=False, sort_keys=True)


def make_record(p: Problem, family: str, split: str, index: int, templates: ModuleType,
                rng: random.Random) -> dict:
    """Renders one problem and packages it with everything evaluation and analysis need."""
    gold = solve(p)
    body = templates.render(p, family, rng)
    return {
        "id": f"{templates.LANG}-{split}-{index:05d}",
        "split": split,
        "task": p.task,
        "attr": p.attr,
        "kind": p.kind,
        "family": family,
        "question": p.question,
        "prompt": templates.prompt(body),
        "answer": templates.answer_text(gold),
        "answer_key": gold,
        "candidates": [templates.answer_text(c) for c in candidates(p)],
        "entities": list(p.entities),
        "facts": p.to_dict(),
    }


def build_split(split: str, n: int, pools: dict, lexicon: ModuleType, templates: ModuleType,
                cfg: BuildConfig, used_keys: set[str], used_prompts: set[str]) -> list[dict]:
    """Generates `n` unique examples for one split, claiming their fact keys and prompts."""
    pool_name, family_set = SPLITS[split]
    families = templates.TRAIN_FAMILIES if family_set == "train" else (templates.HELDOUT_FAMILY,)
    rng = random.Random(f"{cfg.seed}:{split}")
    # Cycled over the full product. Cycling task x attribute and family separately would lock them
    # together whenever one period divides the other (30 and 3 do): F1 would only ever be rendered
    # for two of the six attributes. The first build did exactly that; the skeleton count exposed it.
    combos = [(t, a, f) for t in TASKS for a in ATTRIBUTES for f in families]
    records: list[dict] = []
    attempts = 0
    while len(records) < n:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError(f"{split}: could not find {n} unique problems (got {len(records)})")
        task, attr, family = combos[len(records) % len(combos)]
        kind, value_range = ATTRIBUTES[attr]
        item = rng.choice(lexicon.ITEMS) if attr == "quantity" else None
        p = sample_problem(rng, task, attr, kind, pools[pool_name][kind], value_range,
                           item=item, tie_prob=cfg.tie_prob)
        key = fact_key(p)
        if key in used_keys:
            continue
        rec = make_record(p, family, split, len(records), templates, rng)
        if rec["prompt"] in used_prompts:
            continue
        used_keys.add(key)
        used_prompts.add(rec["prompt"])
        records.append(rec)
    # Cycling fixed the mix; shuffling removes the periodic order so a prefix is also balanced-ish
    # and nothing downstream can key on position. Ids are reassigned to match the final order.
    rng.shuffle(records)
    for i, rec in enumerate(records):
        rec["id"] = f"{templates.LANG}-{split}-{i:05d}"
    return records


def build_dataset(lexicon: ModuleType, templates: ModuleType, cfg: BuildConfig | None = None
                  ) -> tuple[dict[str, list[dict]], dict]:
    """All six splits plus the pools they were drawn from."""
    cfg = cfg or BuildConfig()
    pools = make_pools(lexicon, cfg)
    used_keys: set[str] = set()
    used_prompts: set[str] = set()
    data = {}
    for split in GENERATION_ORDER:
        data[split] = build_split(split, cfg.sizes[split], pools, lexicon, templates, cfg,
                                  used_keys, used_prompts)
    return {s: data[s] for s in SPLITS}, pools


# ------------------------------------------------------------------ io

def write_jsonl(records: list[dict], path: Path) -> Path:
    """One JSON object per line, UTF-8, LF line endings -- byte-stable across platforms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    """Reads a split written by `write_jsonl`."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ text analysis

def body_of(prompt: str) -> str:
    """The prompt without its question/answer markers (text between the first ':' and the last)."""
    return prompt[prompt.index(":") + 1:prompt.rindex(":")].strip() if ":" in prompt else prompt


def premises_of(prompt: str) -> str:
    """The premise sentences of a prompt: its body up to and including the last danda (every
    question ends in '?', every premise in '।')."""
    body = body_of(prompt)
    return body[:body.rindex("।") + 1] if "।" in body else body


def mentions(text: str, names: list[str]) -> dict[str, list[int]]:
    """Character offsets of every occurrence of each name. Safe because no name in a pool is a
    substring of another (asserted in tests), so an occurrence can only belong to one entity."""
    out = {}
    for name in names:
        pos, start = [], text.find(name)
        while start != -1:
            pos.append(start)
            start = text.find(name, start + 1)
        out[name] = pos
    return out


# ------------------------------------------------------------------ heuristic baselines

HEURISTICS = ("first_mentioned", "last_mentioned", "most_frequent", "random_candidate")


def heuristic_answer(record: dict, heuristic: str, rng: random.Random) -> str:
    """What a shortcut would answer for this example, as answer text.

    These are the shortcuts a model can take instead of reasoning. If any scores well above chance,
    the dataset leaks the answer through surface position or frequency, and a finetuned model's
    accuracy could be that shortcut rather than comparison.

    Position and frequency are read from the premises only. The question names the two entities
    it asks about, so "last name in the prompt" would just be the second option of the question --
    a coin flip between the two right candidates, not a property of how the facts were written.
    """
    if heuristic == "random_candidate":
        return rng.choice(record["candidates"])
    text = premises_of(record["prompt"])
    found = {n: p for n, p in mentions(text, record["entities"]).items() if p}
    if not found:
        return ""
    if heuristic == "first_mentioned":
        return min(found, key=lambda n: found[n][0])
    if heuristic == "last_mentioned":
        return max(found, key=lambda n: found[n][-1])
    if heuristic == "most_frequent":
        return max(found, key=lambda n: (len(found[n]), -found[n][0]))
    raise ValueError(heuristic)


def heuristic_accuracy(records: list[dict], seed: int = 0) -> dict[str, dict[str, float]]:
    """{heuristic: {"all": acc, <task>: acc, ...}} over a split."""
    rng = random.Random(seed)
    out = {}
    for h in HEURISTICS:
        hits: dict[str, list[int]] = defaultdict(list)
        for r in records:
            ok = int(heuristic_answer(r, h, rng) == r["answer"])
            hits["all"].append(ok)
            hits[r["task"]].append(ok)
        out[h] = {k: sum(v) / len(v) for k, v in hits.items()}
    return out


def chance_accuracy(records: list[dict]) -> dict[str, float]:
    """Expected accuracy of a uniform pick among candidates, per task and overall."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in records:
        c = 1.0 / len(r["candidates"])
        per["all"].append(c)
        per[r["task"]].append(c)
    return {k: sum(v) / len(v) for k, v in per.items()}


# ------------------------------------------------------------------ statistics & leakage

_DIGITS_RE = re.compile(r"[0-9०-९]+")
_SENTENCE_RE = re.compile(r"[^।?]+[।?]")


def skeletons(record: dict, items: tuple[str, ...]) -> list[str]:
    """Sentence templates observed in one prompt: entities -> X, numbers -> N, items -> I.
    Counting distinct skeletons measures template variety from the data itself rather than from
    what the template file claims to contain."""
    text = body_of(record["prompt"])
    for name in sorted(record["entities"], key=len, reverse=True):
        text = text.replace(name, "X")
    for it in sorted(items, key=len, reverse=True):
        text = re.sub(rf"(?<![\wऀ-ॿ]){re.escape(it)}(?![\wऀ-ॿ])", "I", text)
    text = _DIGITS_RE.sub("N", text)
    return [s.strip() for s in _SENTENCE_RE.findall(text)]


def dataset_stats(data: dict[str, list[dict]], pools: dict, lexicon: ModuleType, cfg: BuildConfig,
                  tokenizer=None) -> dict:
    """The dataset report the assignment asks for: size, template variety, leakage checks."""
    stats: dict = {"config": cfg.to_dict(), "splits": {}, "pools": {}, "leakage": leakage_report(data, pools)}
    for pool, kinds in pools.items():
        stats["pools"][pool] = {k: len(v) for k, v in kinds.items()}
    stats["pools"]["items"] = len(lexicon.ITEMS)

    for split, recs in data.items():
        s: dict = {
            "n": len(recs),
            "distinct_prompts": len({r["prompt"] for r in recs}),
            "by_task": dict(Counter(r["task"] for r in recs)),
            "by_attr": dict(Counter(r["attr"] for r in recs)),
            "by_family": dict(Counter(r["family"] for r in recs)),
            "equal_answers": sum(r["answer_key"] == EQUAL for r in recs),
            "heuristics": heuristic_accuracy(recs),
            "chance": chance_accuracy(recs),
        }
        sk = defaultdict(set)
        for r in recs:
            for t in skeletons(r, lexicon.ITEMS):
                sk[r["family"]].add(t)
        s["sentence_templates"] = {f: len(v) for f, v in sorted(sk.items())}
        chars = sorted(len(r["prompt"]) for r in recs)
        s["prompt_chars"] = _summary(chars)
        if tokenizer is not None:
            s["prompt_tokens"] = _summary(sorted(len(tokenizer.encode(r["prompt"])) for r in recs))
        stats["splits"][split] = s

    all_sk = defaultdict(set)
    for recs in data.values():
        for r in recs:
            for t in skeletons(r, lexicon.ITEMS):
                all_sk[r["family"]].add(t)
    stats["sentence_templates_total"] = {f: len(v) for f, v in sorted(all_sk.items())}
    return stats


def _summary(xs: list[int]) -> dict[str, float]:
    """min / median / p95 / max / mean of an already sorted list."""
    if not xs:
        return {}
    return {"min": xs[0], "median": xs[len(xs) // 2], "p95": xs[int(0.95 * (len(xs) - 1))],
            "max": xs[-1], "mean": round(sum(xs) / len(xs), 1)}


def leakage_report(data: dict[str, list[dict]], pools: dict) -> dict:
    """Every overlap that would make a test number optimistic. All counts must be zero (except the
    seen-entity splits sharing names with train, which is their definition)."""
    report: dict = {}
    names = list(data)
    prompts = {s: {r["prompt"] for r in data[s]} for s in names}
    keys = {s: {fact_key(Problem(**_problem_args(r["facts"]))) for r in data[s]} for s in names}
    report["prompt_overlap"] = {f"{a}|{b}": len(prompts[a] & prompts[b])
                                for i, a in enumerate(names) for b in names[i + 1:]}
    report["fact_overlap"] = {f"{a}|{b}": len(keys[a] & keys[b])
                              for i, a in enumerate(names) for b in names[i + 1:]}
    seen = set(pools["seen"]["person"]) | set(pools["seen"]["object"])
    unseen = set(pools["unseen"]["person"]) | set(pools["unseen"]["object"])
    report["pool_overlap"] = len(seen & unseen)
    report["unseen_names_in_seen_splits"] = sum(
        bool(set(r["entities"]) & unseen) for s in names if SPLITS[s][0] == "seen" for r in data[s])
    report["seen_names_in_unseen_splits"] = sum(
        bool(set(r["entities"]) & seen) for s in names if SPLITS[s][0] == "unseen" for r in data[s])
    train_fams = {r["family"] for s in names if SPLITS[s][1] == "train" for r in data[s]}
    held_fams = {r["family"] for s in names if SPLITS[s][1] == "heldout" for r in data[s]}
    report["family_overlap"] = sorted(train_fams & held_fams)
    return report


def _problem_args(d: dict) -> dict:
    """Problem constructor arguments from `Problem.to_dict()` output (lists back to tuples)."""
    d = dict(d)
    d["relations"] = [tuple(r) for r in d["relations"]]
    d["pair"] = tuple(d["pair"]) if d["pair"] else None
    return d


def problem_from_record(record: dict) -> Problem:
    """Rebuilds the structured problem stored with an example (for re-solving and analysis)."""
    return Problem(**_problem_args(record["facts"]))
