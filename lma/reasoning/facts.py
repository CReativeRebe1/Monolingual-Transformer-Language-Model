"""
Structured comparative-reasoning problems and an independent solver.

A problem is generated as *facts* first -- entities, values or pairwise relations, a question -- and
only rendered into Hindi or Nepali text afterwards, by that language's own templates. Keeping the
facts separate from the words is what makes the dataset trustworthy: every label is recomputed from
the facts by `solve()`, which shares no code with the generator. A generator bug that put the wrong
entity in the answer slot would therefore be caught, rather than taught fluently (L9's "correctness"
criterion: a model finetuned on a wrong label learns the error with full confidence).

Task types, covering every example style the assignment lists (§3.1):

    cmp2   two entities with explicit values        -> the greater / smaller one, or EQUAL on a tie
    cmp3   three entities with explicit values      -> the greatest / smallest, or EQUAL on a tie
    ext3   three entities, two relations A>B, B>C   -> the greatest / smallest (one inference step)
    pair3  the same premises, asked about A vs C    -> the multi-hop comparison itself
    ext4   four entities, three chained relations   -> the greatest / smallest (deeper chain)

Language-neutral by construction: nothing here knows a word of Hindi or Nepali.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

EQUAL = "__EQUAL__"          # sentinel; each language renders it with its own word

TASKS = ("cmp2", "cmp3", "ext3", "pair3", "ext4")
CHAIN_LEN = {"ext3": 3, "pair3": 3, "ext4": 4}


@dataclass
class Problem:
    task: str
    attr: str
    kind: str                                   # "person" | "object"
    entities: list[str]                         # every entity mentioned, in generation order
    question: str                               # "max" | "min" | "more" | "less"
    values: dict[str, int] = field(default_factory=dict)          # cmp tasks
    relations: list[tuple[str, str]] = field(default_factory=list)  # (greater, smaller), chain tasks
    pair: tuple[str, str] | None = None         # "more"/"less" questions: the two entities asked about
    item: str | None = None                     # quantity attribute: the countable noun

    def to_dict(self) -> dict:
        d = asdict(self)
        d["relations"] = [list(r) for r in self.relations]
        d["pair"] = list(self.pair) if self.pair else None
        return d


# ------------------------------------------------------------------ generation

def sample_problem(
    rng: random.Random, task: str, attr: str, kind: str, pool: list[str],
    value_range: tuple[int, int], item: str | None = None, tie_prob: float = 0.25,
) -> Problem:
    """Draws one problem. `pool` is the entity pool for this split (seen or unseen names)."""
    if task == "cmp2":
        a, b = rng.sample(pool, 2)
        lo, hi = value_range
        va = rng.randint(lo, hi)
        vb = va if rng.random() < tie_prob else _distinct(rng, lo, hi, {va})
        pair = (a, b) if rng.random() < 0.5 else (b, a)
        return Problem(task, attr, kind, [a, b], rng.choice(("more", "less")),
                       values={a: va, b: vb}, pair=pair, item=item)

    if task == "cmp3":
        names = rng.sample(pool, 3)
        lo, hi = value_range
        question = rng.choice(("max", "min"))
        used: set[int] = set()
        vals = []
        for _ in names:
            v = _distinct(rng, lo, hi, used)
            used.add(v)
            vals.append(v)
        if rng.random() < tie_prob:
            # The equal case: the two entities at the asked-for extreme share a value, so no single
            # entity is the greatest (or smallest) and the correct answer is EQUAL. The tie is placed
            # at the extreme deliberately -- a tie between the other two would not change the answer
            # and would teach nothing about equality.
            vals.sort(reverse=(question == "max"))
            vals[1] = vals[0]
            rng.shuffle(vals)
        return Problem(task, attr, kind, names, question,
                       values=dict(zip(names, vals)), item=item)

    n = CHAIN_LEN[task]
    chain = rng.sample(pool, n)                  # chain[0] > chain[1] > ... (hidden order)
    relations = [(chain[i], chain[i + 1]) for i in range(n - 1)]
    rng.shuffle(relations)                       # premise order must not reveal the chain
    if task == "pair3":
        pair = (chain[0], chain[2]) if rng.random() < 0.5 else (chain[2], chain[0])
        question = rng.choice(("more", "less"))
    else:
        pair, question = None, rng.choice(("max", "min"))
    mentioned = list(chain)
    rng.shuffle(mentioned)                       # entity list order must not reveal the chain either
    return Problem(task, attr, kind, mentioned, question,
                   relations=relations, pair=pair, item=item)


def _distinct(rng: random.Random, lo: int, hi: int, avoid: set[int]) -> int:
    if hi - lo + 1 <= len(avoid):
        raise ValueError(f"range {lo}-{hi} too small to avoid {len(avoid)} values")
    while True:
        v = rng.randint(lo, hi)
        if v not in avoid:
            return v


# ------------------------------------------------------------------ independent solver

def solve(p: Problem) -> str:
    """Recomputes the answer from the facts alone.

    Deliberately shares nothing with `sample_problem`: for chain tasks it does not use the hidden
    chain order at all, but rebuilds the ordering from the stated relations by transitive closure.
    If a premise were dropped or reversed during generation, this would disagree with the label the
    generator intended -- which is exactly the failure it exists to expose.
    """
    if p.values:
        vals = p.values
        if p.question in ("more", "less"):
            x, y = p.pair
            if vals[x] == vals[y]:
                return EQUAL
            bigger = x if vals[x] > vals[y] else y
            return bigger if p.question == "more" else (y if bigger == x else x)
        pick = max if p.question == "max" else min
        best = pick(vals.values())
        winners = [n for n, v in vals.items() if v == best]
        return winners[0] if len(winners) == 1 else EQUAL

    greater = _closure(p.relations)              # greater[x] = every entity x is greater than
    if p.question in ("more", "less"):
        x, y = p.pair
        if y in greater.get(x, set()):
            bigger, smaller = x, y
        elif x in greater.get(y, set()):
            bigger, smaller = y, x
        else:
            raise ValueError(f"{x} and {y} are not comparable from the premises")
        return bigger if p.question == "more" else smaller

    nodes = {e for rel in p.relations for e in rel}
    if p.question == "max":                      # greater than every other node
        winners = [n for n in nodes if greater.get(n, set()) == nodes - {n}]
    else:                                        # nothing is smaller than it
        winners = [n for n in nodes if not greater.get(n)]
    if len(winners) != 1:
        raise ValueError(f"premises do not determine a unique {p.question}: {p.relations}")
    return winners[0]


def _closure(relations: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Transitive closure of 'is greater than'."""
    direct: dict[str, set[str]] = {}
    for g, s in relations:
        direct.setdefault(g, set()).add(s)
    closed: dict[str, set[str]] = {}
    for start in direct:
        seen: set[str] = set()
        stack = list(direct[start])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(direct.get(node, ()))
        closed[start] = seen
    return closed


def candidates(p: Problem) -> list[str]:
    """The answer options a well-formed reply chooses among: every mentioned entity, plus EQUAL
    where a tie is a possible answer. Used for candidate-choice scoring and heuristic baselines."""
    opts = list(p.entities)
    if p.task in ("cmp2", "cmp3"):
        opts.append(EQUAL)
    return opts
