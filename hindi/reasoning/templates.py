"""
Hindi surface forms for the reasoning dataset: four template families per task type.

Renders a language-neutral `lma.reasoning.facts.Problem` into one line of Hindi. The families differ
in *how* a comparison is worded, not in what it says, so a model that learnt the task rather than
the phrasing should carry over to the one it never saw:

    F1  adjective comparative      "राहुल, अमित से बड़ा है।"          "सबसे बड़ा कौन है?"
    F2  noun + ज्यादा/कम            "राहुल की उम्र अमित से ज्यादा है।"  "किसकी उम्र सबसे ज्यादा है?"
    F3  synonym noun + तुलना        "अमित की तुलना में राहुल की आयु अधिक है।"
    F4  held out: के मुकाबले, इनमें  "अमित के मुकाबले राहुल बड़ा है।"   "इनमें सबसे बड़ा कौन है?"

F1-F3 are used for training; F4 appears only in `test_wording` and `test_both` (L9 slide 34: write
four surface forms, train on three, hold the fourth back). F4 changes the frame (standard of
comparison moved in front of the subject), the question opener and the value sentences, so
matching it needs more than a new synonym.

**Every family states relations in both directions.** "A, B से बड़ा है" and "B, A से छोटा है" carry
the same fact; each premise picks one at random. Without this the greater entity would always be
the grammatical subject and "answer the first name" would score perfectly on every chain task --
and a model trained only on "A > B" is not guaranteed to handle "B < A" (the reversal curse).

Grammar handled rather than ignored:
* adjectives agree with the subject's gender (बड़ा / बड़ी), and the question uses the feminine only
  when every candidate is female ("सबसे बड़ी कौन है?"), the masculine generic otherwise
* the genitive agrees with the possessed noun, not the owner: "राहुल की उम्र", "सीता का वजन"
* objects are asked about as a feminine "कौन सी चीज", because चीज is feminine
* numbers use ASCII digits, which is what 92% of digits in the Hindi corpus are

Everything returned is passed through Hindi's own Phase-1 normaliser, so the text is in exactly the
form the pretraining corpus was in (anusvara for chandrabindu, NFC, danda spacing).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hindi.clean.norm import norm  # noqa: E402
from hindi.reasoning.lexicon import GENDER  # noqa: E402
from lma.reasoning.facts import EQUAL, Problem  # noqa: E402

LANG = "hindi"
FAMILIES = ("F1", "F2", "F3", "F4")
TRAIN_FAMILIES = ("F1", "F2", "F3")
HELDOUT_FAMILY = "F4"

EQUAL_WORD = "बराबर"
PROMPT = "प्रश्न: {body} उत्तर:"

# Comparative vocabulary. `noun1` is used by F1/F2/F4, `noun3` (a synonym) by F3, each with its
# grammatical gender because the genitive before it agrees with it.
ATTRS = {
    "age":        dict(noun1=("उम्र", "f"), noun3=("आयु", "f"), unit1="साल", unit3="वर्ष",
                       pos="बड़ा", neg="छोटा"),
    "height":     dict(noun1=("लंबाई", "f"), noun3=("कद", "m"), unit1="सेंटीमीटर", unit3="सेमी",
                       pos="लंबा", neg="छोटा"),
    "weight":     dict(noun1=("वजन", "m"), noun3=("भार", "m"), unit1="किलो", unit3="किलोग्राम",
                       pos="भारी", neg="हल्का"),
    "price":      dict(noun1=("कीमत", "f"), noun3=("दाम", "m"), unit1="रुपये", unit3="रुपये",
                       pos="महंगा", neg="सस्ता"),
    "obj_weight": dict(noun1=("वजन", "m"), noun3=("भार", "m"), unit1="किलो", unit3="किलोग्राम",
                       pos="भारी", neg="हल्का"),
    "quantity":   dict(pos="ज्यादा", neg="कम"),
}
# Quantity words by family: "ज्यादा" (F1, F2, F4), "अधिक" (F3); "कम" is the opposite in all.
MORE = {"F1": "ज्यादा", "F2": "ज्यादा", "F3": "अधिक", "F4": "ज्यादा"}
LESS = "कम"

# Words a polarity check may look for. Exposed so tests/test_reasoning.py can verify that each
# rendered relation says what the facts say, without re-implementing the templates.
POSITIVE_WORDS = {"बड़ा", "बड़ी", "लंबा", "लंबी", "भारी", "महंगा", "महंगी", "ज्यादा", "अधिक"}
NEGATIVE_WORDS = {"छोटा", "छोटी", "हल्का", "हल्की", "सस्ता", "सस्ती", "कम"}
STANDARD_MARKERS = (("से",), ("की", "तुलना", "में"), ("के", "मुकाबले"))


def num(n: int) -> str:
    """Hindi writes numbers in ASCII digits (92% of digits in the Hindi corpus)."""
    return str(n)


def fem(adj: str) -> str:
    """Feminine form of an -ā adjective (बड़ा -> बड़ी). Adjectives not ending in -ā do not inflect."""
    return adj[:-1] + "ी" if adj.endswith("ा") else adj


def agree(adj: str, gender: str) -> str:
    """The adjective in the form that agrees with a subject of `gender`."""
    return fem(adj) if gender == "f" else adj


def gen(noun_gender: str) -> str:
    """Genitive postposition agreeing with the possessed noun: की (feminine) / का (masculine)."""
    return "की" if noun_gender == "f" else "का"


def answer_text(answer: str) -> str:
    """The target string for a solved answer: the entity itself, or बराबर for a tie."""
    return EQUAL_WORD if answer == EQUAL else answer


# ------------------------------------------------------------------ sentences

def value_sentence(e: str, n: int, attr: str, family: str, item: str | None = None) -> str:
    """One premise stating an entity's value: "राहुल की उम्र 25 साल है।"."""
    g = GENDER[e]
    v = num(n)
    if attr == "quantity":
        return {
            "F1": f"{e} के पास {v} {item} हैं।",
            "F2": f"{v} {item} {e} के पास हैं।",
            "F3": f"{e} के पास पूरे {v} {item} हैं।",
            "F4": f"{e} के पास कुल {v} {item} हैं।",
        }[family]
    a = ATTRS[attr]
    (n1, g1), (n3, g3) = a["noun1"], a["noun3"]
    if family == "F1":
        return f"{e} {gen(g1)} {n1} {v} {a['unit1']} है।"
    if family == "F2":
        tail = agree("लंबा", g) if attr == "height" else gen(g)
        return f"{e} {v} {a['unit1']} {tail} है।"
    if family == "F3":
        return f"{e} {gen(g3)} {n3} {v} {a['unit3']} है।"
    return {                                             # F4
        "age": f"{e} ने {v} साल पूरे कर लिए हैं।",
        "height": f"नापने पर {e} {gen(g1)} {n1} {v} सेंटीमीटर निकली।",
        "weight": f"तौलने पर {e} का वजन {v} किलो निकला।",
        "obj_weight": f"तौलने पर {e} का वजन {v} किलो निकला।",
        "price": f"{e} {v} रुपये में {agree('मिलता', g)} है।",
    }[attr]


def relation_sentence(greater: str, smaller: str, attr: str, family: str, up: bool,
                      item: str | None = None) -> str:
    """One premise stating a comparison. `up=True` makes the greater entity the subject with the
    positive word ("राहुल, अमित से बड़ा है"); `up=False` states the same fact from the other side
    ("अमित, राहुल से छोटा है")."""
    s, t = (greater, smaller) if up else (smaller, greater)
    if attr == "quantity":
        w = MORE[family] if up else LESS
        return {
            "F1": f"{s} के पास {t} से {w} {item} हैं।",
            "F2": f"{s} के पास {item} {t} से {w} हैं।",
            "F3": f"{t} की तुलना में {s} के पास {w} {item} हैं।",
            "F4": f"{t} के मुकाबले {s} के पास {w} {item} हैं।",
        }[family]
    a = ATTRS[attr]
    adj = agree(a["pos"] if up else a["neg"], GENDER[s])
    (n1, g1), (n3, g3) = a["noun1"], a["noun3"]
    return {
        "F1": f"{s}, {t} से {adj} है।",
        "F2": f"{s} {gen(g1)} {n1} {t} से {'ज्यादा' if up else 'कम'} है।",
        "F3": f"{t} की तुलना में {s} {gen(g3)} {n3} {'अधिक' if up else 'कम'} है।",
        "F4": f"{t} के मुकाबले {s} {adj} है।",
    }[family]


def question_sentence(p: Problem, family: str) -> str:
    """The question, in the family's wording, for "max"/"min" over all entities or "more"/"less"
    over `p.pair`."""
    positive = p.question in ("max", "more")
    pair = p.question in ("more", "less")
    x, y = p.pair if pair else (None, None)
    if p.attr == "quantity":
        w = MORE[family] if positive else LESS
        item = p.item
        if pair:
            return {
                "F1": f"{x} और {y} में से किसके पास {w} {item} हैं?",
                "F2": f"{x} और {y} में से किसके पास {item} {w} हैं?",
                "F3": f"{x} और {y} में किसके पास {w} {item} हैं?",
                "F4": f"{x} या {y}, दोनों में {w} {item} किसके पास हैं?",
            }[family]
        return {
            "F1": f"सबसे {w} {item} किसके पास हैं?",
            "F2": f"किसके पास {item} सबसे {w} हैं?",
            "F3": f"किसके पास सबसे {w} {item} हैं?",
            "F4": f"इनमें सबसे {w} {item} किसके पास हैं?",
        }[family]

    a = ATTRS[p.attr]
    (n1, g1), (n3, g3) = a["noun1"], a["noun3"]
    base = a["pos"] if positive else a["neg"]
    amount = {"F2": "ज्यादा", "F3": "अधिक"}.get(family, "ज्यादा") if positive else "कम"
    asked = [x, y] if pair else p.entities

    if p.kind == "object":
        adj = fem(base)                                   # agrees with चीज (feminine)
        if pair:
            return {
                "F1": f"{x} और {y} में से कौन सी चीज {adj} है?",
                "F2": f"{x} और {y} में से किस चीज {gen(g1)} {n1} {amount} है?",
                "F3": f"{x} और {y} में किस चीज {gen(g3)} {n3} {amount} है?",
                "F4": f"{x} या {y}, दोनों में {adj} कौन सी है?",
            }[family]
        return {
            "F1": f"कौन सी चीज सबसे {adj} है?",
            "F2": f"किस चीज {gen(g1)} {n1} सबसे {amount} है?",
            "F3": f"किस चीज {gen(g3)} {n3} सबसे {amount} है?",
            "F4": f"इनमें सबसे {adj} चीज कौन सी है?",
        }[family]

    all_female = all(GENDER[e] == "f" for e in asked)
    adj = fem(base) if all_female else base
    whose1, whose3 = "किस" + gen(g1), "किस" + gen(g3)           # किसकी / किसका
    if pair:
        return {
            "F1": f"{x} और {y} में से कौन {adj} है?",
            "F2": f"{x} और {y} में से {whose1} {n1} {amount} है?",
            "F3": f"{x} और {y} में {whose3} {n3} {amount} है?",
            "F4": f"{x} या {y}, दोनों में {adj} कौन है?",
        }[family]
    return {
        "F1": f"सबसे {adj} कौन है?",
        "F2": f"{whose1} {n1} सबसे {amount} है?",
        "F3": f"{whose3} {n3} सबसे {amount} है?",
        "F4": f"इनमें सबसे {adj} कौन है?",
    }[family]


# ------------------------------------------------------------------ whole prompt

def render(p: Problem, family: str, rng: random.Random) -> str:
    """The full single-line prompt body: premises, then the question. Not yet wrapped in PROMPT.

    Value premises follow `p.entities` (already randomly ordered by the generator). Relation premises
    follow `p.relations` (already shuffled) and each picks its direction independently, so neither
    sentence order nor grammatical role reveals the answer.
    """
    if p.values:
        premises = [value_sentence(e, p.values[e], p.attr, family, p.item) for e in p.entities]
    else:
        premises = [relation_sentence(g, s, p.attr, family, rng.random() < 0.5, p.item)
                    for g, s in p.relations]
    return norm(" ".join(premises + [question_sentence(p, family)]))


def prompt(body: str) -> str:
    """Wraps a rendered body in the question/answer markers the model is finetuned on."""
    return PROMPT.format(body=body)
