"""
Nepali surface forms for the reasoning dataset: four template families per task type.

Renders a language-neutral `lma.reasoning.facts.Problem` into one line of Nepali. Written for
Nepali, not translated from the Hindi file -- the two languages build a comparison differently:

    F1  adjective comparative      "हरि श्यामभन्दा जेठो छ।"            "सबैभन्दा जेठो को हो?"
    F2  noun + बढी/कम               "हरिको उमेर श्यामभन्दा बढी छ।"      "कसको उमेर सबैभन्दा बढी छ?"
    F3  synonym noun + तुलनामा      "श्यामको तुलनामा हरिको आयु अधिक छ।"
    F4  held out: standard first   "श्यामभन्दा हरि जेठो छ।"            "यीमध्ये सबैभन्दा जेठो को हो?"

F1-F3 train; F4 only appears in `test_wording` and `test_both` (L9 slide 34).

Where Nepali diverges from Hindi, and what the templates do about it:

* **The comparative postposition attaches**: हरिभन्दा, not "हरि से". The standard of comparison is
  therefore a suffix on the name, and F4 can move it in front of the subject without any extra
  word -- a reordering Hindi needs "के मुकाबले" for.
* **The copula agrees**: छ for a man or an object, छिन for a woman. Adjectives agree too
  (जेठो / जेठी), and only when every candidate is female does the question use the feminine.
* **Counts use the classifier वटा with a singular noun** ("५ वटा स्याउ छन") and the plural copula छन.
* **Numbers are written in Devanagari digits**, which are 90% of the digits in the Nepali corpus --
  the opposite of Hindi's 92% ASCII.
* **No dangling virama**: the corpus normaliser strips a final halant, so the model saw छन and छिन,
  never छन् and छिन्. The templates write the forms the model saw.

Everything returned is passed through Nepali's own Phase-1 normaliser (which, unlike Hindi's, keeps
chandrabindu: गह्रौँ stays गह्रौँ).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lma.reasoning.facts import EQUAL, Problem  # noqa: E402
from lma.text.normalize import convert_digits  # noqa: E402
from nepali.clean.norm import norm  # noqa: E402
from nepali.reasoning.lexicon import GENDER  # noqa: E402

LANG = "nepali"
FAMILIES = ("F1", "F2", "F3", "F4")
TRAIN_FAMILIES = ("F1", "F2", "F3")
HELDOUT_FAMILY = "F4"

EQUAL_WORD = "बराबर"
PROMPT = "प्रश्न: {body} उत्तर:"

ATTRS = {
    "age":        dict(noun1="उमेर", noun3="आयु", unit1="वर्ष", unit3="साल", pos="जेठो", neg="कान्छो"),
    "height":     dict(noun1="उचाइ", noun3="लम्बाइ", unit1="सेन्टिमिटर", unit3="सेमी",
                       pos="अग्लो", neg="होचो"),
    "weight":     dict(noun1="तौल", noun3="वजन", unit1="किलो", unit3="किलोग्राम",
                       pos="गह्रौँ", neg="हलुका"),
    "price":      dict(noun1="मूल्य", noun3="दाम", unit1="रुपैयाँ", unit3="रुपैयाँ",
                       pos="महँगो", neg="सस्तो"),
    "obj_weight": dict(noun1="तौल", noun3="वजन", unit1="किलो", unit3="किलोग्राम",
                       pos="गह्रौँ", neg="हलुका"),
    "quantity":   dict(pos="बढी", neg="कम"),
}
MORE = {"F1": "बढी", "F2": "बढी", "F3": "अधिक", "F4": "धेरै"}
LESS = {"F1": "कम", "F2": "कम", "F3": "कम", "F4": "थोरै"}

# Exposed for the polarity check in tests/test_reasoning.py.
POSITIVE_WORDS = {"जेठो", "जेठी", "अग्लो", "अग्ली", "गह्रौँ", "महँगो", "महँगी", "बढी", "अधिक", "धेरै"}
NEGATIVE_WORDS = {"कान्छो", "कान्छी", "होचो", "होची", "हलुका", "सस्तो", "सस्ती", "कम", "थोरै"}
STANDARD_SUFFIX = "भन्दा"          # attached: हरिभन्दा
COMPARE_WORD = "तुलनामा"           # after a genitive: हरिको तुलनामा


def num(n: int) -> str:
    """Nepali writes numbers in Devanagari digits (90% of digits in the Nepali corpus)."""
    return convert_digits(str(n), "devanagari")


def fem(adj: str) -> str:
    """Feminine form of an -o adjective (जेठो -> जेठी). Others (गह्रौँ, हलुका) do not inflect."""
    return adj[:-1] + "ी" if adj.endswith("ो") else adj


def agree(adj: str, gender: str) -> str:
    """The adjective in the form that agrees with a subject of `gender`."""
    return fem(adj) if gender == "f" else adj


def cop(gender: str) -> str:
    """Third-person copula: छिन for a woman, छ for a man or an object."""
    return "छिन" if gender == "f" else "छ"


def answer_text(answer: str) -> str:
    """The target string for a solved answer: the entity itself, or बराबर for a tie."""
    return EQUAL_WORD if answer == EQUAL else answer


# ------------------------------------------------------------------ sentences

def value_sentence(e: str, n: int, attr: str, family: str, item: str | None = None) -> str:
    """One premise stating an entity's value: "हरिको उमेर २५ वर्ष छ।"."""
    g = GENDER[e]
    v = num(n)
    if attr == "quantity":
        return {
            "F1": f"{e}सँग {v} वटा {item} छन।",
            "F2": f"{v} वटा {item} {e}सँग छन।",
            "F3": f"{e}सँग पूरा {v} वटा {item} छन।",
            "F4": f"{e}सँग जम्मा {v} वटा {item} छन।",
        }[family]
    a = ATTRS[attr]
    if family == "F1":
        return f"{e}को {a['noun1']} {v} {a['unit1']} {cop('m')}।"
    if family == "F2":
        if attr == "height":
            return f"{e} {v} सेन्टिमिटर {agree('अग्लो', g)} {cop(g)}।"
        genitive = "की" if g == "f" else "को"
        return f"{e} {v} {a['unit1']}{genitive} {cop(g)}।"
    if family == "F3":
        return f"{e}को {a['noun3']} {v} {a['unit3']} {cop('m')}।"
    return {                                             # F4
        "age": f"{e}ले {v} वर्ष पूरा {'गरेकी' if g == 'f' else 'गरेको'} {cop(g)}।",
        "height": f"नाप्दा {e}को उचाइ {v} सेन्टिमिटर आयो।",
        "weight": f"जोख्दा {e}को तौल {v} किलो आयो।",
        "obj_weight": f"जोख्दा {e}को तौल {v} किलो आयो।",
        "price": f"{e} {v} रुपैयाँमा पाइन्छ।",
    }[attr]


def relation_sentence(greater: str, smaller: str, attr: str, family: str, up: bool,
                      item: str | None = None) -> str:
    """One premise stating a comparison. `up=True`: the greater entity is the subject with the
    positive word ("हरि श्यामभन्दा जेठो छ"); `up=False`: the same fact from the other side
    ("श्याम हरिभन्दा कान्छो छ")."""
    s, t = (greater, smaller) if up else (smaller, greater)
    if attr == "quantity":
        w = MORE[family] if up else LESS[family]
        return {
            "F1": f"{s}सँग {t}भन्दा {w} {item} छन।",
            "F2": f"{s}सँग {item} {t}भन्दा {w} छन।",
            "F3": f"{t}को तुलनामा {s}सँग {w} {item} छन।",
            "F4": f"{t}भन्दा {s}सँग {w} {item} छन।",
        }[family]
    a = ATTRS[attr]
    g = GENDER[s]
    adj = agree(a["pos"] if up else a["neg"], g)
    return {
        "F1": f"{s} {t}भन्दा {adj} {cop(g)}।",
        "F2": f"{s}को {a['noun1']} {t}भन्दा {'बढी' if up else 'कम'} छ।",
        "F3": f"{t}को तुलनामा {s}को {a['noun3']} {'अधिक' if up else 'कम'} छ।",
        "F4": f"{t}भन्दा {s} {adj} {cop(g)}।",
    }[family]


def question_sentence(p: Problem, family: str) -> str:
    """The question, in the family's wording, for "max"/"min" over all entities or "more"/"less"
    over `p.pair`."""
    positive = p.question in ("max", "more")
    pair = p.question in ("more", "less")
    x, y = p.pair if pair else (None, None)
    if p.attr == "quantity":
        w = MORE[family] if positive else LESS[family]
        item = p.item
        if pair:
            return {
                "F1": f"{x} र {y}मध्ये कससँग {w} {item} छन?",
                "F2": f"{x} र {y}मध्ये कससँग {item} {w} छन?",
                "F3": f"{x} र {y}मा कससँग {w} {item} छन?",
                "F4": f"{x} कि {y}, दुईमध्ये {w} {item} कससँग छन?",
            }[family]
        return {
            "F1": f"सबैभन्दा {w} {item} कससँग छन?",
            "F2": f"कससँग {item} सबैभन्दा {w} छन?",
            "F3": f"कससँग सबैभन्दा {w} {item} छन?",
            "F4": f"यीमध्ये सबैभन्दा {w} {item} कससँग छन?",
        }[family]

    a = ATTRS[p.attr]
    base = a["pos"] if positive else a["neg"]
    amount2 = "बढी" if positive else "कम"
    amount3 = "अधिक" if positive else "कम"
    asked = [x, y] if pair else p.entities

    if p.kind == "object":
        adj = base
        if pair:
            return {
                "F1": f"{x} र {y}मध्ये कुन वस्तु {adj} छ?",
                "F2": f"{x} र {y}मध्ये कुन वस्तुको {a['noun1']} {amount2} छ?",
                "F3": f"{x} र {y}मा कुन वस्तुको {a['noun3']} {amount3} छ?",
                "F4": f"{x} कि {y}, दुईमध्ये {adj} कुन हो?",
            }[family]
        return {
            "F1": f"कुन वस्तु सबैभन्दा {adj} छ?",
            "F2": f"कुन वस्तुको {a['noun1']} सबैभन्दा {amount2} छ?",
            "F3": f"सबैभन्दा {amount3} {a['noun3']} कुन वस्तुको छ?",
            "F4": f"यीमध्ये सबैभन्दा {adj} वस्तु कुन हो?",
        }[family]

    adj = fem(base) if all(GENDER[e] == "f" for e in asked) else base
    if pair:
        return {
            "F1": f"{x} र {y}मध्ये को {adj} हो?",
            "F2": f"{x} र {y}मध्ये कसको {a['noun1']} {amount2} छ?",
            "F3": f"{x} र {y}मा कसको {a['noun3']} {amount3} छ?",
            "F4": f"{x} कि {y}, दुईमध्ये {adj} को हो?",
        }[family]
    return {
        "F1": f"सबैभन्दा {adj} को हो?",
        "F2": f"कसको {a['noun1']} सबैभन्दा {amount2} छ?",
        "F3": f"सबैभन्दा {amount3} {a['noun3']} कसको छ?",
        "F4": f"यीमध्ये सबैभन्दा {adj} को हो?",
    }[family]


# ------------------------------------------------------------------ whole prompt

def render(p: Problem, family: str, rng: random.Random) -> str:
    """The full single-line prompt body: premises, then the question. Not yet wrapped in PROMPT.

    Value premises follow `p.entities`; relation premises follow `p.relations` (already shuffled)
    and pick their direction independently, so neither order nor grammatical role gives the answer.
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
