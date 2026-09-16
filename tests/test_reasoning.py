"""Tests for Phase 3 -- the reasoning dataset, SFT machinery and scoring.

The dataset tests protect the property every later number depends on: **a label is correct and a test
split measures what its name says.** A template with a flipped comparative would teach the error
fluently (L9 slide 16); a name shared between train and test_entity would turn "unseen entities" into
seen ones; a heuristic that scores well above chance would mean the data leaks the answer by position.
None of these is visible in a loss curve, so each is asserted here.

The polarity test is deliberately independent of the templates: it reads each rendered relation with a
tiny language-specific parser (which entity is the standard of comparison, which polarity word is
present) and checks the facts it recovers against the facts that were rendered.
"""
import json
import random
from dataclasses import replace
from importlib import import_module

import pytest
import torch

from lma.model.transformer import GPT, GPTConfig
from lma.reasoning import build as B
from lma.reasoning.facts import EQUAL, TASKS, Problem, candidates, sample_problem, solve
from lma.reasoning.score import classify, few_shot_prompt, score_records, summarise
from lma.reasoning.sft import SFTConfig, finetune, set_dropout
from lma.reasoning.sft_data import IGNORE, SFTDataset, collate, encode_example
from lma.tokenizer.train_sp import Tokenizer
from lma.train.checkpoint import load_model

LANGS = ("hindi", "nepali")
SMALL = {"train": 600, "val": 60, "test_id": 60, "test_entity": 60, "test_wording": 60, "test_both": 60}


def mods(lang):
    return import_module(f"{lang}.reasoning.lexicon"), import_module(f"{lang}.reasoning.templates")


@pytest.fixture(scope="module", params=LANGS)
def built(request):
    lang = request.param
    lx, tp = mods(lang)
    cfg = B.BuildConfig(sizes=SMALL)
    data, pools = B.build_dataset(lx, tp, cfg)
    return lang, lx, tp, cfg, data, pools


@pytest.fixture(scope="module")
def tokenizers():
    return {lang: Tokenizer(f"{lang}/tokenizer/{lang}_unigram_16000.model") for lang in LANGS}


# ------------------------------------------------------------------ facts and solver

def test_solver_handles_every_task_and_question():
    rng = random.Random(0)
    pool = [f"e{i}" for i in range(20)]
    for task in TASKS:
        for _ in range(200):
            p = sample_problem(rng, task, "age", "person", pool, (1, 50))
            ans = solve(p)
            assert ans in candidates(p)


def test_chain_solution_does_not_depend_on_premise_order():
    p = Problem("ext4", "age", "person", ["a", "b", "c", "d"], "max",
                relations=[("c", "d"), ("a", "b"), ("b", "c")])
    shuffled = replace(p, relations=list(reversed(p.relations)))
    assert solve(p) == solve(shuffled) == "a"
    assert solve(replace(p, question="min")) == "d"


def test_pair_question_needs_the_transitive_step():
    p = Problem("pair3", "age", "person", ["x", "y", "z"], "more",
                relations=[("y", "z"), ("x", "y")], pair=("z", "x"))
    assert solve(p) == "x"
    assert solve(replace(p, question="less")) == "z"


def test_ties_give_equal_in_cmp2_and_at_the_extreme_in_cmp3():
    p2 = Problem("cmp2", "age", "person", ["a", "b"], "more", values={"a": 5, "b": 5}, pair=("a", "b"))
    assert solve(p2) == EQUAL
    p3 = Problem("cmp3", "age", "person", ["a", "b", "c"], "max", values={"a": 9, "b": 9, "c": 1})
    assert solve(p3) == EQUAL
    assert solve(replace(p3, question="min")) == "c"


def test_cmp3_ties_are_generated_only_at_the_asked_extreme():
    rng = random.Random(1)
    pool = [f"e{i}" for i in range(10)]
    ties = 0
    for _ in range(2000):
        p = sample_problem(rng, "cmp3", "age", "person", pool, (1, 99), tie_prob=0.25)
        vals = sorted(p.values.values(), reverse=p.question == "max")
        if len(set(vals)) < 3:
            ties += 1
            assert vals[0] == vals[1], "a tie must sit at the extreme the question asks about"
            assert solve(p) == EQUAL
    assert 400 < ties < 600


# ------------------------------------------------------------------ lexicons

@pytest.mark.parametrize("lang", LANGS)
def test_no_name_is_a_substring_of_another(lang):
    lx, _ = mods(lang)
    names = [n for n, _ in lx.PERSONS] + [n for n, _ in lx.OBJECTS]
    assert len(names) == len(set(names))
    clashes = [(a, b) for a in names for b in names if a != b and a in b]
    assert not clashes


@pytest.mark.parametrize("lang", LANGS)
def test_pools_are_large_enough_for_a_real_unseen_entity_split(lang):
    lx, _ = mods(lang)
    assert len(lx.PERSONS) >= 150 and len(lx.OBJECTS) >= 50
    assert all(" " not in n for n, _ in lx.PERSONS + lx.OBJECTS)


def test_person_names_are_not_shared_between_languages():
    hi, _ = mods("hindi")
    ne, _ = mods("nepali")
    assert not ({n for n, _ in hi.PERSONS} & {n for n, _ in ne.PERSONS})


def test_hindi_has_no_masculine_nouns_that_inflect_before_postpositions():
    lx, _ = mods("hindi")
    assert not [n for n, g in lx.PERSONS + lx.OBJECTS if g == "m" and n.endswith("ा")]


# ------------------------------------------------------------------ dataset

def test_every_label_agrees_with_the_independent_solver(built):
    _, _, tp, _, data, _ = built
    for recs in data.values():
        for r in recs:
            p = B.problem_from_record(r)
            assert tp.answer_text(solve(p)) == r["answer"]
            assert r["answer"] in r["candidates"]


def test_splits_are_disjoint_by_prompt_fact_pool_and_family(built):
    _, _, tp, _, data, pools = built
    lk = B.leakage_report(data, pools)
    assert sum(lk["prompt_overlap"].values()) == 0
    assert sum(lk["fact_overlap"].values()) == 0
    assert lk["pool_overlap"] == 0
    assert lk["unseen_names_in_seen_splits"] == 0 and lk["seen_names_in_unseen_splits"] == 0
    assert lk["family_overlap"] == []
    assert {r["family"] for r in data["test_wording"]} == {tp.HELDOUT_FAMILY}
    assert tp.HELDOUT_FAMILY not in {r["family"] for r in data["train"]}


def test_every_task_attribute_and_family_is_covered_in_train(built):
    _, _, tp, _, data, _ = built
    seen = {(r["task"], r["attr"], r["family"]) for r in data["train"]}
    assert len(seen) == len(TASKS) * len(B.ATTRIBUTES) * len(tp.TRAIN_FAMILIES)


def test_heuristic_baselines_are_not_shortcuts(built):
    _, _, _, _, data, _ = built
    recs = data["train"]
    acc = B.heuristic_accuracy(recs)
    chance = B.chance_accuracy(recs)["all"]
    for h, per in acc.items():
        assert per["all"] < chance + 0.08, f"{h} scores {per['all']:.2f} vs chance {chance:.2f}"


def test_prompts_are_single_line_normalised_and_round_trip(built, tokenizers):
    lang, _, _, _, data, _ = built
    norm = import_module(f"{lang}.clean.norm").norm
    tok = tokenizers[lang]
    for r in data["val"] + data["test_both"]:
        assert "\n" not in r["prompt"]
        assert norm(r["prompt"]) == r["prompt"]
        assert tok.decode(tok.encode(r["prompt"])) == r["prompt"]
        assert tok.unk_rate(r["prompt"]) == 0.0


def test_digit_script_follows_each_language(built):
    lang, _, _, _, data, _ = built
    text = " ".join(r["prompt"] for r in data["train"] if r["task"] in ("cmp2", "cmp3"))
    ascii_digits = sum(c.isascii() and c.isdigit() for c in text)
    deva_digits = sum("०" <= c <= "९" for c in text)
    if lang == "hindi":
        assert ascii_digits > 0 and deva_digits == 0
    else:
        assert deva_digits > 0 and ascii_digits == 0


def test_generation_is_byte_reproducible(tmp_path):
    lx, tp = mods("nepali")
    sizes = {k: 30 for k in SMALL}
    a, _ = B.build_dataset(lx, tp, B.BuildConfig(sizes=sizes))
    b, _ = B.build_dataset(lx, tp, B.BuildConfig(sizes=sizes))
    B.write_jsonl(a["train"], tmp_path / "a.jsonl")
    B.write_jsonl(b["train"], tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


# ------------------------------------------------------------------ polarity (template-independent)

def _read_relation(lang, tp, sentence, names):
    """(greater, smaller) recovered from one rendered relation sentence by a minimal parser."""
    words = [w.strip("।,?") for w in sentence.split()]
    present = [n for n in names if n in sentence]
    assert len(present) == 2, sentence
    standard = None
    if lang == "hindi":
        for marker in tp.STANDARD_MARKERS:
            for i in range(1, len(words) - len(marker) + 1):
                if tuple(words[i:i + len(marker)]) == marker and words[i - 1] in present:
                    standard = words[i - 1]
    else:
        for i, w in enumerate(words):
            if w.endswith(tp.STANDARD_SUFFIX) and w[:-len(tp.STANDARD_SUFFIX)] in present:
                standard = w[:-len(tp.STANDARD_SUFFIX)]
            if w == tp.COMPARE_WORD and i > 0 and words[i - 1].endswith("को"):
                standard = words[i - 1][:-2]
    assert standard in present, sentence
    subject = next(n for n in present if n != standard)
    rest = [w for w in words if w not in present and not any(w.startswith(n) for n in present)]
    pos = any(w in tp.POSITIVE_WORDS for w in rest)
    neg = any(w in tp.NEGATIVE_WORDS for w in rest)
    assert pos != neg, f"exactly one polarity word expected: {sentence}"
    return (subject, standard) if pos else (standard, subject)


@pytest.mark.parametrize("lang", LANGS)
def test_every_relation_template_states_the_fact_it_was_given(lang):
    lx, tp = mods(lang)
    people = [n for n, _ in lx.PERSONS]
    objects = [n for n, _ in lx.OBJECTS]
    rng = random.Random(3)
    for attr, (kind, _) in B.ATTRIBUTES.items():
        pool = people if kind == "person" else objects
        for family in tp.FAMILIES:
            for up in (True, False):
                for _ in range(10):
                    g, s = rng.sample(pool, 2)
                    sent = tp.relation_sentence(g, s, attr, family, up, rng.choice(lx.ITEMS))
                    assert _read_relation(lang, tp, sent, [g, s]) == (g, s), sent


@pytest.mark.parametrize("lang", LANGS)
def test_question_polarity_matches_the_question(lang):
    lx, tp = mods(lang)
    rng = random.Random(4)
    people = [n for n, _ in lx.PERSONS]
    for attr, (kind, rng_v) in B.ATTRIBUTES.items():
        pool = people if kind == "person" else [n for n, _ in lx.OBJECTS]
        for task in TASKS:
            p = sample_problem(rng, task, attr, kind, pool, rng_v, item=lx.ITEMS[0])
            for family in tp.FAMILIES:
                q = tp.question_sentence(p, family)
                words = {w.strip("?,।") for w in q.split()}
                positive = p.question in ("max", "more")
                assert bool(words & tp.POSITIVE_WORDS) == positive, q
                assert bool(words & tp.NEGATIVE_WORDS) == (not positive), q


# ------------------------------------------------------------------ SFT data

TINY = GPTConfig(vocab_size=16000, d_model=32, n_layers=2, n_heads=4, mlp_hidden=64, max_seq_len=128)


def test_loss_mask_covers_exactly_the_answer_and_eos(tokenizers):
    tok = tokenizers["hindi"]
    e = encode_example(tok, "प्रश्न: राहुल, अमित से बड़ा है। सबसे बड़ा कौन है? उत्तर:", "राहुल")
    answer_ids = tok.encode("राहुल") + [tok._sp.eos_id()]
    supervised = [l for l in e.labels if l != IGNORE]
    assert supervised == answer_ids
    assert len(e.input_ids) == len(e.labels)
    assert e.input_ids[0] == tok._sp.eos_id()
    # The first supervised label is predicted from the last prompt token.
    assert e.labels[e.prompt_len - 2] == IGNORE and e.labels[e.prompt_len - 1] == answer_ids[0]


def test_separate_encoding_decodes_to_prompt_space_answer(tokenizers):
    for lang, tok in tokenizers.items():
        prompt, answer = "प्रश्न: क उत्तर:", "बराबर"
        ids = tok.encode(prompt) + tok.encode(answer)
        assert tok.decode(ids) == f"{prompt} {answer}"


def test_collate_right_pads_with_ignored_labels(tokenizers):
    tok = tokenizers["hindi"]
    a = encode_example(tok, "प्रश्न: क उत्तर:", "राहुल")
    b = encode_example(tok, "प्रश्न: एक लंबा प्रश्न यहां है उत्तर:", "सीता")
    x, y = collate([a, b])
    assert x.shape == y.shape
    n = len(a.input_ids)
    assert (x[0, n:] == 0).all() and (y[0, n:] == IGNORE).all()


# ------------------------------------------------------------------ SFT loop

def _tiny_records(n=50):
    lx, tp = mods("hindi")
    data, _ = B.build_dataset(lx, tp, B.BuildConfig(sizes={**{k: 10 for k in SMALL}, "train": n}))
    return data


def test_sft_loss_falls_and_the_run_resumes(tmp_path, tokenizers):
    tok = tokenizers["hindi"]
    data = _tiny_records()
    torch.manual_seed(0)
    model = GPT(TINY)
    cfg = SFTConfig(lr=3e-3, batch_size=16, max_steps=30, eval_every=15, val_em_examples=8,
                    log_every=1, dropout=0.0, patience=100, warmup_fraction=0.1)
    ds = SFTDataset(data["train"], tok)
    x, y = ds.batch(0, 50)
    before = float(model(x, y)[1])
    finetune(model, tok, data["train"], data["val"], cfg, tmp_path / "ck", tmp_path / "lg", torch.device("cpu"))
    after = float(model(x, y)[1])
    assert after < before * 0.8

    resumed = GPT(TINY)
    summary = finetune(resumed, tok, data["train"], data["val"], replace(cfg, max_steps=40),
                       tmp_path / "ck", tmp_path / "lg", torch.device("cpu"))
    assert summary["steps"] == 40
    events = [json.loads(l) for l in (tmp_path / "lg" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e.get("kind") == "resume" and e["from_step"] == 30 for e in events)
    assert len({e["run_id"] for e in events}) == 2, "the resumed session must append, not replace"
    model2, state, extra = load_model(tmp_path / "ck" / "ckpt_best.pt")
    assert extra["sft_config"]["lr"] == 3e-3 and state.step > 0


def test_set_dropout_reaches_every_site():
    model = GPT(TINY)
    set_dropout(model, 0.1)
    assert all(m.p == 0.1 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert all(b.attn.dropout == 0.1 for b in model.blocks)


# ------------------------------------------------------------------ scoring

def _record():
    p = Problem("ext3", "age", "person", ["राहुल", "अमित", "सीता"], "max",
                relations=[("राहुल", "अमित"), ("अमित", "सीता")])
    return {"id": "x", "task": "ext3", "attr": "age", "family": "F1", "question": "max",
            "prompt": "प्रश्न: … उत्तर:", "answer": "राहुल", "answer_key": "राहुल",
            "candidates": ["राहुल", "अमित", "सीता"], "entities": ["राहुल", "अमित", "सीता"],
            "facts": p.to_dict()}


@pytest.mark.parametrize("text,stopped,verdict", [
    ("राहुल", True, "correct"),
    ("राहुल है", False, "no_stop"),
    ("सीता", True, "reversed_extreme"),
    ("अमित", True, "middle_entity"),
    ("गीता", True, "out_of_prompt_name"),
    ("है।", True, "format_failure"),
    ("बराबर", True, "equal_case_miss"),
])
def test_error_taxonomy(text, stopped, verdict):
    assert classify(_record(), {"text": text, "stopped": stopped}, "बराबर", {"गीता", "राहुल", "अमित", "सीता"}) == verdict


def test_exact_match_and_candidate_choice_scoring():
    r = _record()
    rows = score_records([r, r], [{"text": "राहुल", "stopped": True}, {"text": "राहुल और", "stopped": False}],
                         [[-1.0, -2.0, -3.0], [-5.0, -1.0, -3.0]], "बराबर", set(r["entities"]))
    assert [x["em"] for x in rows] == [1, 0]
    assert [x["prefix"] for x in rows] == [1, 1]
    assert [x["cand_acc"] for x in rows] == [1, 0]
    s = summarise(rows)
    assert s["all"]["em"] == 0.5 and s["all"]["cand_acc"] == 0.5 and s["errors"]["no_stop"] == 1


def test_few_shot_prompt_ends_with_the_test_prompt():
    r = _record()
    demo = {**r, "prompt": "प्रश्न: डेमो उत्तर:", "answer": "अमित"}
    fs = few_shot_prompt(r, [demo, demo, demo])
    assert fs.endswith(r["prompt"]) and fs.count("उत्तर:") == 4 and "\n" not in fs

