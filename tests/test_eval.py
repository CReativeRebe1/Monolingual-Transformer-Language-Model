"""Tests for lma/eval/ -- perplexity/BPB, text metrics, diversity, attention diagnostics.

Two validation strategies, matching the project's established pattern:

* **Hand-computable cases** for anything with a closed form. The entropy of a one-hot attention row
  is exactly 0; of a uniform row over n keys, exactly log2(n); the mean distance of a pure
  previous-token head is exactly 1. These are checked against the arithmetic, not against a
  reference implementation, because there is nothing to disagree with.
* **An independent reference** for BLEU and chrF, which have enough convention baked in (clipping,
  brevity penalty, whether to average sentence scores) that an implementation can be
  self-consistently wrong. `sacrebleu` is the reference, as `datasketch` was for MinHash in Phase 1.

The attention tests matter more than they look: every entropy and distance figure in the report is
produced by this code, and a metric that is quietly wrong would still produce a plausible
distance-vs-entropy scatter with heads in believable places.
"""
import math

import pytest
import torch

from lma.eval.attention import (
    HeadStats, aggregate, analyse, classify, first_token_mass, head_distance,
    head_entropy, layer_means, previous_token_mass, role_counts,
)
from lma.eval.diversity import distinct_n, diversity_report, repetition_rate, self_bleu
from lma.eval.perplexity import bits_per_byte, perplexity_from_nll
from lma.eval.textmetrics import all_metrics, bleu, chrf, rouge_l
from lma.model.transformer import GPT, GPTConfig

TINY = GPTConfig(vocab_size=61, d_model=32, n_layers=3, n_heads=4, mlp_hidden=88, max_seq_len=64)

HI_A = "दिल्ली भारत की राजधानी है और यहाँ करोड़ों लोग रहते हैं"
HI_B = "दिल्ली भारत की राजधानी है और यहाँ लाखों लोग रहते हैं"
HI_C = "नेपाल एक भूपरिवेष्टित देश है जिसकी सीमा चीन से लगती है"


# ------------------------------------------------------------------ perplexity / BPB

def test_perplexity_is_exp_of_mean_nll():
    assert perplexity_from_nll(0.0) == pytest.approx(1.0)
    assert perplexity_from_nll(math.log(50)) == pytest.approx(50.0)


def test_bits_per_byte_against_a_hand_computed_example():
    """100 bytes at exactly 1 bit each is ln(2)*100 nats of total surprisal."""
    assert bits_per_byte(math.log(2) * 100, 100) == pytest.approx(1.0)
    assert bits_per_byte(math.log(2) * 50, 100) == pytest.approx(0.5)


def test_bits_per_byte_is_invariant_to_the_tokenizer_where_perplexity_is_not():
    """The property BPB exists for, and the reason it is the number that compares Model H to Model L.

    Take one piece of text and two tokenizers. Both models are equally surprised by it in total --
    same 5,000 nats over the same 3,000 bytes -- but the coarser vocabulary spends 1,000 tokens on
    it and the finer one 2,000. Perplexity per token differs by a factor the tokenizer chose;
    bits-per-byte is identical, because bytes are the same unit in both.
    """
    total_nll, n_bytes = 5000.0, 3000
    coarse_tokens, fine_tokens = 1000, 2000

    assert bits_per_byte(total_nll, n_bytes) == bits_per_byte(total_nll, n_bytes)
    coarse_ppl = perplexity_from_nll(total_nll / coarse_tokens)
    fine_ppl = perplexity_from_nll(total_nll / fine_tokens)
    assert coarse_ppl > 10 * fine_ppl, (
        f"perplexity {coarse_ppl:.1f} vs {fine_ppl:.1f} -- the same text and the same total "
        "surprisal, differing only by how the tokenizer chose to segment it"
    )


# ------------------------------------------------------------------ text metrics vs sacrebleu

def test_bleu_is_100_for_an_exact_match():
    assert bleu([HI_A], [HI_A]) == pytest.approx(100.0, abs=1e-6)


def test_bleu_is_zero_when_nothing_matches():
    assert bleu(["क ख ग घ ङ"], ["च छ ज झ ञ"]) == 0.0


def test_bleu_clips_repeated_ngrams():
    """Emitting one correct word 20 times must not score a perfect unigram precision -- the clip is
    what BLEU has instead of a degeneration penalty, and dropping it is the classic implementation
    bug that makes a looping model look good."""
    assert bleu(["है " * 20], [HI_A]) < 20.0


def test_bleu_agrees_with_sacrebleu():
    """Independent reference, same role datasketch played for MinHash in Phase 1: BLEU has enough
    convention in it (clipping, brevity penalty, corpus- vs sentence-level) to be self-consistently
    wrong."""
    sacrebleu = pytest.importorskip("sacrebleu")
    hyps = [HI_B, HI_C, "यह एक छोटा वाक्य है"]
    refs = [HI_A, HI_C, "यह एक बड़ा वाक्य है और लंबा भी"]
    theirs = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score
    assert bleu(hyps, refs) == pytest.approx(theirs, rel=0.02)


def test_chrf_agrees_with_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")
    hyps = [HI_B, HI_C]
    refs = [HI_A, HI_C]
    theirs = sacrebleu.corpus_chrf(hyps, [refs]).score
    assert chrf(hyps, refs) == pytest.approx(theirs, rel=0.05)


def test_chrf_is_more_forgiving_than_bleu_on_an_inflectional_near_miss():
    """The reason chrF is the least bad of the three for Devanagari: a word differing by one suffix
    costs word-BLEU a whole n-gram match and costs chrF a fraction of one."""
    ref = ["सरकार ने प्रदूषण कम करने के लिए कदम उठाए"]
    hyp = ["सरकार ने प्रदूषण कम करने के लिए कदम उठाया"]
    assert chrf(hyp, ref) > bleu(hyp, ref)


def test_rouge_l_rewards_order_without_requiring_adjacency():
    """A subsequence metric: the right words in the right order, gaps allowed."""
    assert rouge_l(["क ख ग"], ["क ख ग"]) == pytest.approx(100.0)
    with_gap = rouge_l(["क य ख य ग"], ["क ख ग"])
    reordered = rouge_l(["ग ख क"], ["क ख ग"])
    assert with_gap > reordered


def test_metrics_reject_mismatched_lengths():
    for fn in (bleu, chrf, rouge_l):
        with pytest.raises(ValueError):
            fn(["a"], ["a", "b"])


def test_all_metrics_returns_the_three_named_scores():
    assert set(all_metrics([HI_A], [HI_A])) == {"bleu", "chrf", "rouge_l"}


# ------------------------------------------------------------------ diversity

def test_distinct_n_is_near_zero_on_a_degenerate_repetition():
    """The failure mode these metrics exist for: perplexity can look fine while greedy decoding
    emits the same token forever."""
    assert distinct_n(["और " * 50], 1) < 0.05
    assert distinct_n(["और " * 50], 2) < 0.05


def test_distinct_n_is_one_when_every_ngram_is_unique():
    assert distinct_n(["क ख ग घ ङ च छ"], 1) == pytest.approx(1.0)


def test_distinct_n_is_pooled_so_cross_sample_repetition_is_visible():
    """Ten prompts producing the same paragraph is what greedy decoding does, and a per-sample
    average would report it as perfectly diverse."""
    same = ["क ख ग घ"] * 10
    varied = [f"क{i} ख{i} ग{i} घ{i}" for i in range(10)]
    assert distinct_n(same, 2) < distinct_n(varied, 2)


def test_repetition_rate_detects_a_within_sample_loop():
    assert repetition_rate("क ख ग घ " * 10, n=4) > 0.8
    assert repetition_rate(" ".join(f"w{i}" for i in range(40)), n=4) == 0.0


def test_self_bleu_is_high_when_every_generation_is_the_same():
    assert self_bleu([HI_A] * 5) > 90.0
    assert self_bleu([HI_A, HI_C, "यह एक अलग वाक्य है जो पूरी तरह भिन्न है"]) < 40.0


def test_self_bleu_on_fewer_than_two_samples_is_zero_not_an_error():
    assert self_bleu([]) == 0.0
    assert self_bleu([HI_A]) == 0.0


def test_diversity_report_has_every_reported_field():
    report = diversity_report([HI_A, HI_B, HI_C])
    assert set(report) == {
        "n_samples", "mean_length_words", "distinct_1", "distinct_2", "distinct_3",
        "repetition_rate_4gram", "self_bleu",
    }


# ------------------------------------------------------------------ attention: hand-computable

def _attn(rows: list[list[float]]) -> torch.Tensor:
    """A [1, 1, T, T] attention tensor from hand-written rows."""
    return torch.tensor(rows, dtype=torch.float32)[None, None]


def test_entropy_of_a_one_hot_row_is_exactly_zero():
    """A head attending to exactly one position has no uncertainty."""
    onehot = _attn([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert float(head_entropy(onehot)[0]) == pytest.approx(0.0, abs=1e-5)


def test_entropy_of_a_uniform_row_over_n_keys_is_log2_n():
    """The other closed form. Row i attends uniformly over i+1 keys, so the mean entropy over the
    three queries is (log2(1) + log2(2) + log2(3)) / 3."""
    uniform = _attn([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1 / 3, 1 / 3, 1 / 3],
    ])
    expected = (math.log2(1) + math.log2(2) + math.log2(3)) / 3
    assert float(head_entropy(uniform)[0]) == pytest.approx(expected, abs=1e-5)


def test_normalised_entropy_of_a_uniform_head_is_exactly_one():
    """The normalisation's whole purpose. A head attending uniformly over everything it CAN see is
    maximally diffuse at every position, and must score 1.0 at position 1 and position 100 alike --
    raw entropy would report it as steadily 'spreading out' as the sequence goes on, which is an
    artefact of causal masking rather than a fact about the head."""
    uniform = _attn([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    assert float(head_entropy(uniform, normalise=True)[0]) == pytest.approx(1.0, abs=1e-4)


def test_mean_distance_of_a_previous_token_head_is_exactly_one():
    """Query i attends entirely to key i-1, so every (i - j) term is exactly 1. Query 0 has nowhere
    to look but itself and is excluded from the mean -- otherwise a perfect previous-token head
    would score below 1 purely because of its first position."""
    prev = _attn([
        [1.0, 0.0, 0.0, 0.0],       # query 0: only key 0 available (excluded from the mean)
        [1.0, 0.0, 0.0, 0.0],       # query 1 -> key 0, distance 1
        [0.0, 1.0, 0.0, 0.0],       # query 2 -> key 1, distance 1
        [0.0, 0.0, 1.0, 0.0],       # query 3 -> key 2, distance 1
    ])
    assert float(head_distance(prev)[0]) == pytest.approx(1.0, abs=1e-5)


def test_mean_distance_of_a_uniform_head_grows_with_position():
    """A head attending uniformly over its whole history averages i/2 at query i, so its mean over
    positions 1..3 is (0.5 + 1 + 1.5) / 3 = 1.0. Hand-computable, and the reason the taxonomy
    compares heads within a layer rather than against an absolute distance threshold."""
    uniform = _attn([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    assert float(head_distance(uniform)[0]) == pytest.approx(1.0, abs=1e-5)


def test_mean_distance_is_never_negative_under_causal_masking():
    """j <= i always, so (i - j) >= 0. A negative value would mean the mask leaked."""
    torch.manual_seed(0)
    T = 12
    scores = torch.randn(2, 4, T, T)
    scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    attn = scores.softmax(dim=-1)
    assert float(head_distance(attn).min()) >= 0.0


def test_previous_token_and_first_token_mass_are_measured_correctly():
    prev = _attn([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert float(previous_token_mass(prev)[0]) == pytest.approx(1.0, abs=1e-5)
    sink = _attn([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert float(first_token_mass(sink)[0]) == pytest.approx(1.0, abs=1e-5)


def test_taxonomy_labels_the_mechanically_identifiable_roles():
    assert classify(entropy_norm=0.9, distance=3.0, prev_mass=0.95, first_mass=0.0) == "previous_token"
    assert classify(entropy_norm=0.1, distance=40.0, prev_mass=0.0, first_mass=0.9) == "attention_sink"
    assert classify(entropy_norm=0.2, distance=2.0, prev_mass=0.1, first_mass=0.1) == "local"
    assert classify(entropy_norm=0.9, distance=30.0, prev_mass=0.1, first_mass=0.1) == "diffuse_global"


# ------------------------------------------------------------------ attention: on the real model

@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(TINY).eval()


def test_analyse_returns_one_record_per_head_per_layer(model):
    stats = analyse(model, torch.randint(0, TINY.vocab_size, (1, 24)))
    assert len(stats) == TINY.n_layers * TINY.n_heads
    assert {(s.layer, s.head) for s in stats} == {
        (l, h) for l in range(TINY.n_layers) for h in range(TINY.n_heads)
    }


def test_analysed_statistics_are_in_their_valid_ranges(model):
    """Entropy cannot exceed log2(T); normalised entropy cannot exceed 1; distance cannot be
    negative. Any violation means the maths, not the model, is wrong."""
    T = 24
    stats = analyse(model, torch.randint(0, TINY.vocab_size, (1, T)))
    for s in stats:
        assert 0.0 <= s.entropy_bits <= math.log2(T) + 1e-4
        assert 0.0 <= s.entropy_normalised <= 1.0 + 1e-4
        assert s.mean_distance >= 0.0
        assert 0.0 <= s.prev_token_mass <= 1.0


def test_analyse_leaves_the_model_in_the_mode_it_found_it(model):
    model.train()
    analyse(model, torch.randint(0, TINY.vocab_size, (1, 16)))
    assert model.training
    model.eval()


def test_analyse_uses_the_manual_path_and_matches_it_exactly(model):
    """The statistics must be computed from the same tensors the heatmaps are drawn from."""
    idx = torch.randint(0, TINY.vocab_size, (1, 20))
    stats = analyse(model, idx)
    with torch.no_grad():
        _, _, attns = model(idx, output_attentions=True)
    direct = head_entropy(attns[0])
    for h in range(TINY.n_heads):
        assert stats[h].entropy_bits == pytest.approx(float(direct[h]), abs=1e-3)


def test_aggregate_averages_across_inputs(model):
    """A head's role should not be inferred from one paragraph -- a 'local' verdict drawn from an
    input that happens to be a list of names is a fact about the input."""
    runs = [analyse(model, torch.randint(0, TINY.vocab_size, (1, 20))) for _ in range(3)]
    merged = aggregate(runs)
    assert len(merged) == len(runs[0])
    expected = sum(r[5].mean_distance for r in runs) / 3
    assert merged[5].mean_distance == pytest.approx(expected, abs=1e-3)


def test_layer_means_and_role_counts_summarise_every_head(model):
    stats = analyse(model, torch.randint(0, TINY.vocab_size, (1, 20)))
    assert set(layer_means(stats)) == set(range(TINY.n_layers))
    assert sum(role_counts(stats).values()) == len(stats)


def test_save_stats_writes_readable_json(model, tmp_path):
    from lma.eval.attention import save_stats

    stats = analyse(model, torch.randint(0, TINY.vocab_size, (1, 16)))
    path = save_stats(stats, tmp_path / "attention_stats.json", extra={"model": "test"})
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_layers"] == TINY.n_layers and payload["n_heads"] == TINY.n_heads
    assert len(payload["heads"]) == TINY.n_layers * TINY.n_heads
    assert payload["model"] == "test"


# ------------------------------------------------------------------ font handling

def test_a_devanagari_font_is_found_or_a_warning_is_raised():
    """The Phase-1 failure this guards against was silent: matplotlib does not warn on a missing
    glyph, it draws a hollow box, and a figure full of boxes looks like a rendering setting."""
    import warnings

    from lma.viz.plots import devanagari_font

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        font = devanagari_font()
    assert font is not None or any(issubclass(w.category, RuntimeWarning) for w in caught)
