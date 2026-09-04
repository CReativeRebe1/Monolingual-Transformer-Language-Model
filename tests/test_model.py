"""Tests for lma/model/ -- the from-scratch decoder-only Transformer.

Three groups of tests, and the middle group is the one that earns its keep:

* **Structure** -- parameter counts, tied embeddings, shapes. These catch a budget break, which is
  otherwise invisible: a model 6M parameters over budget trains and produces a perfectly normal
  loss curve.
* **Correctness of the maths** -- causality, RoPE's relative-distance property, and the equivalence
  of the fused (SDPA) and explicit attention paths. A causal-masking bug is the classic silent
  failure in a from-scratch LM: the model sees the future, the loss drops beautifully, and the
  result is meaningless. The test used here is behavioural rather than structural -- change a LATER
  token and assert an EARLIER token's logits do not move -- which would catch an off-by-one in the
  mask that inspecting the mask tensor itself would not.
* **Generation** -- determinism under a seed, greedy determinism without one, and EOS handling.

The manual-vs-SDPA equivalence test is what licenses every attention figure in the report: the
figures are produced by the explicit path, while the trained weights come from the fused path. If
the two ever computed different functions, the analysis would describe a model that was never
trained, and nothing else in the pipeline would notice.
"""
import math

import pytest
import torch

from lma.model.attention import CausalSelfAttention
from lma.model.block import RMSNorm, SwiGLU
from lma.model.rope import apply_rope, rope_cache
from lma.model.transformer import GPT, GPTConfig, analytic_param_count

torch.manual_seed(0)

SMALL = GPTConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=4, mlp_hidden=88, max_seq_len=16)


@pytest.fixture(scope="module")
def model():
    return GPT(SMALL).eval()


# ------------------------------------------------------------------ RoPE

def test_rope_makes_the_dot_product_depend_only_on_relative_distance():
    """The property RoPE exists for. If this fails, positions are being encoded absolutely and the
    rotation is doing something other than what the paper describes."""
    head_dim, T = 16, 12
    cos, sin = rope_cache(T, head_dim)
    q = torch.randn(1, 1, T, head_dim)
    k = torch.randn(1, 1, T, head_dim)
    # Same content vector placed at different absolute positions, same relative gap.
    qr = apply_rope(q[:, :, :1].expand(1, 1, T, head_dim).contiguous(), cos, sin)
    kr = apply_rope(k[:, :, :1].expand(1, 1, T, head_dim).contiguous(), cos, sin)
    gap = 3
    dots = [float(qr[0, 0, m] @ kr[0, 0, m - gap]) for m in range(gap, T)]
    assert max(dots) - min(dots) < 1e-4, f"dot product varies with absolute position: {dots}"


def test_rope_at_position_zero_is_the_identity():
    cos, sin = rope_cache(4, 8)
    x = torch.randn(1, 1, 4, 8)
    assert torch.allclose(apply_rope(x, cos, sin)[:, :, 0], x[:, :, 0], atol=1e-6)


def test_rope_preserves_vector_norm():
    """A rotation cannot change length. If it does, the maths is not a rotation."""
    cos, sin = rope_cache(9, 16)
    x = torch.randn(2, 3, 9, 16)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_rejects_an_odd_head_dim():
    with pytest.raises(ValueError):
        rope_cache(4, 7)


# ------------------------------------------------------------------ attention: the two paths

def test_manual_and_fused_attention_paths_agree():
    """The test that licenses the attention analysis.

    Training runs through F.scaled_dot_product_attention; every attention figure comes from the
    explicit softmax(QK^T/sqrt(d_k) + M)V path. A silent divergence -- a missing scale, a mask off
    by one row -- would mean the report describes a model that was never trained, and the heatmaps
    would still look like plausible attention.
    """
    torch.manual_seed(7)
    attn = CausalSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(2, 11, 32)
    cos, sin = rope_cache(11, 8)
    fused, none_weights = attn(x, cos, sin, output_attentions=False)
    manual, weights = attn(x, cos, sin, output_attentions=True)
    assert none_weights is None
    assert weights is not None
    assert torch.allclose(fused, manual, atol=1e-5), (
        f"max |fused - manual| = {(fused - manual).abs().max():.2e}"
    )


def test_attention_weights_are_a_causal_distribution():
    attn = CausalSelfAttention(d_model=32, n_heads=4).eval()
    T = 9
    cos, sin = rope_cache(T, 8)
    _, w = attn(torch.randn(1, T, 32), cos, sin, output_attentions=True)
    assert w.shape == (1, 4, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(1, 4, T), atol=1e-6)   # rows are distributions
    assert float(w.triu(1).abs().max()) == 0.0                          # nothing above the diagonal


def test_attention_rejects_a_head_count_that_does_not_divide_d_model():
    with pytest.raises(ValueError):
        CausalSelfAttention(d_model=32, n_heads=5)


# ------------------------------------------------------------------ causality, end to end

def test_a_later_token_cannot_change_an_earlier_tokens_logits(model):
    """The real test of the causal mask -- behavioural, not structural.

    Inspecting the mask tensor proves the mask exists; this proves it is applied in the right
    orientation and with the right offset. A model that can see one token into the future still
    trains to a low, entirely believable loss.
    """
    idx = torch.randint(0, SMALL.vocab_size, (1, 10))
    with torch.no_grad():
        before, _, _ = model(idx)
        changed = idx.clone()
        changed[0, 7] = (changed[0, 7] + 1) % SMALL.vocab_size
        after, _, _ = model(changed)
    assert torch.allclose(before[0, :7], after[0, :7], atol=1e-6), "position 7 leaked backwards"
    assert not torch.allclose(before[0, 7], after[0, 7], atol=1e-6), "position 7 had no effect at all"


def test_causality_holds_on_the_manual_path_too(model):
    idx = torch.randint(0, SMALL.vocab_size, (1, 10))
    with torch.no_grad():
        before, _, _ = model(idx, output_attentions=True)
        changed = idx.clone()
        changed[0, 6] = (changed[0, 6] + 1) % SMALL.vocab_size
        after, _, _ = model(changed, output_attentions=True)
    assert torch.allclose(before[0, :6], after[0, :6], atol=1e-6)


# ------------------------------------------------------------------ shapes, loss, gradients

def test_forward_shapes_and_loss(model):
    idx = torch.randint(0, SMALL.vocab_size, (3, 12))
    logits, loss, attns = model(idx, targets=idx)
    assert logits.shape == (3, 12, SMALL.vocab_size)
    assert attns is None
    assert loss.ndim == 0 and float(loss) > 0


def test_untrained_loss_is_near_ln_vocab_size():
    """A freshly initialised LM predicts uniformly, so cross-entropy should start at ln(V).

    Catches a broken initialisation or a mis-shaped logit tensor before any training time is spent:
    a run that starts at 2.0 instead of 9.2 is not learning fast, it is reading its own targets.

    Targets must be genuinely NEXT-token here. Written the lazy way (`targets=idx`, i.e. predict the
    current token) this test measured 3.98 against ln(97)=4.57 on an untrained model -- not a bug but
    a real property of weight tying: the residual stream still carries emb(idx[i]) at position i, and
    with a tied head logits[i, idx[i]] is that vector's dot product with itself, so the identity
    mapping is already partly learned at initialisation. Worth knowing when reading the first steps
    of a real loss curve.
    """
    m = GPT(SMALL).eval()
    idx = torch.randint(0, SMALL.vocab_size, (8, 17))
    _, loss, _ = m(idx[:, :-1], targets=idx[:, 1:])
    assert float(loss) == pytest.approx(math.log(SMALL.vocab_size), abs=0.25)


def test_weight_tying_makes_the_identity_mapping_partly_free_at_init():
    """The flip side of the test above, asserted directly so the effect is recorded rather than
    rediscovered: predicting the CURRENT token starts well below ln(V) with tied embeddings, and at
    ln(V) without them."""
    idx = torch.randint(0, SMALL.vocab_size, (8, 16))
    tied = GPT(SMALL).eval()
    untied = GPT(GPTConfig(**{**SMALL.to_dict(), "tie_embeddings": False})).eval()
    _, tied_loss, _ = tied(idx, targets=idx)
    _, untied_loss, _ = untied(idx, targets=idx)
    assert float(tied_loss) < float(untied_loss) - 0.2
    assert float(untied_loss) == pytest.approx(math.log(SMALL.vocab_size), abs=0.25)


def test_attentions_are_returned_per_layer(model):
    _, _, attns = model(torch.randint(0, SMALL.vocab_size, (2, 7)), output_attentions=True)
    assert len(attns) == SMALL.n_layers
    assert all(a.shape == (2, SMALL.n_heads, 7, 7) for a in attns)


def test_forward_backward_runs_on_cpu():
    m = GPT(SMALL)
    idx = torch.randint(0, SMALL.vocab_size, (2, 8))
    _, loss, _ = m(idx, targets=idx)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_sequence_longer_than_context_is_rejected(model):
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.zeros(1, SMALL.max_seq_len + 1, dtype=torch.long))


# ------------------------------------------------------------------ parameter budget

def test_analytic_param_count_matches_the_real_module():
    """The arithmetic in analytic_param_count() is what the language configs assert their budget
    against, and what the report's parameter table prints. It is only trustworthy if it agrees with
    an actually-constructed model."""
    m = GPT(SMALL)
    total, non_embedding = m.param_count()
    expected = analytic_param_count(SMALL)
    assert total == expected["total"]
    assert non_embedding == expected["non_embedding"]


def test_tied_embeddings_share_storage_and_are_counted_once():
    m = GPT(SMALL)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    untied = GPT(GPTConfig(**{**SMALL.to_dict(), "tie_embeddings": False}))
    assert untied.param_count()[0] - m.param_count()[0] == SMALL.vocab_size * SMALL.d_model


@pytest.mark.parametrize("lang", ["hindi", "nepali"])
def test_each_language_model_is_inside_the_parameter_budget(lang):
    """Both configs assert this at import; this proves the assertion fires against a real model and
    not just against the arithmetic."""
    import importlib

    cfg_mod = importlib.import_module(f"{lang}.model.config")
    total, _ = cfg_mod.build().param_count()
    assert total == cfg_mod.N_PARAMS
    assert cfg_mod.PARAM_BUDGET_MIN <= total <= cfg_mod.PARAM_BUDGET_MAX


def test_the_two_language_configs_are_separate_objects():
    """Same shape today, but independent -- changing one must never move the other."""
    from hindi.model.config import CONFIG as HI
    from nepali.model.config import CONFIG as NE

    assert HI is not NE


def test_weight_decay_groups_exclude_norms_and_biases():
    m = GPT(SMALL)
    groups = m.param_groups(weight_decay=0.1)
    decay, no_decay = groups[0]["params"], groups[1]["params"]
    assert all(p.dim() >= 2 for p in decay)
    assert all(p.dim() == 1 for p in no_decay)
    assert sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay) == m.param_count()[0]


# ------------------------------------------------------------------ components

def test_rmsnorm_scales_to_unit_rms():
    n = RMSNorm(64)
    out = n(torch.randn(5, 64) * 17.0)
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(5), atol=1e-3)


def test_rmsnorm_survives_activations_that_would_overflow_fp16():
    """x.pow(2) on half-precision activations of this size gives inf, rsqrt(inf) is 0, and the layer
    silently outputs zeros. Normalising in fp32 is what prevents it."""
    out = RMSNorm(64)(torch.full((2, 64), 300.0, dtype=torch.float16))
    assert torch.isfinite(out).all() and float(out.abs().max()) > 0


def test_swiglu_hidden_width_keeps_the_4x_parameter_budget():
    """3 * d * (8/3 * d) == 2 * d * (4 * d): a gated FFN at the cost of a plain one."""
    d = 384
    swiglu = SwiGLU(d, 1024)
    gated = sum(p.numel() for p in swiglu.parameters())
    plain = 2 * d * (4 * d)
    assert gated == pytest.approx(plain, rel=0.02)


# ------------------------------------------------------------------ generation

def test_greedy_generation_is_deterministic(model):
    idx = torch.randint(0, SMALL.vocab_size, (1, 4))
    a = model.generate(idx, max_new_tokens=5, temperature=0.0)
    b = model.generate(idx, max_new_tokens=5, temperature=0.0)
    assert torch.equal(a, b)
    assert a.shape == (1, 9)


def test_sampling_is_reproducible_under_a_fixed_seed(model):
    idx = torch.randint(0, SMALL.vocab_size, (1, 4))
    torch.manual_seed(1234)
    a = model.generate(idx, max_new_tokens=6, temperature=1.0, top_k=10)
    torch.manual_seed(1234)
    b = model.generate(idx, max_new_tokens=6, temperature=1.0, top_k=10)
    assert torch.equal(a, b)


def test_top_p_never_empties_the_candidate_set(model):
    """A token whose probability already exceeds top_p must still be selectable; the off-by-one that
    drops it leaves nothing to sample from and multinomial raises."""
    idx = torch.randint(0, SMALL.vocab_size, (2, 3))
    out = model.generate(idx, max_new_tokens=4, temperature=0.05, top_p=0.1)
    assert out.shape == (2, 7) and torch.isfinite(out.float()).all()


def test_generation_pads_with_eos_after_a_sequence_finishes(model):
    """Sequences finish at different times but the tensor stays rectangular, so the caller can
    truncate at the first EOS instead of tracking lengths."""
    eos = 5
    idx = torch.randint(0, SMALL.vocab_size, (1, 3))
    tail = model.generate(idx, max_new_tokens=6, temperature=0.0, eos_id=eos)[0, 3:].tolist()
    if eos in tail:
        assert all(t == eos for t in tail[tail.index(eos):])


def test_generation_does_not_leave_the_model_in_eval_mode():
    """generate() flips to eval internally; if it forgot to restore, the next training step would
    silently run with dropout disabled."""
    m = GPT(SMALL)
    m.train()
    m.generate(torch.randint(0, SMALL.vocab_size, (1, 3)), max_new_tokens=2, temperature=0.0)
    assert m.training
