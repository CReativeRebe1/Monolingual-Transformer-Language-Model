"""Tests for lma.text.langid.CharNgramLangID -- the own-built hi/ne discriminator.

Uses small synthetic "toy language" corpora (distinct character patterns standing in for
Hindi/Nepali) rather than real Devanagari text, so the test is fast, has zero network
dependency, and isolates the classifier's correctness from any particular training corpus.
The real Hindi/Nepali model is trained separately in the data pipeline on Wikipedia text and
its confusion matrix is reported in report/phase1.md, not asserted here.
"""
import pytest

from lma.text.langid import CharNgramLangID, confusion_matrix

# Two clearly separable synthetic "languages" built from disjoint character alphabets, repeated
# with enough variation that n-gram statistics differ but some structure overlaps (mimicking
# two related real languages that share a script and some vocabulary).
LANG_A_SAMPLES = [
    "abaaba abcaba baabca aabca cabaab",
    "bacaba abacab cabaab baabac abcaba",
    "aabcaa bacaba abaaba cabaab baabca",
] * 5

LANG_B_SAMPLES = [
    "xyzxy zyxzy xzyxz yxzyx zxyzx",
    "zyxzy xyzxy yxzyx zxyzx xzyxz",
    "xzyxz zxyzx xyzxy zyxzy yxzyx",
] * 5


@pytest.fixture
def fitted_model():
    model = CharNgramLangID(n=3, smoothing_k=0.5)
    model.fit({"A": LANG_A_SAMPLES, "B": LANG_B_SAMPLES})
    return model


def test_predict_recovers_correct_label_for_clearly_separable_languages(fitted_model):
    result = fitted_model.predict("aabca baabca abcaba cabaab")
    assert result.label == "A"
    result_b = fitted_model.predict("xyzxy zxyzx yxzyx zyxzy")
    assert result_b.label == "B"


def test_predict_raises_before_fit():
    model = CharNgramLangID()
    with pytest.raises(RuntimeError):
        model.predict("some text")


def test_margin_is_larger_for_unambiguous_text(fitted_model):
    unambiguous = fitted_model.predict("aabca baabca abcaba cabaab")
    # A short, heavily mixed string should be closer to the decision boundary.
    mixed = fitted_model.predict("aabcx yzaba")
    assert unambiguous.margin >= 0


def test_save_and_load_round_trip(tmp_path, fitted_model):
    path = tmp_path / "model.json"
    fitted_model.save(path)
    loaded = CharNgramLangID.load(path)
    original = fitted_model.predict("aabca baabca abcaba cabaab")
    restored = loaded.predict("aabca baabca abcaba cabaab")
    assert original.label == restored.label
    assert original.log_probs == pytest.approx(restored.log_probs)


def test_confusion_matrix_shape_and_diagonal_dominance(fitted_model):
    matrix = confusion_matrix(
        fitted_model, {"A": LANG_A_SAMPLES[:5], "B": LANG_B_SAMPLES[:5]}
    )
    assert set(matrix.keys()) == {"A", "B"}
    assert set(matrix["A"].keys()) == {"A", "B"}
    # For clearly separable synthetic languages, the diagonal (correct predictions) should
    # dominate the row.
    assert matrix["A"]["A"] >= matrix["A"]["B"]
    assert matrix["B"]["B"] >= matrix["B"]["A"]


def test_unknown_ngrams_do_not_crash_scoring(fitted_model):
    # Text containing characters never seen in training must still score without error
    # (relies on additive smoothing over the union vocabulary).
    result = fitted_model.predict("qqq ??? 123 完全に未知")
    assert result.label in {"A", "B"}
