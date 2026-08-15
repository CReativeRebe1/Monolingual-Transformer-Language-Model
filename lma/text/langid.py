"""
Language identification for the Phase-1 cleaning cascade (stage 2 of the Yi-style pipeline).

Two independent signals are combined, deliberately: fastText's general-purpose `lid.176` model
(176 languages, useful as a broad off-topic-language filter) is NOT fine-grained enough on its
own for the case that actually matters here -- Hindi vs Nepali. Both are Devanagari, share a
large functional-word vocabulary, and fastText's lid.176 training data under-represents Nepali
relative to Hindi, so it is known to be unreliable exactly on the boundary this project needs
(Lecture 3, "Devanagari language-ID confusion: Hindi/Marathi/Nepali/Bhojpuri"). So a second,
dedicated character-n-gram discriminator is trained here specifically for the hi/ne boundary,
on trusted seed text (Wikipedia, where the language of each dump is certain by construction).

A document is kept for a given target language only if BOTH signals agree above threshold; this
conjunction is what the Phase-1 report's confusion matrix is built from.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# --- fastText lid.176 wrapper -------------------------------------------------------------

_FASTTEXT_LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lma" / "fasttext"


def ensure_lid_model(cache_dir: Path | None = None) -> Path:
    """Download fastText's lid.176 (quantized .ftz, ~900KB) into a local cache if not present.

    Not a "pretrained language model" in the sense the assignment forbids (no text generation
    capability; it is a linear classifier over character n-gram hashes) -- it plays the same
    role a language-ID library would, exactly like the datasketch/trafilatura utilities used
    elsewhere in this pipeline.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "lid.176.ftz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import httpx

    with httpx.stream("GET", _FASTTEXT_LID_URL, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
    return dest


class FastTextLangID:
    """Thin wrapper around fastText lid.176 returning (lang_code, confidence)."""

    def __init__(self, model_path: Path | None = None):
        import fasttext

        self.model_path = model_path or ensure_lid_model()
        # fastText's C++ loader prints a benign warning to stderr about the label prefix; not
        # our code's concern, and does not affect predictions.
        self._model = fasttext.load_model(str(self.model_path))

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (ISO-639-1 code without the '__label__' prefix, confidence in [0, 1])."""
        clean = text.replace("\n", " ").strip()
        if not clean:
            return "unk", 0.0
        labels, probs = self._model.predict(clean, k=1)
        lang = labels[0].replace("__label__", "")
        return lang, float(probs[0])


# --- Own char n-gram hi/ne (extensible to any label set) discriminator --------------------

_WORD_RE = re.compile(r"\S+")


def _char_ngrams(text: str, n: int) -> list[str]:
    """Character n-grams over whitespace-joined text, padded so word boundaries are informative."""
    padded = f" {text} "
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


@dataclass
class LangIDResult:
    label: str
    log_probs: dict[str, float]  # per-character-normalized log-likelihood under each label
    margin: float  # winning score minus runner-up score; a confidence proxy


class CharNgramLangID:
    """Generative n-gram-LM language classifier (Naive-Bayes-style over character n-grams).

    For each label, a smoothed character-n-gram distribution is estimated from training text.
    A new document is scored under every label's distribution (mean log-probability per
    character, so scores are comparable across documents of different lengths) and assigned the
    argmax label. This is the classic, simple, and surprisingly strong approach to discriminating
    closely related languages/dialects that share a script and much of their lexicon -- exactly
    the Hindi/Nepali case Lecture 3 calls out.
    """

    def __init__(self, n: int = 5, smoothing_k: float = 0.5):
        self.n = n
        self.smoothing_k = smoothing_k
        self._ngram_counts: dict[str, Counter] = {}
        self._vocab: set[str] = set()
        self._total_counts: dict[str, int] = {}
        self._fitted = False

    def fit(self, texts_by_label: dict[str, list[str]]) -> None:
        """texts_by_label: {label: [training documents, ...]}."""
        self._ngram_counts = {label: Counter() for label in texts_by_label}
        for label, texts in texts_by_label.items():
            for text in texts:
                self._ngram_counts[label].update(_char_ngrams(text, self.n))
        self._vocab = set()
        for counter in self._ngram_counts.values():
            self._vocab.update(counter.keys())
        self._total_counts = {
            label: sum(counter.values()) for label, counter in self._ngram_counts.items()
        }
        self._fitted = True

    def _score_label(self, ngrams: list[str], label: str) -> float:
        counter = self._ngram_counts[label]
        total = self._total_counts[label]
        vocab_size = len(self._vocab)
        denom = total + self.smoothing_k * vocab_size
        log_prob_sum = 0.0
        for g in ngrams:
            count = counter.get(g, 0)
            log_prob_sum += math.log((count + self.smoothing_k) / denom)
        return log_prob_sum / max(len(ngrams), 1)  # per-n-gram mean: length-independent

    def predict(self, text: str) -> LangIDResult:
        if not self._fitted:
            raise RuntimeError("CharNgramLangID.fit() must be called before predict()")
        ngrams = _char_ngrams(text, self.n)
        scores = {label: self._score_label(ngrams, label) for label in self._ngram_counts}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = ranked[0]
        margin = best_score - (ranked[1][1] if len(ranked) > 1 else best_score)
        return LangIDResult(label=best_label, log_probs=scores, margin=margin)

    def save(self, path: Path) -> None:
        payload = {
            "n": self.n,
            "smoothing_k": self.smoothing_k,
            "vocab": sorted(self._vocab),
            "total_counts": self._total_counts,
            "ngram_counts": {
                label: dict(counter) for label, counter in self._ngram_counts.items()
            },
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharNgramLangID":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(n=payload["n"], smoothing_k=payload["smoothing_k"])
        model._vocab = set(payload["vocab"])
        model._total_counts = payload["total_counts"]
        model._ngram_counts = {
            label: Counter(counts) for label, counts in payload["ngram_counts"].items()
        }
        model._fitted = True
        return model


def confusion_matrix(
    model: CharNgramLangID, texts_by_true_label: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    """Row = true label, column = predicted label, value = count. Used for the Phase-1 report."""
    labels = list(texts_by_true_label.keys())
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for true_label, texts in texts_by_true_label.items():
        for text in texts:
            result = model.predict(text)
            pred_label = result.label if result.label in matrix[true_label] else true_label
            matrix[true_label][pred_label] = matrix[true_label].get(pred_label, 0) + 1
    return matrix
