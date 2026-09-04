"""Shared evaluation metrics: perplexity/BPB, generation protocol, text metrics, diversity
diagnostics, and attention analysis. Language-neutral -- `<lang>/eval/` supplies the model,
tokenizer and paths."""
from lma.eval.attention import analyse, head_distance, head_entropy, layer_means, role_counts
from lma.eval.diversity import distinct_n, diversity_report, repetition_rate, self_bleu
from lma.eval.generate import DECODING_GRID, EXTRA_GRID, REQUIRED_GRID, Continuation, continuations
from lma.eval.perplexity import PerplexityResult, bits_per_byte, evaluate_split
from lma.eval.textmetrics import all_metrics, bleu, chrf, rouge_l

__all__ = [
    "analyse", "head_entropy", "head_distance", "layer_means", "role_counts",
    "distinct_n", "repetition_rate", "self_bleu", "diversity_report",
    "Continuation", "continuations", "DECODING_GRID", "REQUIRED_GRID", "EXTRA_GRID",
    "PerplexityResult", "evaluate_split", "bits_per_byte",
    "bleu", "chrf", "rouge_l", "all_metrics",
]
