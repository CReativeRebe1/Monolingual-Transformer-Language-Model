"""
Model L (Nepali): architecture configuration and the parameter budget it has to satisfy.

Separate from Hindi's for the same reason `nepali/clean/config.py` is separate from Hindi's: these
are two independent models, and a shared config file would mean neither could be changed without
moving the other. Phase 1 selected 16,000 pieces for both languages, so the two shapes currently
coincide -- but that is a measured outcome, not a constraint, and this file is where a divergence
would be recorded if Nepali's evaluation later argued for one.

The budget assertion at the bottom is not decoration; see Hindi's copy for what it catches.
"""
from __future__ import annotations

from pathlib import Path

from lma.model.transformer import GPT, GPTConfig, analytic_param_count

LANGUAGE = "nepali"
MODEL_NAME = "model_l"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOKENIZER_PATH = REPO_ROOT / "nepali" / "tokenizer" / "nepali_unigram_16000.model"
ENCODED_DIR = REPO_ROOT / "nepali" / "data" / "encoded"

VOCAB_SIZE = 16_000

CONFIG = GPTConfig(
    vocab_size=VOCAB_SIZE,
    d_model=384,
    n_layers=10,
    n_heads=6,          # head_dim = 64
    mlp_hidden=1024,    # (8/3)*d_model
    max_seq_len=1024,
    dropout=0.0,        # 571M training tokens against 23.8M parameters -- ~24 tokens per parameter,
                        # even more data-rich than Hindi. Nothing to regularise against.
    rope_base=10_000.0,
    tie_embeddings=True,
)

PARAM_BUDGET_MIN = 23_000_000
PARAM_BUDGET_MAX = 25_000_000

PARAMS = analytic_param_count(CONFIG)
N_PARAMS = PARAMS["total"]

if not PARAM_BUDGET_MIN <= N_PARAMS <= PARAM_BUDGET_MAX:
    raise AssertionError(
        f"{MODEL_NAME} has {N_PARAMS:,} parameters, outside the "
        f"{PARAM_BUDGET_MIN:,}-{PARAM_BUDGET_MAX:,} budget this project is held to"
    )


def build() -> GPT:
    """Constructs Model L. The only place Nepali's config meets the shared architecture."""
    return GPT(CONFIG)
