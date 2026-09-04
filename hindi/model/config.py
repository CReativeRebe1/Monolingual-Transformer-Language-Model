"""
Model H (Hindi): architecture configuration and the parameter budget it has to satisfy.

A per-language file, mirroring the Phase-1 split where `hindi/clean/config.py` and
`nepali/clean/config.py` hold their own thresholds. Here the two languages happen to land on the
same shape -- both tokenizers were selected at 16,000 pieces in Phase 1 -- but they are separate
files because they are separate models: changing Hindi's depth must not silently change Nepali's,
and the vocabulary size (which sets the embedding table, 26% of the budget) is a per-language fact.

The `assert` at the bottom is the point of the file. A one-character edit to `mlp_hidden` moves the
total by 6M parameters, and nothing in a training run would report that the model had drifted out of
budget -- the loss curve would look entirely normal. Failing at import is the only place it is cheap.
"""
from __future__ import annotations

from pathlib import Path

from lma.model.transformer import GPT, GPTConfig, analytic_param_count

LANGUAGE = "hindi"
MODEL_NAME = "model_h"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOKENIZER_PATH = REPO_ROOT / "hindi" / "tokenizer" / "hindi_unigram_16000.model"
ENCODED_DIR = REPO_ROOT / "hindi" / "data" / "encoded"

VOCAB_SIZE = 16_000

CONFIG = GPTConfig(
    vocab_size=VOCAB_SIZE,
    d_model=384,
    n_layers=10,
    n_heads=6,          # head_dim = 64: the width where tensor cores are efficient, and small
                        # enough that 6 heads at d_model=384 stays under the "heads must have room
                        # to specialise" floor of ~32 dims per head.
    mlp_hidden=1024,    # (8/3)*d_model -- keeps a gated FFN at the parameter cost of a 4x plain one
    max_seq_len=1024,
    dropout=0.0,        # 517M training tokens against 23.8M parameters is ~22 tokens per parameter:
                        # this model is data-rich, not data-poor, and will not complete even one
                        # epoch's worth of memorisation. Dropout here would only slow convergence.
    rope_base=10_000.0,
    tie_embeddings=True,
)

# Budget: the assignment specifies ~25M parameters. The band is asserted rather than the exact
# number so that a deliberate architecture change is allowed to move it, but an accidental one is not.
PARAM_BUDGET_MIN = 23_000_000
PARAM_BUDGET_MAX = 25_000_000


def build() -> GPT:
    """Constructs Model H. The only place Hindi's config meets the shared architecture."""
    return GPT(CONFIG)


PARAMS = analytic_param_count(CONFIG)
N_PARAMS = PARAMS["total"]

if not PARAM_BUDGET_MIN <= N_PARAMS <= PARAM_BUDGET_MAX:
    raise AssertionError(
        f"{MODEL_NAME} has {N_PARAMS:,} parameters, outside the "
        f"{PARAM_BUDGET_MIN:,}-{PARAM_BUDGET_MAX:,} budget this project is held to"
    )
