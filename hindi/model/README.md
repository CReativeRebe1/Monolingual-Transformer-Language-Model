# hindi/model

*Phase 2.* Model configuration for this language's ~25M-parameter decoder-only
Transformer (layer/head/dim counts, context length, tied-embedding flag).
The architecture itself is shared code in `lma/model/`; only the
per-language config and exact parameter count live here, because the
vocabulary size (and therefore the embedding table) differs per language.
