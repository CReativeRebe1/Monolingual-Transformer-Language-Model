#!/usr/bin/env bash
# Runs the full Phase-1 pipeline end to end on a small bundled sample corpus: normalize ->
# language-ID -> heuristic filter -> learned quality filter -> dedup -> split -> stats/figures ->
# tokenizer training -> corpus encoding. No GPU needed, takes a couple of minutes.
#
# Deliberately does NOT touch hindi/ or nepali/ (the real corpus, possibly being added to by a
# live crawl right now) -- everything here runs against an isolated demo/ directory instead, via
# build_corpus.py's --lang-dir override. This demonstrates that the pipeline works and is
# reproducible; it is not a substitute for the real ~500M-token corpus (see PHASE1_LOG.md and
# report/phase1.md for that).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEMO_DIR="$REPO_ROOT/scripts/sample_data/demo_run"
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"

echo "=== [1/4] writing sample raw shards ==="
python scripts/make_sample_shards.py "$DEMO_DIR"

for LANG in hindi nepali; do
    echo ""
    echo "=== [2/4] building corpus (clean -> filter -> dedup -> split -> stats) : $LANG ==="
    python scripts/build_corpus.py --lang "$LANG" --lang-dir "$DEMO_DIR/$LANG" --report-dir "$DEMO_DIR/report_figures"

    echo ""
    echo "=== [3/4] training a tiny tokenizer : $LANG ==="
    python - "$LANG" "$DEMO_DIR/$LANG" <<'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in dir() else "."))
lang, lang_dir = sys.argv[1], Path(sys.argv[2])

from lma.tokenizer.encode_corpus import encode_split
from lma.tokenizer.train_sp import Tokenizer, TokenizerTrainConfig, train_tokenizer_from_texts

splits_dir = lang_dir / "data" / "splits"
train_path = splits_dir / "train.txt"
if not train_path.exists():
    print(f"  (no train split found for {lang}, skipping tokenizer/encode step)")
    sys.exit(0)

texts = [line.strip() for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(texts) < 2:
    print(f"  (too few documents survived the pipeline for {lang} to train a tokenizer -- "
          f"expected on a 6-7 document demo corpus; the real corpus does not have this problem)")
    sys.exit(0)

tok_dir = lang_dir / "tokenizer"
config = TokenizerTrainConfig(algorithm="bpe", vocab_size=350, character_coverage=1.0)
model_path, _ = train_tokenizer_from_texts(texts, tok_dir / f"{lang}_demo", config)
tok = Tokenizer(model_path)
print(f"  trained a {tok.vocab_size}-token demo BPE tokenizer for {lang} -> {model_path}")

print(f"  === [4/4] encoding corpus to .bin/.idx : {lang} ===")
info = encode_split(tok, train_path, lang_dir / "data" / "encoded", "train")
print(f"  encoded {info.n_documents} documents, {info.n_tokens} tokens -> {info.bin_path}")
PYEOF
done

echo ""
echo "=== done ==="
echo "Demo artifacts written under $DEMO_DIR (safe to delete; re-run this script any time)."
echo "This demonstrates the PIPELINE works end to end. The real corpus (target ~500M tokens"
echo "per language, >=20% manually collected) is built by scripts/build_corpus.py against"
echo "hindi/ and nepali/ directly, fed by the live crawler -- see PHASE1_LOG.md for current"
echo "real numbers and report/phase1.md for the final report."
