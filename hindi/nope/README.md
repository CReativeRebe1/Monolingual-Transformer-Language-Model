# hindi/nope

*Bonus ablation.* Model H retrained with positional embeddings removed, evaluated with the full
Phase-2 suite, and compared with the standard Model H.

Model H's only position signal is RoPE on the attention queries and keys; `GPTConfig(use_rope=False)`
removes it and changes nothing else. Same architecture, parameter count (23,846,784), data,
tokenizer, training configuration (the Phase-2 `TrainConfig` object itself) and evaluation scripts.

| file | what it does |
|---|---|
| `config.py` | the NoPE model config (`use_rope=False`), paths, and the baseline's result locations |
| `pretrain.py` | torchrun entry point — identical to `hindi/train/pretrain.py` apart from the model |
| `position_probe.py` | loss by position in the window, and how much shuffling the context hurts |
| `compare.py` | standard vs NoPE: perplexity/BPB, generation, diversity, attention, position probes |
| `results/` | `metrics_{test,val}.json`, `attention_stats.json`, `position_probe_*.json`, `comparison.md` |
| `logs/` | append-only NoPE pretraining log |
| `checkpoints/` | gitignored; on Drive |

```bash
python scripts/pack_bonus.py code                  # code-only archive for Kaggle
# on Kaggle: notebooks/bonus_kaggle.md (train, evaluate, attention, probe, compare, results zip)
python scripts/pack_bonus.py results               # logs + results + figures archive
```

Evaluation reuses the Phase-2 scripts with output options:

```bash
python hindi/eval/evaluate.py --checkpoint hindi/nope/checkpoints/ckpt_best.pt --split test \
    --results-dir hindi/nope/results --figure-prefix hindi_nope --label model_h_nope
python hindi/eval/attention_report.py --checkpoint hindi/nope/checkpoints/ckpt_best.pt \
    --results-dir hindi/nope/results --figure-prefix hindi_nope --label model_h_nope \
    --extra-heads 0:3,0:4,0:5,5:0,5:1,5:2,9:0,9:3
python hindi/nope/position_probe.py --checkpoint hindi/nope/checkpoints/ckpt_best.pt --label model_h_nope
python hindi/nope/compare.py
```
