# hindi/reasoning

*Phase 3.* Model H's synthetic comparative-reasoning dataset, finetuning, evaluation and
post-finetuning attention analysis. Everything language-specific lives here; the machinery is shared
in `lma/reasoning/`.

| file | what it holds |
|---|---|
| `lexicon.py` | 200 person names (with gender), 60 objects, 13 countable items — Hindi spellings as the corpus writes them |
| `templates.py` | four surface-form families (F1–F3 train, F4 held out), gender agreement, ASCII digits |
| `config.py` | dataset seed, SFT hyperparameters, sweep grid, paths |
| `generate.py` | writes `data/*.jsonl` and `results/dataset_stats.{json,md}` |
| `finetune.py` | one run: `sft` or `replay`; resumes automatically |
| `evaluate.py` | EM, candidate choice, prefix match, error taxonomy, Phase-2 val perplexity; `--summarize` |
| `attention_compare.py` | pretrained vs finetuned heatmaps, entropy/distance, head roles, answer-pointer grids |
| `data/` | `train` (Drive; regenerable), `val`, `test_id`, `test_entity`, `test_wording`, `test_both` |
| `logs/<run>/` | append-only `train_log.jsonl` + `train.log` + `summary.json` per run |
| `results/` | dataset stats, per-run evaluations (`eval/`), `summary.md`, `attention_compare.json` |
| `checkpoints/<run>/` | `ckpt_best.pt` (by val EM), `ckpt_last.pt` — gitignored, on Drive |

```bash
python hindi/reasoning/generate.py                       # dataset (byte-reproducible)
python scripts/run_phase3.py --lang hindi                # every experiment, in order, resumable
# or by hand:
python hindi/reasoning/evaluate.py --pretrained --shots 0
python hindi/reasoning/finetune.py --condition sft --lr 1e-4 --seed 0
python hindi/reasoning/evaluate.py --run sft_lr0.0001_s0
python hindi/reasoning/evaluate.py --summarize
python hindi/reasoning/attention_compare.py --run replay_lr0.0001_s0
```
