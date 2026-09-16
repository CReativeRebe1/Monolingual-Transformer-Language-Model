# model_h (hindi) reasoning results

Mean ± std over seeds where more than one seed was run.

## Exact match by split

| condition | seeds | val | test_id | test_entity | test_wording | test_both | Phase-2 val PPL |
|---|---|---|---|---|---|---|---|
| pretrained_0shot | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 30.337 |
| pretrained_3shot | 1 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| replay_lr0.0001 | 3 | 0.434 ± 0.029 | 0.422 ± 0.045 | 0.129 ± 0.009 | 0.397 ± 0.036 | 0.102 ± 0.005 | 35.507 ± 0.106 |
| sft_lr0.0001_n1000 | 1 | 0.297 | 0.296 | 0.227 | 0.311 | 0.208 | - |
| sft_lr0.0001_n5000 | 1 | 0.331 | 0.330 | 0.158 | 0.297 | 0.109 | - |
| sft_lr0.0001 | 3 | 0.408 ± 0.038 | 0.388 ± 0.049 | 0.136 ± 0.043 | 0.331 ± 0.056 | 0.102 ± 0.020 | 142.359 ± 85.189 |
| sft_lr0.0003 | 1 | 0.362 | - | - | - | - | - |
| sft_lr5e-05 | 1 | 0.368 | - | - | - | - | - |

## Candidate-choice accuracy by split

| condition | val | test_id | test_entity | test_wording | test_both |
|---|---|---|---|---|---|
| pretrained_0shot | 0.243 | 0.222 | 0.224 | 0.248 | 0.221 |
| pretrained_3shot | 0.260 | 0.245 | 0.238 | 0.267 | 0.249 |
| replay_lr0.0001 | 0.433 ± 0.028 | 0.423 ± 0.047 | 0.358 ± 0.004 | 0.402 ± 0.042 | 0.339 ± 0.006 |
| sft_lr0.0001_n1000 | 0.301 | 0.299 | 0.300 | 0.319 | 0.301 |
| sft_lr0.0001_n5000 | 0.333 | 0.329 | 0.363 | 0.330 | 0.329 |
| sft_lr0.0001 | 0.407 ± 0.035 | 0.386 ± 0.046 | 0.349 ± 0.024 | 0.354 ± 0.069 | 0.319 ± 0.012 |
| sft_lr0.0003 | 0.368 | - | - | - | - |
| sft_lr5e-05 | 0.372 | - | - | - | - |

## Exact match by task type (test_both)

| condition | cmp2 | cmp3 | ext3 | pair3 | ext4 |
|---|---|---|---|---|---|
| pretrained_0shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pretrained_3shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replay_lr0.0001 | 0.219 ± 0.003 | 0.264 ± 0.010 | 0.012 ± 0.008 | 0.007 ± 0.003 | 0.003 ± 0.006 |
| sft_lr0.0001_n1000 | 0.304 | 0.262 | 0.177 | 0.187 | 0.106 |
| sft_lr0.0001_n5000 | 0.167 | 0.248 | 0.086 | 0.015 | 0.025 |
| sft_lr0.0001 | 0.198 ± 0.044 | 0.264 ± 0.006 | 0.030 ± 0.052 | 0.010 ± 0.017 | 0.003 ± 0.006 |

## Error types (test_both, mean count over seeds)

| condition | no_stop | reversed_extreme | middle_entity | equal_case_miss | out_of_prompt_name | format_failure |
|---|---|---|---|---|---|---|
| pretrained_0shot | 139 | 150 | 354 | 38 | 21 | 298 |
| pretrained_3shot | 19 | 9 | 32 | 3 | 546 | 391 |
| replay_lr0.0001 | 0 | 3 | 0 | 267 | 598 | 30 |
| sft_lr0.0001_n1000 | 0 | 152 | 118 | 361 | 151 | 10 |
| sft_lr0.0001_n5000 | 0 | 16 | 2 | 278 | 523 | 72 |
| sft_lr0.0001 | 0 | 13 | 2 | 291 | 560 | 32 |

## Heuristic baselines (test splits)

| split | chance | first_mentioned | last_mentioned | most_frequent | random_candidate |
|---|---|---|---|---|---|
| test_id | 0.300 | 0.241 | 0.263 | 0.119 | 0.303 |
| test_entity | 0.300 | 0.245 | 0.280 | 0.115 | 0.316 |
| test_wording | 0.300 | 0.274 | 0.265 | 0.132 | 0.313 |
| test_both | 0.300 | 0.243 | 0.257 | 0.128 | 0.296 |
