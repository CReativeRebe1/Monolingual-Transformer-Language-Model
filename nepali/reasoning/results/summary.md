# model_l (nepali) reasoning results

Mean ± std over seeds where more than one seed was run.

## Exact match by split

| condition | seeds | val | test_id | test_entity | test_wording | test_both | Phase-2 val PPL |
|---|---|---|---|---|---|---|---|
| pretrained_0shot | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 51.233 |
| pretrained_3shot | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - |
| replay_lr0.0003 | 3 | 0.371 ± 0.005 | 0.368 ± 0.011 | 0.134 ± 0.004 | 0.352 ± 0.011 | 0.107 ± 0.006 | 76.318 ± 0.194 |
| sft_lr0.0001 | 1 | 0.356 | - | - | - | - | - |
| sft_lr0.0003_n1000 | 1 | 0.300 | 0.280 | 0.154 | 0.304 | 0.133 | - |
| sft_lr0.0003_n5000 | 1 | 0.329 | 0.351 | 0.108 | 0.283 | 0.087 | - |
| sft_lr0.0003 | 3 | 0.368 ± 0.003 | 0.353 ± 0.007 | 0.103 ± 0.002 | 0.322 ± 0.002 | 0.085 ± 0.003 | 418137.447 ± 313846.086 |
| sft_lr5e-05 | 1 | 0.339 | - | - | - | - | - |

## Candidate-choice accuracy by split

| condition | val | test_id | test_entity | test_wording | test_both |
|---|---|---|---|---|---|
| pretrained_0shot | 0.255 | 0.266 | 0.244 | 0.261 | 0.265 |
| pretrained_3shot | 0.266 | 0.290 | 0.270 | 0.297 | 0.267 |
| replay_lr0.0003 | 0.373 ± 0.006 | 0.368 ± 0.010 | 0.335 ± 0.009 | 0.352 ± 0.010 | 0.292 ± 0.003 |
| sft_lr0.0001 | 0.357 | - | - | - | - |
| sft_lr0.0003_n1000 | 0.300 | 0.289 | 0.300 | 0.306 | 0.271 |
| sft_lr0.0003_n5000 | 0.325 | 0.351 | 0.315 | 0.283 | 0.282 |
| sft_lr0.0003 | 0.365 ± 0.002 | 0.356 ± 0.006 | 0.293 ± 0.005 | 0.326 ± 0.001 | 0.273 ± 0.004 |
| sft_lr5e-05 | 0.338 | - | - | - | - |

## Exact match by task type (test_both)

| condition | cmp2 | cmp3 | ext3 | pair3 | ext4 |
|---|---|---|---|---|---|
| pretrained_0shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pretrained_3shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replay_lr0.0003 | 0.190 ± 0.007 | 0.251 ± 0.017 | 0.030 ± 0.010 | 0.032 ± 0.003 | 0.029 ± 0.008 |
| sft_lr0.0003_n1000 | 0.201 | 0.267 | 0.096 | 0.045 | 0.051 |
| sft_lr0.0003_n5000 | 0.181 | 0.248 | 0.000 | 0.000 | 0.000 |
| sft_lr0.0003 | 0.180 ± 0.011 | 0.241 ± 0.003 | 0.000 | 0.000 | 0.000 |

## Error types (test_both, mean count over seeds)

| condition | no_stop | reversed_extreme | middle_entity | equal_case_miss | out_of_prompt_name | format_failure |
|---|---|---|---|---|---|---|
| pretrained_0shot | 20 | 16 | 29 | 5 | 2 | 928 |
| pretrained_3shot | 2 | 6 | 4 | 2 | 92 | 894 |
| replay_lr0.0003 | 0 | 21 | 17 | 348 | 501 | 5 |
| sft_lr0.0003_n1000 | 0 | 47 | 61 | 301 | 444 | 14 |
| sft_lr0.0003_n5000 | 0 | 2 | 6 | 299 | 587 | 19 |
| sft_lr0.0003 | 0 | 0 | 0 | 386 | 528 | 0 |

## Heuristic baselines (test splits)

| split | chance | first_mentioned | last_mentioned | most_frequent | random_candidate |
|---|---|---|---|---|---|
| test_id | 0.300 | 0.241 | 0.263 | 0.119 | 0.303 |
| test_entity | 0.300 | 0.245 | 0.280 | 0.115 | 0.316 |
| test_wording | 0.300 | 0.274 | 0.265 | 0.132 | 0.313 |
| test_both | 0.300 | 0.243 | 0.257 | 0.128 | 0.296 |
