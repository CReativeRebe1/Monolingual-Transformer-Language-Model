# Phase 3 — Model H vs Model L on comparative reasoning

Mean ± std over seeds. EM = generative exact match; cand = candidate-choice accuracy; PPL = Phase-2 validation perplexity (forgetting).

| model | condition | seeds | EM test_id | EM test_entity | EM test_wording | EM test_both | cand test_both | PPL |
|---|---|---|---|---|---|---|---|---|
| Model H (Hindi) | pretrained 0-shot | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.221 | 30.337 |
| Model H (Hindi) | pretrained 3-shot | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.249 | - |
| Model H (Hindi) | SFT | 3 | 0.388 ± 0.049 | 0.136 ± 0.043 | 0.331 ± 0.056 | 0.102 ± 0.020 | 0.319 ± 0.012 | 142.359 ± 85.189 |
| Model H (Hindi) | SFT + replay | 3 | 0.422 ± 0.045 | 0.129 ± 0.009 | 0.397 ± 0.036 | 0.102 ± 0.005 | 0.339 ± 0.006 | 35.507 ± 0.106 |
| Model L (Nepali) | pretrained 0-shot | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.265 | 51.233 |
| Model L (Nepali) | pretrained 3-shot | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.267 | - |
| Model L (Nepali) | SFT | 3 | 0.353 ± 0.007 | 0.103 ± 0.002 | 0.322 ± 0.002 | 0.085 ± 0.003 | 0.273 ± 0.004 | 418137.447 ± 313846.086 |
| Model L (Nepali) | SFT + replay | 3 | 0.368 ± 0.011 | 0.134 ± 0.004 | 0.352 ± 0.011 | 0.107 ± 0.006 | 0.292 ± 0.003 | 76.318 ± 0.194 |

## Exact match by task type (SFT, test_both)

| model | cmp2 | cmp3 | ext3 | pair3 | ext4 |
|---|---|---|---|---|---|
| Model H (Hindi) | 0.198 ± 0.044 | 0.264 ± 0.006 | 0.030 ± 0.052 | 0.010 ± 0.017 | 0.003 ± 0.006 |
| Model L (Nepali) | 0.180 ± 0.011 | 0.241 ± 0.003 | 0.000 | 0.000 | 0.000 |
