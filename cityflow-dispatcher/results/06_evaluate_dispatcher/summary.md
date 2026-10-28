# 06 evaluate dispatcher (test split, evaluated once)

Command: `06_evaluate_dispatcher.py`

Inputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\dispatcher
- test split embeddings and pool table

Outputs:
- policies.csv
- routing_quality.csv
- confusion.csv
- confusion.png
- routing_vs_ideal_split.csv
- per_camera.csv

Head: scheme=inv, penalty=8

| routing_accuracy | underestimate_rate | overestimate_rate |
|---|---|---|
| 0.1763 | 0.1147 | 0.709 |

Routing vs ideal split:

| model | ideal | routed |
|---|---|---|
| n | 0.5179 | 0.0031 |
| s | 0.3101 | 0.6819 |
| m | 0.1122 | 0.0031 |
| l | 0.0598 | 0.312 |

Confusion (rows ideal, cols predicted):

| ideal | n | s | m | l |
|---|---|---|---|---|
| n | 2 | 670 | 1 | 167 |
| s | 3 | 254 | 1 | 245 |
| m | 0 | 114 | 2 | 66 |
| l | 0 | 68 | 1 | 28 |

Dispatcher cost includes the extractor; always-X baselines do not.

Key numbers:

| policy | alpha_sys | avg_gflops | saving_vs_always_l |
|---|---|---|---|
| dispatcher | 0.8514 | 74.45 | 0.5492 |
| always_n | 0.5179 | 8.74 | 0.9471 |
| always_s | 0.7799 | 28.6 | 0.8268 |
| always_m | 0.8674 | 78.94 | 0.522 |
| always_l | 1.0 | 165.15 | -0.0 |
| oracle (ideal label) | 1.0 | 32.13 | 0.8054 |

Flags:
- none
