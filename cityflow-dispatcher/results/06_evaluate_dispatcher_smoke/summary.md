# 06 evaluate dispatcher (test split, evaluated once)

Command: `06_evaluate_dispatcher.py --smoke`

Inputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\dispatcher_smoke
- test split embeddings and pool table

Outputs:
- policies.csv
- routing_quality.csv
- confusion.csv
- confusion.png
- routing_vs_ideal_split.csv
- per_camera.csv

Head: scheme=none, penalty=1

| routing_accuracy | underestimate_rate | overestimate_rate |
|---|---|---|
| 0.19 | 0.15 | 0.66 |

Routing vs ideal split:

| model | ideal | routed |
|---|---|---|
| n | 0.04 | 0.17 |
| s | 0.41 | 0.0 |
| m | 0.38 | 0.0 |
| l | 0.17 | 0.83 |

Confusion (rows ideal, cols predicted):

| ideal | n | s | m | l |
|---|---|---|---|---|
| n | 2 | 0 | 0 | 2 |
| s | 13 | 0 | 0 | 28 |
| m | 2 | 0 | 0 | 36 |
| l | 0 | 0 | 0 | 17 |

Dispatcher cost includes the extractor; always-X baselines do not.

Key numbers:

| policy | alpha_sys | avg_gflops | saving_vs_always_l |
|---|---|---|---|
| dispatcher | 0.85 | 141.72 | 0.1419 |
| always_n | 0.04 | 8.74 | 0.9471 |
| always_s | 0.45 | 28.6 | 0.8268 |
| always_m | 0.78 | 78.94 | 0.522 |
| always_l | 1.0 | 165.15 | -0.0 |
| oracle (ideal label) | 1.0 | 70.15 | 0.5752 |

Flags:
- none
