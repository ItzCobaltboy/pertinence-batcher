# 10 PERTINENCE front vs static policies (S02)

Command: `10_pertinence_report.py`

Inputs:
- /sessions/rcw-01gguojw7voswjszyp5ndq6h/mnt/pertinence-batcher/cityflow-dispatcher/results/pertinence_train/eval/final_val_summary.csv
- /sessions/rcw-01gguojw7voswjszyp5ndq6h/mnt/pertinence-batcher/cityflow-dispatcher/results/pertinence_train/eval/predictions/final_val_predictions.csv
- /sessions/rcw-01gguojw7voswjszyp5ndq6h/mnt/pertinence-batcher/cityflow-dispatcher/data/pertinence_train/final_val_ground_truth.csv

Outputs:
- static_policies.csv
- front_vs_static.csv
- front_vs_static.png

Run: pertinence_train; final split = S02 (reported once).
Front points: 119; above the static hull: 28; balanced (max share <= 0.7, min share >= 0.05): 0.
Best gain over the static hull: +0.0382 alpha_sys at 21.8 GFLOPs (individual 13, shares n/s/m/l 0.50/0.50/0.00/0.00).

Static policies on S02:

| policy | alpha_sys | avg_gflops |
|---|---|---|
| always_yolov8n | 0.5179 | 8.744 |
| always_yolov8s | 0.7799 | 28.6026 |
| always_yolov8m | 0.8674 | 78.9367 |
| always_yolov8l | 1.0 | 165.1485 |
| oracle | 1.0 | 32.1319 |

Key numbers:

| individual | alpha_sys | avg_gflops | gain_over_static_hull | exact_match_vs_ideal | share_n | share_s | share_m | share_l | balanced |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5179 | 11.8995 | -0.0416 | 0.5179 | 1.0 | 0.0 | 0.0 | 0.0 | False |
| 75 | 0.5345 | 13.3565 | -0.0442 | 0.4994 | 0.9266 | 0.0734 | 0.0 | 0.0 | False |
| 105 | 0.5919 | 16.6377 | -0.0302 | 0.4883 | 0.7614 | 0.2386 | 0.0 | 0.0 | False |
| 33 | 0.5894 | 20.6672 | -0.0858 | 0.4143 | 0.6991 | 0.2454 | 0.0555 | 0.0 | False |
| 74 | 0.6843 | 23.4695 | -0.0278 | 0.4149 | 0.4174 | 0.5826 | 0.0 | 0.0 | False |
| 95 | 0.7232 | 27.7041 | -0.0449 | 0.3631 | 0.2084 | 0.791 | 0.0 | 0.0006 | False |
| 39 | 0.7793 | 31.7337 | -0.0061 | 0.3101 | 0.0012 | 0.9988 | 0.0 | 0.0 | False |
| 117 | 0.7762 | 35.3062 | -0.0154 | 0.3138 | 0.0487 | 0.9112 | 0.0111 | 0.029 | False |
| 36 | 0.7793 | 47.7964 | -0.034 | 0.3261 | 0.127 | 0.7355 | 0.0025 | 0.135 | False |
| 100 | 0.8921 | 93.6209 | 0.0021 | 0.1905 | 0.0006 | 0.5462 | 0.0 | 0.4531 | False |
| 116 | 0.9371 | 119.8985 | 0.0067 | 0.135 | 0.0 | 0.3545 | 0.0 | 0.6455 | False |
| 88 | 1.0 | 167.8587 | 0.0 | 0.061 | 0.0012 | 0.0018 | 0.0 | 0.9969 | False |

Flags:
- table shows 12 evenly spaced front points; all of them are in front_vs_static.csv
- dispatcher cost includes extractor + FC; static policies do not (they need no dispatcher)
- balanced rule is checked after the search, it is not a constraint inside NSGA-II
