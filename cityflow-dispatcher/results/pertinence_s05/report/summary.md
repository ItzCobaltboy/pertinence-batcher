# 10 PERTINENCE front vs static policies (S02)

Command: `10_pertinence_report.py`

Inputs:
- /sessions/rcw-01gguojw7voswjszyp5ndq6h/mnt/pertinence-batcher/cityflow-dispatcher/results/pertinence_s05/eval/final_val_summary.csv
- /sessions/rcw-01gguojw7voswjszyp5ndq6h/mnt/pertinence-batcher/cityflow-dispatcher/results/pertinence_s05/eval/predictions/final_val_predictions.csv
- /sessions/rcw-01gguojw7voswjszyp5ndq6h/mnt/pertinence-batcher/cityflow-dispatcher/data/pertinence_s05/final_val_ground_truth.csv

Outputs:
- static_policies.csv
- front_vs_static.csv
- front_vs_static.png

Run: pertinence_s05; final split = S02 (reported once).
Front points: 109; above the static hull: 8; balanced (max share <= 0.7, min share >= 0.05): 0.
Best gain over the static hull: +0.0127 alpha_sys at 146.1 GFLOPs (individual 85, shares n/s/m/l 0.00/0.16/0.00/0.84).

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
| 88 | 0.5179 | 11.8995 | -0.0416 | 0.5179 | 1.0 | 0.0 | 0.0 | 0.0 | False |
| 12 | 0.5179 | 11.8995 | -0.0416 | 0.5179 | 1.0 | 0.0 | 0.0 | 0.0 | False |
| 13 | 0.5191 | 12.0924 | -0.0429 | 0.5185 | 0.9988 | 0.0 | 0.0 | 0.0012 | False |
| 47 | 0.5382 | 13.0994 | -0.0371 | 0.5049 | 0.9396 | 0.0604 | 0.0 | 0.0 | False |
| 90 | 0.5456 | 17.6147 | -0.0893 | 0.5099 | 0.9581 | 0.0062 | 0.0 | 0.0358 | False |
| 30 | 0.6924 | 23.6286 | -0.0219 | 0.4174 | 0.4094 | 0.5906 | 0.0 | 0.0 | False |
| 37 | 0.7737 | 34.2947 | -0.0161 | 0.3705 | 0.1054 | 0.8607 | 0.0 | 0.0339 | False |
| 62 | 0.8243 | 57.6026 | -0.0061 | 0.2972 | 0.0 | 0.8107 | 0.0 | 0.1893 | False |
| 49 | 0.8354 | 72.3839 | -0.0207 | 0.307 | 0.0993 | 0.5888 | 0.0 | 0.312 | False |
| 83 | 0.8767 | 88.8347 | -0.006 | 0.2053 | 0.0 | 0.582 | 0.0 | 0.418 | False |
| 77 | 0.939 | 126.0516 | -0.0009 | 0.1301 | 0.0123 | 0.2953 | 0.0 | 0.6924 | False |
| 80 | 1.0 | 168.3041 | 0.0 | 0.0598 | 0.0 | 0.0 | 0.0 | 1.0 | False |

Flags:
- table shows 12 evenly spaced front points; all of them are in front_vs_static.csv
- dispatcher cost includes extractor + FC; static policies do not (they need no dispatcher)
- balanced rule is checked after the search, it is not a constraint inside NSGA-II
