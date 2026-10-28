# 05 build dispatcher

Command: `05_build_dispatcher.py --smoke`

Inputs:
- data/images (train, val)
- models/finetuned/
- models/yolov8*.pt

Outputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\dispatcher_smoke
- sweep.csv
- alpha_vs_cost.png
- ideal_split_by_split.csv

Chosen: scheme=none, penalty=1
GFLOPs per model {'n': 8.7439872, 's': 28.602624, 'm': 78.9367296, 'l': 165.1485184}, extractor 3.16 (FlopCounterMode, added for every frame)
Ideal-label split per split (train labels come from models that were trained on those frames, so train may be skewed to small models):

| split | n | s | m | l |
|---|---|---|---|---|
| train | 0.1 | 0.02 | 0.48 | 0.4 |
| val | 0.42 | 0.38 | 0.09 | 0.11 |

Key numbers:

| scheme | penalty | val_alpha_sys | val_gflops | share_n | share_s | share_m | share_l | balanced | chosen |
|---|---|---|---|---|---|---|---|---|---|
| none | 1 | 0.81 | 114.95 | 0.28 | 0.07 | 0.0 | 0.65 | False | True |
| none | 2 | 0.57 | 48.1 | 0.69 | 0.09 | 0.0 | 0.22 | False | False |
| none | 4 | 0.47 | 18.95 | 0.92 | 0.04 | 0.0 | 0.04 | False | False |
| none | 8 | 0.43 | 12.3 | 0.98 | 0.02 | 0.0 | 0.0 | False | False |
| none | 16 | 0.42 | 11.9 | 1.0 | 0.0 | 0.0 | 0.0 | False | False |
| inv | 1 | 0.85 | 93.75 | 0.11 | 0.42 | 0.0 | 0.47 | False | False |
| inv | 2 | 0.72 | 70.44 | 0.39 | 0.27 | 0.0 | 0.34 | False | False |
| inv | 4 | 0.64 | 48.54 | 0.53 | 0.27 | 0.0 | 0.2 | False | False |
| inv | 8 | 0.64 | 53.61 | 0.55 | 0.21 | 0.0 | 0.24 | False | False |
| inv | 16 | 0.62 | 45.99 | 0.59 | 0.22 | 0.0 | 0.19 | False | False |
| sqrt_inv | 1 | 0.86 | 136.65 | 0.15 | 0.06 | 0.0 | 0.79 | False | False |
| sqrt_inv | 2 | 0.77 | 103.61 | 0.37 | 0.05 | 0.0 | 0.58 | False | False |
| sqrt_inv | 4 | 0.58 | 50.45 | 0.64 | 0.13 | 0.0 | 0.23 | False | False |
| sqrt_inv | 8 | 0.48 | 26.57 | 0.88 | 0.03 | 0.0 | 0.09 | False | False |
| sqrt_inv | 16 | 0.42 | 13.46 | 0.99 | 0.0 | 0.0 | 0.01 | False | False |

Flags:
- NO CONFIG IS BALANCED on val (max share <= 0.70 and min share >= 0.05). Fell back to the config with the smallest violation. Ideal labels on val: {'n': 0.42, 's': 0.38, 'm': 0.09, 'l': 0.11}
