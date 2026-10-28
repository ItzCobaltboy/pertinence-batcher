# 05 build dispatcher

Command: `05_build_dispatcher.py`

Inputs:
- data/images (train, val)
- models/finetuned/
- models/yolov8*.pt

Outputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\dispatcher
- sweep.csv
- alpha_vs_cost.png
- ideal_split_by_split.csv

Chosen: scheme=inv, penalty=8
GFLOPs per model {'n': 8.7439872, 's': 28.602624, 'm': 78.9367296, 'l': 165.1485184}, extractor 3.16 (FlopCounterMode, added for every frame)
Ideal-label split per split (train labels come from models that were trained on those frames, so train may be skewed to small models):

| split | n | s | m | l |
|---|---|---|---|---|
| train | 0.884 | 0.053 | 0.015 | 0.047 |
| val | 0.42 | 0.321 | 0.054 | 0.205 |

Key numbers:

| scheme | penalty | val_alpha_sys | val_gflops | share_n | share_s | share_m | share_l | balanced | chosen |
|---|---|---|---|---|---|---|---|---|---|
| none | 1 | 0.4342 | 14.94 | 0.947 | 0.033 | 0.009 | 0.011 | False | False |
| none | 2 | 0.457 | 20.3 | 0.874 | 0.059 | 0.037 | 0.03 | False | False |
| none | 4 | 0.4778 | 24.94 | 0.781 | 0.118 | 0.06 | 0.041 | False | False |
| none | 8 | 0.5066 | 29.27 | 0.661 | 0.214 | 0.073 | 0.051 | True | False |
| none | 16 | 0.5367 | 33.04 | 0.557 | 0.301 | 0.082 | 0.06 | True | False |
| inv | 1 | 0.5637 | 39.49 | 0.538 | 0.25 | 0.123 | 0.09 | True | False |
| inv | 2 | 0.5897 | 43.68 | 0.464 | 0.3 | 0.129 | 0.107 | True | False |
| inv | 4 | 0.6142 | 50.01 | 0.369 | 0.361 | 0.132 | 0.139 | True | False |
| inv | 8 | 0.6175 | 51.45 | 0.356 | 0.36 | 0.139 | 0.145 | True | True |
| inv | 16 | 0.596 | 45.27 | 0.397 | 0.354 | 0.146 | 0.103 | True | False |
| sqrt_inv | 1 | 0.4992 | 28.29 | 0.733 | 0.131 | 0.087 | 0.049 | False | False |
| sqrt_inv | 2 | 0.5235 | 32.43 | 0.66 | 0.183 | 0.09 | 0.068 | True | False |
| sqrt_inv | 4 | 0.556 | 38.32 | 0.555 | 0.262 | 0.086 | 0.097 | True | False |
| sqrt_inv | 8 | 0.5739 | 40.89 | 0.466 | 0.35 | 0.078 | 0.106 | True | False |
| sqrt_inv | 16 | 0.5958 | 44.9 | 0.41 | 0.39 | 0.071 | 0.13 | True | False |

Flags:
- none
