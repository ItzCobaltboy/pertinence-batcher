# 03 finetune

Command: `03_finetune.py`

Inputs:
- data/images, data/labels (pseudo-labels from yolov8l)

Outputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\finetuned\yolov8n_cityflow.pt
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\finetuned\yolov8s_cityflow.pt
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\finetuned\yolov8m_cityflow.pt
- yolov8*_curve.csv
- timing.csv

Key numbers:

| model | epochs | batch | lr0 | total_min | min_per_epoch |
|---|---|---|---|---|---|
| yolov8n | 10 | 16 | 0.0001 | 71.8 | 7.18 |
| yolov8s | 10 | 16 | 0.0001 | 63.7 | 6.37 |
| yolov8m | 10 | 16 | 0.0001 | 67.1 | 6.71 |

Flags:
- optimizer set to AdamW (ultralytics 'auto' would ignore lr0=0.0001); seed 0
