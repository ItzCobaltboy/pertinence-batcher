# 03 finetune

Command: `03_finetune.py --smoke`

Inputs:
- data/images, data/labels (pseudo-labels from yolov8l)

Outputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\finetuned\smoke\yolov8n_cityflow.pt
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\finetuned\smoke\yolov8s_cityflow.pt
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\models\finetuned\smoke\yolov8m_cityflow.pt
- yolov8*_curve.csv
- timing.csv

Key numbers:

| model | epochs | batch | lr0 | total_min | min_per_epoch |
|---|---|---|---|---|---|
| yolov8n | 1 | 16 | 0.0001 | 2.4 | 2.38 |
| yolov8s | 1 | 16 | 0.0001 | 0.7 | 0.73 |
| yolov8m | 1 | 16 | 0.0001 | 0.8 | 0.81 |

Flags:
- optimizer set to AdamW (ultralytics 'auto' would ignore lr0=0.0001); seed 0
