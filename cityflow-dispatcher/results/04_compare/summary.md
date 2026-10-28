# 04 compare

Command: `04_compare.py`

Inputs:
- models/yolov8*.pt, models/finetuned/
- data/manifest.csv
- data/gt_test_boxes.csv

Outputs:
- overall.csv
- per_camera.csv
- ideal_split.csv
- ideal_split.png

Ideal-label split (fraction of frames per cheapest correct model):

| split | pool | yolov8n | yolov8s | yolov8m | yolov8l |
|---|---|---|---|---|---|
| test | original | 0.1985 | 0.3755 | 0.2256 | 0.2004 |
| test | fine-tuned | 0.5179 | 0.3101 | 0.1122 | 0.0598 |
| val | original | 0.1217 | 0.3241 | 0.2801 | 0.274 |
| val | fine-tuned | 0.4202 | 0.3209 | 0.0536 | 0.2053 |

Key numbers:

| split | variant | model | recall_vs_l | precision_vs_l | mAP50 | mAP50_95 | frame_correct | gt_recall |
|---|---|---|---|---|---|---|---|---|
| test | orig | yolov8n | 0.5996 | 0.7207 | 0.3737 | 0.2679 | 0.1985 | 0.7513 |
| test | orig | yolov8s | 0.7886 | 0.7629 | 0.5601 | 0.4122 | 0.5573 | 0.7539 |
| test | orig | yolov8m | 0.8526 | 0.824 | 0.6495 | 0.54 | 0.7213 | 0.7222 |
| test | ft | yolov8n | 0.7801 | 0.6281 | 0.4337 | 0.3176 | 0.5179 | 0.8035 |
| test | ft | yolov8s | 0.8663 | 0.6693 | 0.4959 | 0.3785 | 0.7799 | 0.7766 |
| test | ft | yolov8m | 0.9004 | 0.7102 | 0.5819 | 0.4733 | 0.8674 | 0.7566 |
| test | orig | yolov8l | nan | nan | nan | nan | nan | 0.7697 |
| val | orig | yolov8n | 0.5727 | 0.7825 | 0.336 | 0.2494 | 0.1217 | nan |
| val | orig | yolov8s | 0.7653 | 0.8092 | 0.4653 | 0.3748 | 0.4405 | nan |
| val | orig | yolov8m | 0.8428 | 0.8553 | 0.6105 | 0.5245 | 0.6832 | nan |
| val | ft | yolov8n | 0.7442 | 0.7053 | 0.3226 | 0.2267 | 0.4202 | nan |
| val | ft | yolov8s | 0.842 | 0.6936 | 0.3876 | 0.2967 | 0.6974 | nan |
| val | ft | yolov8m | 0.8102 | 0.8103 | 0.4511 | 0.3717 | 0.5832 | nan |

Flags:
- gt_recall is class-agnostic over car/motorcycle/bus/truck, micro-averaged; GT is incomplete.
