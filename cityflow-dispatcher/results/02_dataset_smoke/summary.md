# 02 make dataset

Command: `02_make_dataset.py --smoke`

Inputs:
- data/cityflow (videos)
- models/yolov8l.pt

Outputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\data\smoke\manifest.csv
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\data\smoke\cityflow.yaml
- per_camera.csv
- class_histogram.csv

Cameras are never shared between splits: scenarios are disjoint, so no camera is in two splits.
Class histogram of pseudo-labels: car=2790, truck=364, person=45, motorcycle=4, bus=1
Test split: 368 official GT boxes vs 1497 yolov8l pseudo-boxes on the same frames (overlays in gt_check/).

Key numbers:

| split | frames | boxes | empty_frac |
|---|---|---|---|
| test | 100 | 1497 | 0.0 |
| train | 100 | 539 | 0.0 |
| val | 100 | 1168 | 0.0 |

Flags:
- GT completeness: 368 GT boxes vs 1497 yolov8l boxes on the same test frames.
