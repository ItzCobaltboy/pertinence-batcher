# 02 make dataset

Command: `02_make_dataset.py`

Inputs:
- data/cityflow (videos)
- models/yolov8l.pt

Outputs:
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\data\manifest.csv
- D:\VSCode\pertinence-batcher\cityflow-dispatcher\data\cityflow.yaml
- per_camera.csv
- class_histogram.csv

Cameras are never shared between splits: scenarios are disjoint, so no camera is in two splits.
Class histogram of pseudo-labels: car=257885, truck=21595, person=9763, bus=974, bicycle=344, motorcycle=221
Test split: 4198 official GT boxes vs 16070 yolov8l pseudo-boxes on the same frames (overlays in gt_check/).

Key numbers:

| split | frames | boxes | empty_frac |
|---|---|---|---|
| test | 1622 | 16070 | 0.007 |
| train | 6924 | 70069 | 0.003 |
| val | 14775 | 204643 | 0.0 |

Flags:
- GT completeness: 4198 GT boxes vs 16070 yolov8l boxes on the same test frames.
