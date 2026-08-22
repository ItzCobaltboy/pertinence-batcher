"""
Dispatcher pipeline — standalone entry point.

Model pool (decided in Step 0 / Model Analysis):
  FP32 ResNet18 / ResNet34 / ResNet50 / ResNet152, ImageNette dataset.

Steps:
  1A  generate_labels  — per-image cheapest-correct-model labels on train set
  1B  (EDA)            — run after labels exist
  1C  train            — train dispatcher FC layer  (TODO)
  1D  evaluate         — confusion matrix, accuracy-FLOPs plot  (TODO)
  1E  batch_dispatch   — batched inference demo  (TODO)
"""

import os
from labeler import generate_labels

DATASET_PATH  = "./../dataset/"
RESULTS_PATH  = "./results/"
LABELS_CSV    = os.path.join(RESULTS_PATH, "dispatcher_labels.csv")


if __name__ == "__main__":
    # --- Step 1A: label generation ---
    generate_labels(
        dataset_path=DATASET_PATH,
        output_path=LABELS_CSV,
    )
