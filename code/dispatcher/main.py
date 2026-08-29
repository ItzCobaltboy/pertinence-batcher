"""
Dispatcher pipeline — standalone entry point.

Model pool (decided in Step 0 / Model Analysis):
  FP32 ResNet18 / ResNet34 / ResNet50 / ResNet152, ImageNette dataset.

Steps:
  1A  generate_labels  — per-image cheapest-correct-model labels on train set
  1A+ clean_labels     — drop 0000 noise images, save to results/cleaned/
  1B  (EDA)            — run eda.py after labels exist
  1C  train            — train dispatcher FC layer  (TODO)
  1D  evaluate         — confusion matrix, accuracy-FLOPs plot  (TODO)
  1E  batch_dispatch   — batched inference demo  (TODO)
"""

import os
from labeler import generate_labels
from cleaner import clean_labels
from nsga2 import run_nsga2

DATASET_PATH  = "./../dataset/"
RESULTS_PATH  = "./results/"
LABELS_CSV    = os.path.join(RESULTS_PATH, "dispatcher_labels.csv")
CLEANED_CSV   = os.path.join(RESULTS_PATH, "cleaned-data", "dispatcher_labels_clean.csv")


if __name__ == "__main__":
    # --- Step 1A: label generation ---
    generate_labels(
        dataset_path=DATASET_PATH,
        output_path=LABELS_CSV,
    )

    # --- Step 1A+: drop unlearnable noise images ---
    clean_labels(
        input_path=LABELS_CSV,
        output_path=CLEANED_CSV,
    )

    # --- Step 2: NSGA-II search over penalty matrix + weighting scheme ---
    # run_nsga2()  # uncomment to run (~2hrs on RTX 5070 Ti)
