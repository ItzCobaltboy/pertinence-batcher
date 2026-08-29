"""
Dispatcher Pareto-front analysis — single entry point.

Everything is triggered from here, in two phases:

  PHASE 1 (src/train_and_predict.py)
    For each of the 50 Pareto-front configs: train a fresh dispatcher head
    on train data, predict on both train and val images, save every
    prediction to predictions/train_predictions.csv and
    predictions/val_predictions.csv.

  PHASE 2 (src/analysis.py)
    Re-read the saved prediction CSVs and compute accuracy, confusion
    matrix, recall, precision, and under/correct/over counts for every
    config, on both train and val. Save everything to results/.

Folder layout:
  main.py                 <- you are here, run this file
  src/
    constants.py           paths + hyperparameters, shared by every module
    embeddings.py           computes/caches ResNet18 embeddings
    config_utils.py          turns one pareto_front.csv row into a penalty matrix
    trainer.py                 trains one dispatcher head, predicts with it
    train_and_predict.py         phase 1 driver
    metrics.py                    confusion matrix / accuracy / recall / precision
    analysis.py                     phase 2 driver

  data/                  input ground truth (copied in beforehand)
  embeddings_cache/       cached ResNet18 embeddings (so re-runs are fast)
  predictions/             raw per-config predictions
  results/                   final summary CSVs + confusion matrices
"""

import os
import sys

# main.py lives in the project root, but the actual code lives in src/ —
# add it to the import path so "from train_and_predict import ..." below works.

from src.train_and_predict import run_training_and_prediction
from src.analysis import run_analysis

if __name__ == "__main__":
    run_training_and_prediction()
    run_analysis()
    print("\nDone.")
