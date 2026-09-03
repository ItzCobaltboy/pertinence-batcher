"""
Dispatcher Pareto-front evaluation — single entry point.

Standalone from code/Dispatcher/, which owns only the NSGA-II search itself
(finding the Pareto front). This pipeline's job is evaluating that front:
for every individual, alpha_sys (Eq. 3 of the PERTINENCE paper — is the
model the dispatcher actually picked correct, not "did it match the ideal
label") and avg_flops_G, on both the train set and a held-out val set.

No retraining: code/Dispatcher already saved the actual trained FC weights
per Pareto individual (model_cache/individual_<id>.npz, copied over from
there), so this just loads and predicts.

Two phases:
  PHASE 1 (src/cache_predictions.py)
    Load each individual's saved weights, predict on train + val embeddings,
    cache every prediction to predictions/{train,val}_predictions.csv.

  PHASE 2 (src/summarize.py)
    Re-read the cached predictions and compute alpha_sys + avg_flops_G per
    individual, per split. Saved to results/{train,val}_summary.csv.

Deeper analysis (per-class recall/precision, confusion matrices, plots) is
not built yet — deliberately stopped after caching predictions + summaries,
per the current scope. The cached predictions carry everything needed to
add that later without re-predicting.

Folder layout:
  main.py                    <- you are here, run this file
  src/
    constants.py               paths + pool constants, shared by every module
    embeddings.py                loads cached ResNet18 embeddings (train + val)
    cache_predictions.py           phase 1 driver
    summarize.py                     phase 2 driver

  data/                     pareto_front.csv, train/val ground truth (copied
                             from code/Dispatcher — same underlying data, no
                             recomputation)
  embeddings_cache/           cached ResNet18 embeddings (gitignored)
  model_cache/                  trained FC weights per Pareto individual
                                 (copied from code/Dispatcher's own
                                 results/nsga2/models/)
  predictions/                    cached raw predictions, one column per
                                   individual
  results/                          alpha_sys + avg_flops_G summary per
                                     individual, per split
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)

import torch

from embeddings import load_train_embeddings, load_val_embeddings
from cache_predictions import cache_predictions
from summarize import summarize


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_embeddings, _ = load_train_embeddings(device)
    val_embeddings, _ = load_val_embeddings(device)

    train_predictions_df, val_predictions_df = cache_predictions(train_embeddings, val_embeddings)
    train_summary, val_summary = summarize(train_predictions_df, val_predictions_df)

    print("\nDone.")
