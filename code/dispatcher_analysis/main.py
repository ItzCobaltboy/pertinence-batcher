"""
Dispatcher Pareto-front evaluation — single entry point.

Standalone from code/Dispatcher/, which owns only the NSGA-II search itself
(finding the Pareto front). This pipeline's job is evaluating that front:
for every individual, alpha_sys (Eq. 3 of the PERTINENCE paper — is the
model the dispatcher actually picked correct, not "did it match the ideal
label"), avg_flops_G, and per-class recall/precision against ideal_label,
on both the train set and a held-out val set.

Four phases:
  PHASE 0 (src/build_models.py)
    Retrain every Pareto individual's FC head directly from its chromosome
    (pareto_front.csv) — never trusted as a copied cache. Cheap (~3-5s each)
    and removes any doubt after a real staleness bug was found once this
    session (stale NSGA-II checkpoints from a superseded run).

  PHASE 1 (src/cache_predictions.py)
    Predict on train + val embeddings with the freshly retrained weights,
    cache every prediction to predictions/{train,val}_predictions.csv.

  PHASE 2 (src/summarize.py)
    Re-read the cached predictions and compute alpha_sys, avg_flops_G, and
    per-class recall/precision per individual, per split. Saved to
    results/{train,val}_summary.csv.

  PHASE 3 (src/plots.py)
    Pareto scatter (1 - alpha_sys vs avg_flops_G, train + val, non-dominated
    points highlighted) and confusion matrices for a few representative
    individuals (cheapest / highest-alpha_sys / middle). Saved to
    results/plots/.

Folder layout:
  main.py                    <- you are here, run this file
  src/
    constants.py               paths + pool constants + FC hyperparameters
    embeddings.py                loads cached ResNet18 embeddings (train + val)
    penalty_matrix.py              chromosome -> 4x4 penalty matrix
    class_weights.py                 INS class weights
    loss.py                            penalized loss
    train_fc.py                         trains one FC head, predicts with it
    build_models.py                       phase 0 driver
    cache_predictions.py                    phase 1 driver
    metrics.py                                confusion matrix / recall / precision
    summarize.py                                phase 2 driver
    plots.py                                      phase 3 driver

  data/                     pareto_front.csv, train/val ground truth (copied
                             from code/Dispatcher — same underlying data, no
                             recomputation)
  embeddings_cache/           cached ResNet18 embeddings (gitignored)
  model_cache/                  retrained FC weights per Pareto individual
                                 (rebuilt from chromosomes every run)
  predictions/                    cached raw predictions, one column per
                                   individual
  results/                          alpha_sys + avg_flops_G + recall/precision
                                     summary per individual, per split
    plots/                            Pareto scatter + confusion matrices
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)

import torch

import constants as c
from embeddings import load_train_embeddings, load_val_embeddings
from build_models import build_models
from cache_predictions import cache_predictions
from summarize import summarize
from plots import plot_pareto_scatter, plot_confusion_matrices


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_embeddings, train_labels = load_train_embeddings(device)
    val_embeddings, _ = load_val_embeddings(device)

    build_models(train_embeddings, train_labels, device)

    train_predictions_df, val_predictions_df = cache_predictions(train_embeddings, val_embeddings)
    train_summary, val_summary = summarize(train_predictions_df, val_predictions_df)

    plot_pareto_scatter(train_summary, val_summary,
                         os.path.join(c.PLOTS_DIR, "pareto_scatter.png"))

    val_by_alpha = val_summary.sort_values("alpha_sys", ascending=False)
    val_by_flops = val_summary.sort_values("avg_flops_G")
    highest_alpha_sys = int(val_by_alpha.iloc[0]["individual"])
    cheapest = int(val_by_flops.iloc[0]["individual"])
    middle = int(val_by_alpha.iloc[len(val_by_alpha) // 2]["individual"])
    showcase_ids = sorted(set([cheapest, middle, highest_alpha_sys]),
                           key=lambda i: val_summary.set_index("individual").loc[i, "avg_flops_G"])

    plot_confusion_matrices(val_predictions_df, val_summary, showcase_ids,
                             os.path.join(c.PLOTS_DIR, "confusion_matrices.png"))

    print("\nDone.")
