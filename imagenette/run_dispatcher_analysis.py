"""
ImageNette dispatcher_analysis — Pareto-front evaluation entry point.

Thin: only wires this track's config (config.py, same directory) into the
shared evaluation code (../dispatcher_analysis/). All actual evaluation
logic lives there.

Four phases:
  PHASE 0 (build_models.py)      retrain every Pareto individual's FC head
                                  directly from its chromosome — never a
                                  copied cache.
  PHASE 1 (cache_predictions.py) predict on train + val embeddings, cache
                                  every prediction to disk.
  PHASE 2 (summarize.py)         alpha_sys, accuracy_exact_match_vs_ideal,
                                  avg_model_cost, per-class recall/precision
                                  per individual, per split.
  PHASE 3 (plots.py)             Pareto scatter + confusion matrices.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dispatcher_analysis"))

import torch

import config
from embeddings import load_train_embeddings, load_val_embeddings
from build_models import build_models
from cache_predictions import cache_predictions
from summarize import summarize
from plots import plot_pareto_scatter, plot_confusion_matrices


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_embeddings, train_labels = load_train_embeddings(device, config)
    val_embeddings, _ = load_val_embeddings(device, config)

    build_models(train_embeddings, train_labels, device, config)

    train_predictions_df, val_predictions_df = cache_predictions(train_embeddings, val_embeddings, config)
    train_summary, val_summary = summarize(train_predictions_df, val_predictions_df, config)

    plot_pareto_scatter(train_summary, val_summary,
                         os.path.join(config.PLOTS_DIR, "pareto_scatter.png"), config)

    # pick 3 representative val configs for the confusion-matrix figure
    val_by_alpha = val_summary.sort_values("alpha_sys", ascending=False)
    val_by_cost = val_summary.sort_values("avg_model_cost")
    highest_alpha_sys = int(val_by_alpha.iloc[0]["individual"])
    cheapest = int(val_by_cost.iloc[0]["individual"])
    middle = int(val_by_alpha.iloc[len(val_by_alpha) // 2]["individual"])
    showcase_ids = sorted(set([cheapest, middle, highest_alpha_sys]),
                           key=lambda i: val_summary.set_index("individual").loc[i, "avg_model_cost"])

    plot_confusion_matrices(val_predictions_df, val_summary, showcase_ids,
                             os.path.join(config.PLOTS_DIR, "confusion_matrices.png"), config)

    print("\nDone.")
