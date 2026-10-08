"""
YOLOv8 dispatcher evaluation, entry point. All evaluation logic lives in the
shared ../dispatcher_analysis/ package; this runs it twice (same structure as
the CIFAR-100 fig9c track):

  PASS 1: test split (paper's "test set"), the data NSGA-II fitness was
  computed on. A reproduction check (seeded retraining should give back the
  search's numbers), not a held-out result.

  PASS 2: final_val split (paper's "validation set"), untouched by FC
  training and the search. THESE are the numbers to report.

One set of retrained FC weights is evaluated on both.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dispatcher_analysis"))

import torch

import config
from embeddings import load_train_embeddings, load_val_embeddings
from build_models import build_models
from cache_predictions import cache_predictions
from summarize import summarize, correctness_matrix_from_csv
from plots import plot_pareto_scatter, plot_accuracy_vs_cost, plot_confusion_matrices


def _as_final_validation_view(config):
    """Returns a shallow copy of config with its VAL_* attributes repointed
    at the carved final-validation split (config.FINAL_VAL_*), so the
    shared dispatcher_analysis functions below — which only know about one
    held-out "val" role — run PASS 2 against the paper's actual validation
    set instead of the test set PASS 1 used. Every other attribute (model
    pool, cost, PARETO_FRONT_CSV, MODEL_CACHE_DIR, TRAIN_*, ...) passes
    through untouched, so PASS 2 evaluates the exact same retrained FC
    weights PASS 1 did, just against different held-out images."""
    view = types.SimpleNamespace(**vars(config))
    view.VAL_GROUND_TRUTH_CSV = config.FINAL_VAL_GROUND_TRUTH_CSV
    view.VAL_EMBEDDINGS_NPZ = config.FINAL_VAL_EMBEDDINGS_NPZ
    view.VAL_PREDICTIONS_CSV = config.FINAL_VAL_PREDICTIONS_CSV
    view.VAL_SUMMARY_CSV = config.FINAL_VAL_SUMMARY_CSV
    return view


def _showcase_individual_ids(summary_df):
    """Picks the cheapest, highest-alpha_sys, and one middle individual off
    a summary, for the confusion-matrix figure — same selection every other
    track/variant uses."""
    by_alpha = summary_df.sort_values("alpha_sys", ascending=False)
    by_cost = summary_df.sort_values("avg_model_cost")
    highest_alpha_sys = int(by_alpha.iloc[0]["individual"])
    cheapest = int(by_cost.iloc[0]["individual"])
    middle = int(by_alpha.iloc[len(by_alpha) // 2]["individual"])
    return sorted(set([cheapest, middle, highest_alpha_sys]),
                  key=lambda i: summary_df.set_index("individual").loc[i, "avg_model_cost"])


if __name__ == "__main__":
    if not os.path.exists(config.PARETO_FRONT_CSV):
        sys.exit(f"{config.PARETO_FRONT_CSV} missing: run run_dispatcher.py first")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_embeddings, train_labels = load_train_embeddings(device, config)

    build_models(train_embeddings, train_labels, device, config)

    print("\n" + "=" * 70)
    print("PASS 1 -- test-fitness split (paper's 'test set': the split")
    print("NSGA-II's own fitness evaluation used during the search itself)")
    print("=" * 70)
    test_embeddings, _ = load_val_embeddings(device, config)
    train_predictions_df, test_predictions_df = cache_predictions(train_embeddings, test_embeddings, config)
    train_summary, test_summary = summarize(train_predictions_df, test_predictions_df, config)

    plot_pareto_scatter(train_summary, test_summary,
                         os.path.join(config.PLOTS_DIR, "pareto_scatter_test.png"), config)
    plot_accuracy_vs_cost(test_summary, correctness_matrix_from_csv(config.VAL_GROUND_TRUTH_CSV, config),
                           os.path.join(config.PLOTS_DIR, "accuracy_vs_gflops_test.png"), config,
                           title="YOLOv8 pool -- test split (fitness)")
    plot_confusion_matrices(test_predictions_df, test_summary, _showcase_individual_ids(test_summary),
                             os.path.join(config.PLOTS_DIR, "confusion_matrices_test.png"), config)

    print("\n" + "=" * 70)
    print("PASS 2 -- final validation split (paper's 'validation set': genuinely")
    print("untouched by FC training or the MOEA until this exact point) -- THESE")
    print("are the numbers to report, not PASS 1's")
    print("=" * 70)
    final_config = _as_final_validation_view(config)
    final_val_embeddings, _ = load_val_embeddings(device, final_config)
    _, final_val_predictions_df = cache_predictions(train_embeddings, final_val_embeddings, final_config)
    _, final_val_summary = summarize(train_predictions_df, final_val_predictions_df, final_config)

    plot_pareto_scatter(train_summary, final_val_summary,
                         os.path.join(config.PLOTS_DIR, "pareto_scatter_final_val.png"), config)
    plot_accuracy_vs_cost(final_val_summary, correctness_matrix_from_csv(config.FINAL_VAL_GROUND_TRUTH_CSV, config),
                           os.path.join(config.PLOTS_DIR, "accuracy_vs_gflops_final_val.png"), config,
                           title="YOLOv8 pool -- final validation split (report this one)")
    plot_confusion_matrices(final_val_predictions_df, final_val_summary,
                             _showcase_individual_ids(final_val_summary),
                             os.path.join(config.PLOTS_DIR, "confusion_matrices_final_val.png"), config)

    print("\nDone.")
