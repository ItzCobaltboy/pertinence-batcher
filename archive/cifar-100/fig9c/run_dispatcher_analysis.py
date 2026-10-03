"""
Fig. 9(c) dispatcher_analysis — Pareto-front evaluation entry point.

Thin like every other track/variant in that all the actual evaluation logic
(build_models, cache_predictions, summarize, plots) lives in the shared
../../dispatcher_analysis/ package unchanged. Not thin in one respect: this
runs that shared logic TWICE, because this variant has three splits where
the shared package's config contract only has a slot for two (TRAIN_* and a
single held-out VAL_*) — see config.py's "Split naming" comment for the
full mapping between this variant's file names and the paper's own
train/test/validation terminology.

  PASS 1 evaluates against config.VAL_GROUND_TRUTH_CSV/VAL_EMBEDDINGS_NPZ,
  which this variant's config.py points at the test-fitness split — the
  exact set NSGA-II's per-individual fitness used during the search
  (dispatcher/fitness.py). Useful as a sanity check that retraining each
  Pareto individual's FC head from its saved chromosome reproduces
  something close to the fitness the search itself measured, but it is NOT
  a genuinely held-out number — the search optimized against this same
  data, generation after generation.

  PASS 2 evaluates against config.FINAL_VAL_GROUND_TRUTH_CSV/
  FINAL_VAL_EMBEDDINGS_NPZ — a class-stratified 30% slice of the official
  CIFAR-100 test set, never touched by FC training or the MOEA (the other
  70% is what PASS 1 uses). This is the paper's actual "validation set"
  role, and PASS 2's numbers (final_val_summary.csv,
  confusion_matrices_final_val.png, pareto_scatter_final_val.png) are this
  variant's true final results — the ones to report, not PASS 1's.

The retrained FC weights (build_models, keyed only off TRAIN_* + the saved
chromosomes) are shared between both passes — there's exactly one set of
per-individual weights, evaluated on two different held-out sets, not two
separate trainings.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dispatcher_analysis"))

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
                           os.path.join(config.PLOTS_DIR, "accuracy_vs_mflops_test.png"), config,
                           title="Fig. 9(c) reproduction -- test-fitness split")
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
                           os.path.join(config.PLOTS_DIR, "accuracy_vs_mflops_final_val.png"), config,
                           title="Fig. 9(c) reproduction -- final validation split (report this one)")
    plot_confusion_matrices(final_val_predictions_df, final_val_summary,
                             _showcase_individual_ids(final_val_summary),
                             os.path.join(config.PLOTS_DIR, "confusion_matrices_final_val.png"), config)

    print("\nDone.")
