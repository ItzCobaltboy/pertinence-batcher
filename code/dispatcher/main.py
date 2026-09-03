"""
Dispatcher NSGA-II search — single entry point.

This folder's ONLY responsibility is finding the Pareto front (the NSGA-II
search over penalty-matrix chromosomes). Evaluating that front (train + val
alpha_sys, per-config summaries) is a separate, standalone pipeline —
code/dispatcher_analysis/ — which reads this folder's pareto_front.csv and
saved model weights but does not live here. Don't add eval logic back into
this folder; keep the two pipelines' responsibilities split.

Evolves the penalty matrix (12 genes) to trace the Pareto front of
alpha_sys (Eq. 3 of the PERTINENCE paper: does the DISPATCHED model actually
classify correctly) vs avg-FLOPs. Class imbalance is handled with INS
(Inverse Number of Samples) weighting applied directly in the loss function
— every image is seen once per epoch at its natural frequency, and a
minority-class image's loss counts for more when the model gets it wrong.

This replaces two earlier versions of this search, both in git history:
  - the original, which applied the INS formula to a WeightedRandomSampler
    instead of the loss (a different, incorrect technique)
  - a second run which used exact-match accuracy against the ideal
    argmin-cheapest-correct label as the objective, instead of the paper's
    real alpha_sys — see Journel/step1.md for why that was wrong (it punishes
    overestimation exactly as hard as underestimation, which the paper does
    not do). That run's raw output is archived under
    code/dispatcher_analysis/ alongside this run's, for reference.

The actual NSGA-II algorithm (selection, crossover, mutation, non-dominated
sorting) is pymoo's, not hand-rolled — only the domain logic (what a
chromosome means, how to train+evaluate one) is ours.

Folder layout:
  main.py                  <- you are here, run this file
  src/
    constants.py              paths + hyperparameters, shared by every module
    embeddings.py               computes/caches ResNet18 embeddings (train only)
    penalty_matrix.py             chromosome -> 4x4 penalty matrix
    class_weights.py                INS class weighting (for the loss)
    loss.py                           the penalized loss function itself
    dispatcher_model.py                 trains one FC head, predicts with it
    fitness.py                            chromosome -> (alpha_sys_loss, avg_flops_G)
    dispatcher_problem.py                   wraps fitness.py into a pymoo Problem
    progress_logger.py                        pymoo Callback: logging + checkpoints
    save_results.py                             writes the final Pareto front CSV
    save_models.py                                saves trained FC weights per individual
    logging_setup.py                                live progress logging
    nsga2_search.py                                   sets up + runs pymoo's NSGA2

  data/                   train ground truth (image_path, label) — train only
  embeddings_cache/         cached ResNet18 embeddings — train only
  results/
    logs/                     live run logs
    nsga2/                      pareto_front.csv + checkpoint_gen*.npz + models/
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)

from nsga2_search import run_nsga2


if __name__ == "__main__":
    run_nsga2()
    print("\nDone.")
