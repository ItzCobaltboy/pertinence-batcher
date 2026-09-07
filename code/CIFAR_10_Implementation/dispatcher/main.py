"""
Dispatcher NSGA-II search — single entry point.

This folder's only responsibility is finding the Pareto front: an NSGA-II
search over 12-gene penalty-matrix chromosomes. Evaluating a front (train +
val alpha_sys, per-config summaries) is a separate pipeline,
code/dispatcher_analysis/, which reads this folder's pareto_front.csv and
saved model weights but shares no code. Keep the two responsibilities split.

Objectives (both minimised): alpha_sys_loss = 1 - alpha_sys (Eq. 3 of the
PERTINENCE paper — does the DISPATCHED model actually classify correctly)
and avg_flops_G. Class imbalance is handled with INS (Inverse Number of
Samples) weighting applied inside the loss function: every image is seen
once per epoch at its natural frequency, and a minority-class image's loss
counts for more when the model gets it wrong.

The NSGA-II algorithm itself (selection, crossover, mutation, non-dominated
sorting) is pymoo's; only the domain logic (what a chromosome means, how to
train+evaluate one) is ours.

Folder layout:
  main.py                  <- entry point, run this file
  src/
    constants.py              paths + hyperparameters, shared by every module
    embeddings.py               computes/caches resnet20 embeddings (train only)
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
  embeddings_cache/         cached resnet20 embeddings — train only
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
