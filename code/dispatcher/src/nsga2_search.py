"""
Sets up and runs the NSGA-II search using pymoo (not a hand-rolled GA) —
selection, crossover, mutation, and non-dominated sorting are all pymoo's
well-tested implementations. Only the domain logic (what a chromosome means,
how to train+evaluate one) is ours.
"""

import time
import torch
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

import constants as c
from embeddings import load_or_compute_train_embeddings
from class_weights import compute_ins_class_weights
from dispatcher_problem import DispatcherProblem
from progress_logger import ProgressLogger
from save_results import save_pareto_front
from save_models import save_pareto_models
from logging_setup import setup_logging


def run_nsga2():
    logger = setup_logging()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Pop: {c.POPULATION_SIZE}  |  "
                f"Gen: {c.GENERATIONS}  |  FC epochs: {c.FC_EPOCHS}  |  "
                f"Bias handling: INS class-weighted LOSS (not sampling)  |  "
                f"GA engine: pymoo")

    logger.info("Loading train embeddings...")
    train_embeddings, train_labels = load_or_compute_train_embeddings(device)
    logger.info(f"Embeddings shape: {train_embeddings.shape}")

    # Per-image, per-model correctness (row order matches train_ground_truth.csv,
    # which is also the order embeddings were computed in) — used to compute the
    # real alpha_sys (Eq. 3 of the paper): was the DISPATCHED model actually
    # correct on this image, not "did we match the ideal argmin label."
    correctness_columns = [f"{name}_correct" for name in c.MODEL_NAMES]
    correctness_matrix = pd.read_csv(c.TRAIN_GROUND_TRUTH_CSV)[correctness_columns].values.astype(bool)
    logger.info(f"Correctness matrix shape: {correctness_matrix.shape}")

    class_weights = compute_ins_class_weights(train_labels)
    logger.info(f"INS class weights: {class_weights}")

    problem = DispatcherProblem(train_embeddings, train_labels, correctness_matrix,
                                 class_weights, device, logger)

    algorithm = NSGA2(
        pop_size=c.POPULATION_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=c.SBX_CROSSOVER_PROBABILITY, eta=c.SBX_ETA),
        mutation=PM(eta=c.MUTATION_ETA),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", c.GENERATIONS)

    logger.info(f"\nStarting search: {c.POPULATION_SIZE} initial individuals, "
                f"then {c.GENERATIONS} generations of {c.POPULATION_SIZE} offspring each...\n")
    start_time = time.time()

    result = minimize(
        problem,
        algorithm,
        termination,
        callback=ProgressLogger(logger),
        save_history=False,
        verbose=False,   # our own logging (per-individual + per-generation) replaces pymoo's
    )

    logger.info(f"\nSearch finished in {time.time()-start_time:.0f}s")
    logger.info(f"Final Pareto front: {len(result.X)} individuals")

    save_pareto_front(result.X, result.F, logger)
    save_pareto_models(result.X, train_embeddings, train_labels, class_weights, device, logger)
