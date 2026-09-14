"""
Sets up and runs the NSGA-II search using pymoo — selection, crossover,
mutation, and non-dominated sorting are pymoo's implementations. Only the
domain logic (what a chromosome means, how to train+evaluate one) is ours.
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

from embeddings import load_or_compute_train_embeddings, load_or_compute_val_embeddings
from weighting_scheme import searches_weighting_scheme
from dispatcher_problem import DispatcherProblem
from progress_logger import ProgressLogger
from save_results import save_pareto_front
from save_models import save_pareto_models
from logging_setup import setup_logging


def run_nsga2(config):
    """End-to-end search: load embeddings, build the pymoo problem, run
    NSGA2 for config.GENERATIONS generations, then save the resulting
    Pareto front's chromosomes and trained FC weights."""
    logger = setup_logging(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bias_handling = ("chromosome-selected INS/ISNS/ENS class-weighted LOSS"
                      if searches_weighting_scheme(config) else
                      "INS class-weighted LOSS (not sampling)")
    logger.info(f"Device: {device}  |  Pop: {config.POPULATION_SIZE}  |  "
                f"Gen: {config.GENERATIONS}  |  FC epochs: {config.FC_EPOCHS}  |  "
                f"Bias handling: {bias_handling}  |  "
                f"GA engine: pymoo")

    logger.info("Loading train + val embeddings...")
    train_embeddings, train_labels = load_or_compute_train_embeddings(device, config)
    val_embeddings, _ = load_or_compute_val_embeddings(device, config)
    logger.info(f"Train embeddings shape: {train_embeddings.shape}  |  "
                f"Val embeddings shape: {val_embeddings.shape}")

    # Per-image, per-model correctness on the held-out VAL set (row order
    # matches val_ground_truth.csv, same order val embeddings were computed
    # in) — used to compute alpha_sys (Eq. 3 of the paper). Fitness is
    # evaluated on val, not train, per the paper's own methodology ("we
    # perform the fitness evaluation for each individual on the test set")
    # — see fitness.py's module docstring.
    correctness_columns = [f"{name}_correct" for name in config.MODEL_NAMES]
    val_correctness_matrix = pd.read_csv(config.VAL_GROUND_TRUTH_CSV)[correctness_columns].values.astype(bool)
    logger.info(f"Val correctness matrix shape: {val_correctness_matrix.shape}")

    # class weights are no longer computed once upfront — if this track
    # searches the weighting scheme (config.N_GENES includes the extra
    # gene, see weighting_scheme.py), each individual's own chromosome
    # picks INS/ISNS/ENS and fitness.py computes that individual's weights
    # fresh. Tracks that don't search the scheme still get plain INS,
    # computed the same way it always was, just inside fitness.py now
    # instead of once here.
    problem = DispatcherProblem(train_embeddings, train_labels, val_embeddings, val_correctness_matrix,
                                 device, logger, config)

    algorithm = NSGA2(
        pop_size=config.POPULATION_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config.SBX_CROSSOVER_PROBABILITY, eta=config.SBX_ETA),
        mutation=PM(eta=config.MUTATION_ETA),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", config.GENERATIONS)

    logger.info(f"\nStarting search: {config.POPULATION_SIZE} initial individuals, "
                f"then {config.GENERATIONS} generations of {config.POPULATION_SIZE} offspring each...\n")
    start_time = time.time()

    result = minimize(
        problem,
        algorithm,
        termination,
        callback=ProgressLogger(logger, config),
        save_history=False,
        verbose=False,   # our own per-individual/per-generation logging replaces pymoo's
    )

    logger.info(f"\nSearch finished in {time.time()-start_time:.0f}s")
    logger.info(f"Final Pareto front: {len(result.X)} individuals")

    save_pareto_front(result.X, result.F, logger, config)
    save_pareto_models(result.X, train_embeddings, train_labels, device, logger, config)
