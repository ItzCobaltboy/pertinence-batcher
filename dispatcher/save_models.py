"""
Trains and saves the actual FC weights for every individual on the final
Pareto front, not just their chromosomes — so using any of these configs
later (e.g. in dispatcher_analysis) doesn't require retraining from scratch.

Note: this retrains each individual one more time after the search ends.
Random init + shuffle mean these weights won't be bit-identical to whatever
pymoo saw during the search, but with the same chromosome, hyperparameters,
and class weights, results should be very close.
"""

import os
import numpy as np

from penalty_matrix import build_penalty_matrix
from dispatcher_model import train_fc


def save_pareto_models(chromosomes, train_embeddings, train_labels, class_weights, device,
                        logger, config):
    """Retrains an FC head for every Pareto chromosome and saves (W, b,
    chromosome) to results/nsga2/models/individual_<id>.npz."""
    models_dir = os.path.join(config.NSGA2_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    logger.info(f"\nTraining + saving final weights for {len(chromosomes)} Pareto individuals...")
    for individual_id, chromosome in enumerate(chromosomes):
        penalty_matrix = build_penalty_matrix(chromosome, config)
        W, b = train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device, config)

        path = os.path.join(models_dir, f"individual_{individual_id}.npz")
        np.savez(path, W=W, b=b, chromosome=chromosome)

        if (individual_id + 1) % 10 == 0 or individual_id == len(chromosomes) - 1:
            logger.info(f"  saved {individual_id + 1}/{len(chromosomes)} models")

    logger.info(f"All Pareto models saved -> {models_dir}")
