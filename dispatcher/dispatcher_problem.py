"""
Wraps the fitness function into a pymoo Problem so pymoo's NSGA2
implementation can search it.
"""

import numpy as np
from pymoo.core.problem import Problem

from fitness import evaluate_individual


class DispatcherProblem(Problem):
    """A 12-gene, 2-objective (alpha_sys_loss, avg_model_cost) problem,
    evaluated by training a real FC head per individual."""

    def __init__(self, train_embeddings, train_labels, val_embeddings, val_correctness_matrix,
                 class_weights, device, logger, config):
        super().__init__(
            n_var=config.N_GENES,
            n_obj=2,
            xl=config.PENALTY_LOWER_BOUND,
            xu=config.PENALTY_UPPER_BOUND,
        )
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_correctness_matrix = val_correctness_matrix
        self.class_weights = class_weights
        self.device = device
        self.logger = logger
        self.config = config
        self.total_evaluated = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """pymoo hands us a whole population/offspring batch at once via X;
        each row is one chromosome. Evaluations can't be vectorized — each
        one trains a real FC head on the GPU — so we loop."""
        num_individuals = X.shape[0]
        objectives = np.zeros((num_individuals, 2))

        for i in range(num_individuals):
            chromosome = X[i]
            alpha_sys_loss, avg_model_cost = evaluate_individual(
                chromosome, self.train_embeddings, self.train_labels,
                self.val_embeddings, self.val_correctness_matrix,
                self.class_weights, self.device, self.config,
            )
            objectives[i, 0] = alpha_sys_loss
            objectives[i, 1] = avg_model_cost

            self.total_evaluated += 1
            self.logger.info(f"  [eval #{self.total_evaluated:>4}]  "
                              f"alpha_sys={100*(1-alpha_sys_loss):.2f}%  "
                              f"cost={avg_model_cost:.3f} {self.config.MODEL_COST_UNIT}")

        out["F"] = objectives
