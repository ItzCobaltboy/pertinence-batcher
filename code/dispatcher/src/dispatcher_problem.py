"""
Wraps our fitness function into a pymoo Problem, so pymoo's NSGA2
implementation can search it.

pymoo hands us a whole batch of chromosomes at once (the current population
or generation of offspring) via the X argument — we just loop over them and
call our own evaluate_individual for each one, since each evaluation trains
a real FC head on the GPU and can't be vectorized.
"""

import numpy as np
from pymoo.core.problem import Problem

import constants as c
from fitness import evaluate_individual


class DispatcherProblem(Problem):
    def __init__(self, train_embeddings, train_labels, correctness_matrix, class_weights, device, logger):
        super().__init__(
            n_var=c.N_GENES,
            n_obj=2,
            xl=c.PENALTY_LOWER_BOUND,
            xu=c.PENALTY_UPPER_BOUND,
        )
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.correctness_matrix = correctness_matrix
        self.class_weights = class_weights
        self.device = device
        self.logger = logger
        self.total_evaluated = 0

    def _evaluate(self, X, out, *args, **kwargs):
        num_individuals = X.shape[0]
        objectives = np.zeros((num_individuals, 2))

        for i in range(num_individuals):
            chromosome = X[i]
            alpha_sys_loss, avg_flops_G = evaluate_individual(
                chromosome, self.train_embeddings, self.train_labels, self.correctness_matrix,
                self.class_weights, self.device,
            )
            objectives[i, 0] = alpha_sys_loss
            objectives[i, 1] = avg_flops_G

            self.total_evaluated += 1
            self.logger.info(f"  [eval #{self.total_evaluated:>4}]  "
                              f"alpha_sys={100*(1-alpha_sys_loss):.2f}%  flops={avg_flops_G:.3f}G")

        out["F"] = objectives
