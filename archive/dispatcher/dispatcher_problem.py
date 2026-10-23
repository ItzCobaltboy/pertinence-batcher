"""
Wraps the fitness function into a pymoo Problem so pymoo's NSGA2
implementation can search it.
"""

import numpy as np
from pymoo.core.problem import Problem

from fitness import evaluate_individual
from weighting_scheme import searches_weighting_scheme, num_penalty_genes


def _gene_bounds(config):
    """Returns (xl, xu) for pymoo — a scalar pair if every gene shares the
    same penalty range (ImageNette/CIFAR-10's historical, INS-only
    chromosomes), or a per-gene array if the last gene is the
    weighting-scheme selector (see weighting_scheme.py), since that gene
    needs its own, different range from the penalty genes."""
    if not searches_weighting_scheme(config):
        return config.PENALTY_LOWER_BOUND, config.PENALTY_UPPER_BOUND

    penalty_genes = num_penalty_genes(config)
    xl = np.full(config.N_GENES, config.PENALTY_LOWER_BOUND, dtype=np.float64)
    xu = np.full(config.N_GENES, config.PENALTY_UPPER_BOUND, dtype=np.float64)
    xl[penalty_genes] = config.SCHEME_GENE_LOWER_BOUND
    xu[penalty_genes] = config.SCHEME_GENE_UPPER_BOUND
    return xl, xu


class DispatcherProblem(Problem):
    """An N_GENES-gene, 2-objective (alpha_sys_loss, avg_model_cost)
    problem, evaluated by training a real FC head per individual. N_GENES
    is either the penalty-only count or that plus one weighting-scheme
    gene — see weighting_scheme.py."""

    def __init__(self, train_embeddings, train_labels, val_embeddings, val_correctness_matrix,
                 device, logger, config):
        xl, xu = _gene_bounds(config)
        super().__init__(
            n_var=config.N_GENES,
            n_obj=2,
            xl=xl,
            xu=xu,
        )
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_correctness_matrix = val_correctness_matrix
        self.device = device
        self.logger = logger
        self.config = config
        self.total_evaluated = 0
        # every (chromosome, objectives) pair ever evaluated, in order — the
        # paper builds its reported front from ALL evaluated solutions, not
        # just the final population (see non_dominated_indices below)
        self.archive_X = []
        self.archive_F = []

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
                self.device, self.config,
            )
            objectives[i, 0] = alpha_sys_loss
            objectives[i, 1] = avg_model_cost

            self.total_evaluated += 1
            self.logger.info(f"  [eval #{self.total_evaluated:>4}]  "
                              f"alpha_sys={100*(1-alpha_sys_loss):.2f}%  "
                              f"cost={avg_model_cost:.3f} {self.config.MODEL_COST_UNIT}")

        out["F"] = objectives
        self.archive_X.append(np.array(X, dtype=np.float64, copy=True))
        self.archive_F.append(objectives.copy())

    def archive(self):
        """Returns (X_all, F_all) over every evaluation so far."""
        return np.concatenate(self.archive_X, axis=0), np.concatenate(self.archive_F, axis=0)


def non_dominated_indices(F):
    """Indices of the non-dominated rows of F (both columns minimized), one
    index per distinct objective vector. Matches the paper's reporting: "at
    the end of the run, all evaluated solutions are ranked and the subset of
    non-dominated solutions ... are retained to form the reported Pareto
    front". The final population alone can miss some of these: NSGA-II's
    crowding truncation drops non-dominated points once the front outgrows
    the population."""
    _, first = np.unique(F, axis=0, return_index=True)
    first = np.sort(first)
    candidates = F[first]
    keep = []
    for i, f in enumerate(candidates):
        dominated = np.any(np.all(candidates <= f, axis=1) & np.any(candidates < f, axis=1))
        if not dominated:
            keep.append(first[i])
    return np.array(keep, dtype=int)
