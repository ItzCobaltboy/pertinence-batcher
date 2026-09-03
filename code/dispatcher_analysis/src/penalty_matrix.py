"""
Turns a 12-gene chromosome into the 4x4 penalty matrix used by the loss.
Exact copy of code/Dispatcher/src/penalty_matrix.py — kept in sync manually
since the two pipelines are standalone (no shared imports).
"""

import numpy as np

import constants as c


def build_penalty_matrix(chromosome):
    """
    Chromosome layout: 12 genes = the off-diagonal entries of the 4x4
    penalty matrix, filled in row-major order (row = true class, column =
    predicted class). Diagonal is always 0 — a correct prediction is never
    penalized.
    """
    penalty_matrix = np.zeros((c.NUM_CLASSES, c.NUM_CLASSES), dtype=np.float32)

    gene_index = 0
    for true_class in range(c.NUM_CLASSES):
        for pred_class in range(c.NUM_CLASSES):
            if true_class != pred_class:
                penalty_matrix[true_class, pred_class] = chromosome[gene_index]
                gene_index += 1

    return penalty_matrix
