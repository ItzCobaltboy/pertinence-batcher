"""
Turns a 12-gene chromosome into the 4x4 penalty matrix used by the loss.
"""

import numpy as np


def build_penalty_matrix(chromosome, config):
    """
    Chromosome layout: 12 genes = the off-diagonal entries of the 4x4
    penalty matrix, filled in row-major order (row = true class, column =
    predicted class). Diagonal is always 0 — a correct prediction is never
    penalized.
    """
    penalty_matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.float32)

    gene_index = 0
    for true_class in range(config.NUM_CLASSES):
        for pred_class in range(config.NUM_CLASSES):
            if true_class != pred_class:
                penalty_matrix[true_class, pred_class] = chromosome[gene_index]
                gene_index += 1

    return penalty_matrix
