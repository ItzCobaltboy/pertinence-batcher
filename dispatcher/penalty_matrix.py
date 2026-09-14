"""
Turns a chromosome's penalty genes into the NUM_CLASSES x NUM_CLASSES
penalty matrix used by the loss.
"""

import numpy as np


def build_penalty_matrix(chromosome, config):
    """
    Chromosome layout: the first (config.NUM_CLASSES^2 - config.NUM_CLASSES)
    genes are the off-diagonal entries of the NUM_CLASSES x NUM_CLASSES
    penalty matrix, filled in row-major order (row = true class, column =
    predicted class). Diagonal is always 0 — a correct prediction is never
    penalized. Tracks that also search the weighting scheme (see
    weighting_scheme.py) carry one extra gene after these; this function
    only ever reads the penalty genes, so it doesn't care whether that
    extra gene is present.
    """
    penalty_matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.float32)

    gene_index = 0
    for true_class in range(config.NUM_CLASSES):
        for pred_class in range(config.NUM_CLASSES):
            if true_class != pred_class:
                penalty_matrix[true_class, pred_class] = chromosome[gene_index]
                gene_index += 1

    return penalty_matrix
