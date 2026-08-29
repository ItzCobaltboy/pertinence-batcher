"""
Turns one row of pareto_front.csv into the pieces needed for training:
a 4x4 penalty matrix, and per-image sample weights for class balancing.
"""

import numpy as np

import constants as c


def build_penalty_matrix(pareto_row):
    """
    Builds the 4x4 penalty matrix from one row of pareto_front.csv.
    Diagonal is always 0 (no penalty for a correct prediction).
    Row = true class, column = predicted class.
    """
    penalty_matrix = np.zeros((c.NUM_CLASSES, c.NUM_CLASSES), dtype=np.float32)
    for true_class in range(c.NUM_CLASSES):
        for pred_class in range(c.NUM_CLASSES):
            if true_class != pred_class:
                column_name = f"P_{true_class}{pred_class}"
                penalty_matrix[true_class, pred_class] = pareto_row[column_name]
    return penalty_matrix


def build_sample_weights(labels, alpha):
    """
    Computes a sampling weight for every image, so that during training
    minority classes get shown more often.

    weight(class) = 1 / count(class)^alpha

    alpha=0   -> every class weighted equally (uniform sampling)
    alpha=1   -> strong oversampling of rare classes
    """
    class_counts = [0] * c.NUM_CLASSES
    for label in labels:
        class_counts[label] += 1

    class_weights = []
    for count in class_counts:
        if count == 0:
            class_weights.append(0.0)
        else:
            class_weights.append(1.0 / (count ** alpha))

    sample_weights = np.zeros(len(labels), dtype=np.float32)
    for i, label in enumerate(labels):
        sample_weights[i] = class_weights[label]

    return sample_weights
