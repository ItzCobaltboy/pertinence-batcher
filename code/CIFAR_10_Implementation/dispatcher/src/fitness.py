"""
Fitness evaluation: turns one chromosome into the two NSGA-II objectives.

  obj1 = alpha_sys_loss = 1 - alpha_sys   (system accuracy loss)
  obj2 = avg_flops_G                      (average compute cost of wherever images got routed;
                                            variable/column name kept as avg_flops_G for continuity
                                            with the ImageNette pipeline, but for this CIFAR-10
                                            reproduction the underlying units are MAdds-M, not GFLOPs
                                            — see c.MADDS_M in constants.py)

Both are minimised.

alpha_sys is the paper's accuracy definition (Eq. 3): the fraction of images
where the DISPATCHED model itself classifies correctly — looked up directly
from correctness_matrix[image, predicted_model], not compared against the
"ideal" argmin-cheapest-correct label. Overestimating to a bigger-but-still-
correct model costs only FLOPs (obj2), not accuracy (obj1), except in the
~9.7% of images where correctness is non-monotonic across the model pool (a
bigger model happens to fail where a smaller one succeeded).
"""

import numpy as np

import constants as c
from penalty_matrix import build_penalty_matrix
from dispatcher_model import train_fc, predict


def evaluate_individual(chromosome, train_embeddings, train_labels, correctness_matrix,
                         class_weights, device):
    """Trains an FC head for this chromosome's penalty matrix, predicts on
    the train set, and returns (alpha_sys_loss, avg_flops_G)."""
    penalty_matrix = build_penalty_matrix(chromosome)

    W, b = train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device)
    predictions = predict(train_embeddings, W, b)

    flops_array = np.array(c.MADDS_M)
    avg_flops_G = flops_array[predictions].mean()

    image_indices = np.arange(len(predictions))
    alpha_sys = correctness_matrix[image_indices, predictions].mean()
    alpha_sys_loss = 1.0 - alpha_sys

    return float(alpha_sys_loss), float(avg_flops_G)
