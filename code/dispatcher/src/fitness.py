"""
Fitness evaluation: turns one chromosome into the two NSGA-II objectives.

  obj1 = alpha_sys_loss = 1 - alpha_sys   (system accuracy loss)
  obj2 = avg_flops_G                      (average compute cost of wherever images got routed)

Both are minimised.

alpha_sys is the paper's actual accuracy definition (Eq. 3): the fraction of
images where the DISPATCHED model itself classifies correctly — looked up
directly from correctness_matrix[image, predicted_model], not compared
against the "ideal" argmin-cheapest-correct label.

This replaces an earlier, incorrect definition (exact-match against the
ideal label) that penalized overestimation — routing to a bigger model that
was still correct — exactly as hard as underestimation. That's NOT what the
paper does: overestimation only costs FLOPs (obj2), not accuracy (obj1),
except in the ~9.7% of images where model correctness is non-monotonic (a
bigger model happens to fail where a smaller one succeeded) — measured
directly from this project's own ground truth, see Journel/step1.md.
"""

import numpy as np

import constants as c
from penalty_matrix import build_penalty_matrix
from dispatcher_model import train_fc, predict


def evaluate_individual(chromosome, train_embeddings, train_labels, correctness_matrix,
                         class_weights, device):
    penalty_matrix = build_penalty_matrix(chromosome)

    W, b = train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device)
    predictions = predict(train_embeddings, W, b)

    flops_array = np.array(c.FLOPS_G)
    avg_flops_G = flops_array[predictions].mean()

    image_indices = np.arange(len(predictions))
    alpha_sys = correctness_matrix[image_indices, predictions].mean()
    alpha_sys_loss = 1.0 - alpha_sys

    return float(alpha_sys_loss), float(avg_flops_G)
