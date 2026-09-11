"""
Fitness evaluation: turns one chromosome into the two NSGA-II objectives.

  obj1 = alpha_sys_loss = 1 - alpha_sys   (system accuracy loss)
  obj2 = avg_model_cost                   (average compute cost of wherever
                                            images got routed — units are
                                            dataset-specific, see
                                            config.MODEL_COST_UNIT)

Both are minimised.

alpha_sys is the paper's accuracy definition (Eq. 3): the fraction of images
where the DISPATCHED model itself classifies correctly — looked up directly
from correctness_matrix[image, predicted_model], not compared against the
"ideal" argmin-cheapest-correct label. Overestimating to a bigger-but-still-
correct model costs only compute (obj2), not accuracy (obj1).

The FC head is trained on the TRAIN set, but fitness itself (both alpha_sys
and avg_model_cost) is measured by predicting on the held-out VAL set — this
matches the paper's own methodology ("we perform the fitness evaluation for
each individual on the test set"), which evaluates every individual's
fitness on held-out data rather than the data it was just trained on.
"""

import numpy as np

from penalty_matrix import build_penalty_matrix
from dispatcher_model import train_fc, predict


def evaluate_individual(chromosome, train_embeddings, train_labels, val_embeddings,
                         val_correctness_matrix, class_weights, device, config):
    """Trains an FC head for this chromosome's penalty matrix on the train
    set, predicts on the held-out val set, and returns (alpha_sys_loss,
    avg_model_cost) computed from those val predictions."""
    penalty_matrix = build_penalty_matrix(chromosome, config)

    W, b = train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device, config)
    predictions = predict(val_embeddings, W, b)

    cost_array = np.array(config.MODEL_COST)
    avg_model_cost = cost_array[predictions].mean()

    image_indices = np.arange(len(predictions))
    alpha_sys = val_correctness_matrix[image_indices, predictions].mean()
    alpha_sys_loss = 1.0 - alpha_sys

    return float(alpha_sys_loss), float(avg_model_cost)
