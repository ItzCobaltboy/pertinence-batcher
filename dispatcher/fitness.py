"""
Fitness evaluation: turns one chromosome into the two NSGA-II objectives.

  obj1 = alpha_sys_loss = 1 - alpha_sys   (system accuracy loss)
  obj2 = avg_model_cost                   (average compute cost of wherever
                                            images got routed, PLUS the
                                            dispatcher's own fixed overhead
                                            — units are dataset-specific,
                                            see config.MODEL_COST_UNIT)

Both are minimised.

avg_model_cost includes config.DISPATCHER_OVERHEAD_COST — the feature
extractor's + FC layer's own forward-pass cost — on top of the dispatched
model's cost. This matches the paper's Eq. 4, which defines the objective
as "the average number of operations per input sample, considering all
operations performed by both the dispatcher and the selected model," and
page 7 of the paper confirms this inclusive figure is what's used to decide
Pareto dominance during the search itself, not just what gets reported
afterward. DISPATCHER_OVERHEAD_COST is the same fixed value added to every
individual's cost regardless of chromosome (same extractor, same FC head
shape, every run) — it shifts the whole front by one constant offset
without changing its shape or the ranking between individuals, but it does
mean avg_model_cost is now directly comparable to how the paper itself
reports MFLOPS, rather than undercounting relative to it.

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

class_weights are no longer a fixed input computed once before the search —
if config.N_GENES includes the extra weighting-scheme gene (see
weighting_scheme.py), each individual's own chromosome decides which of
INS/ISNS/ENS gets used for ITS training run, matching the paper's own
chromosome (which searches penalties and weighting scheme together). Tracks
that don't search the scheme (ImageNette, CIFAR-10) get "INS" back from
decode_scheme() unconditionally, so class_weights work out identical to the
old fixed-INS behavior for them.
"""

import numpy as np

from penalty_matrix import build_penalty_matrix
from class_weights import compute_class_weights
from weighting_scheme import decode_scheme
from dispatcher_model import train_fc, predict


def evaluate_individual(chromosome, train_embeddings, train_labels, val_embeddings,
                         val_correctness_matrix, device, config):
    """Trains an FC head for this chromosome's penalty matrix (and, for
    tracks that search it, this chromosome's own weighting scheme) on the
    train set, predicts on the held-out val set, and returns
    (alpha_sys_loss, avg_model_cost) computed from those val predictions."""
    penalty_matrix = build_penalty_matrix(chromosome, config)
    scheme = decode_scheme(chromosome, config)
    class_weights = compute_class_weights(train_labels, scheme, config)

    W, b = train_fc(train_embeddings, train_labels, penalty_matrix, class_weights, device, config)
    predictions = predict(val_embeddings, W, b)

    cost_array = np.array(config.MODEL_COST)
    avg_model_cost = cost_array[predictions].mean() + config.DISPATCHER_OVERHEAD_COST

    image_indices = np.arange(len(predictions))
    alpha_sys = val_correctness_matrix[image_indices, predictions].mean()
    alpha_sys_loss = 1.0 - alpha_sys

    return float(alpha_sys_loss), float(avg_model_cost)
