"""
INS (Inverse Number of Samples) class weighting, applied in the loss rather
than a sampler. Kept identical to code/Dispatcher/src/class_weights.py so
retraining here reproduces the same weighting the search used.
"""

import numpy as np

import constants as c


def compute_ins_class_weights(labels):
    """weight(class) = 1 / count(class), normalized so the weights sum to
    NUM_CLASSES. Returns a (NUM_CLASSES,) array."""
    class_counts = [0] * c.NUM_CLASSES
    for label in labels:
        class_counts[label] += 1

    raw_weights = []
    for count in class_counts:
        raw_weights.append(1.0 / count if count > 0 else 0.0)

    total_weight = sum(raw_weights)
    normalized_weights = [w / total_weight * c.NUM_CLASSES for w in raw_weights]

    return np.array(normalized_weights, dtype=np.float32)
