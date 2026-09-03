"""
INS (Inverse Number of Samples) class weighting — applied in the loss, not
the sampler. Every image is seen once per epoch at its natural frequency;
a minority-class image's loss just counts for more when it's wrong.

This replaces the earlier (incorrect) implementation, which used these same
weights to bias a WeightedRandomSampler instead — that oversampled the same
few hundred minority images with replacement, which is a likely cause of
the train-to-val generalization collapse seen in Step 2D-v2.
"""

import numpy as np

import constants as c


def compute_ins_class_weights(labels):
    """
    weight(class) = 1 / count(class)

    Returns an array of length NUM_CLASSES, normalized so the weights sum
    to NUM_CLASSES (keeps the average loss magnitude comparable across
    different label distributions / chromosome runs).
    """
    class_counts = [0] * c.NUM_CLASSES
    for label in labels:
        class_counts[label] += 1

    raw_weights = []
    for count in class_counts:
        if count == 0:
            raw_weights.append(0.0)
        else:
            raw_weights.append(1.0 / count)

    total_weight = sum(raw_weights)
    normalized_weights = [w / total_weight * c.NUM_CLASSES for w in raw_weights]

    return np.array(normalized_weights, dtype=np.float32)
