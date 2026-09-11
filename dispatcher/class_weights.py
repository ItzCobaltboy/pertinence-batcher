"""
INS (Inverse Number of Samples) class weighting, applied inside the loss
(see loss.py) rather than biasing which images get drawn — every image is
seen once per epoch at its natural frequency; a minority-class image's loss
simply counts for more when the model gets it wrong.
"""

import numpy as np


def compute_ins_class_weights(labels, config):
    """
    weight(class) = 1 / count(class), normalized so the weights sum to
    NUM_CLASSES (keeps average loss magnitude comparable across different
    label distributions / chromosome runs). Returns a (NUM_CLASSES,) array.
    """
    class_counts = [0] * config.NUM_CLASSES
    for label in labels:
        class_counts[label] += 1

    raw_weights = []
    for count in class_counts:
        if count == 0:
            raw_weights.append(0.0)
        else:
            raw_weights.append(1.0 / count)

    total_weight = sum(raw_weights)
    normalized_weights = [w / total_weight * config.NUM_CLASSES for w in raw_weights]

    return np.array(normalized_weights, dtype=np.float32)
