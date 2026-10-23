"""
Class-weighting schemes applied inside the loss (see loss.py) rather than
biasing which images get drawn — every image is seen once per epoch at its
natural frequency; a minority-class image's loss simply counts for more
when the model gets it wrong.

Three schemes, matching the paper's own chromosome (Section II-B, page 6:
"we explored three weighting schemes to compute sample weights: the
inverse of the number of samples (INS), the inverse of the square root of
the number of samples (ISNS), and the Effective Number of Samples (ENS)
weighting scheme"):

  INS(c)  = 1 / count(c)
  ISNS(c) = 1 / sqrt(count(c))
  ENS(c)  = (1 - beta) / (1 - beta^count(c))   (Cui et al., 2019)

All three get renormalized so the weights sum to NUM_CLASSES, keeping
average loss magnitude comparable across different label distributions,
chromosomes, and schemes — without this, switching schemes would also
silently rescale the loss's overall magnitude, confounding the penalty
matrix's own effect.

Which scheme actually gets used for a given individual is decided by the
chromosome (see weighting_scheme.py), not hardcoded here — this module
only implements the three formulas plus a dispatcher that picks one by
name. Tracks that don't search over the scheme (ImageNette, CIFAR-10 —
settled decision, INS only, see CLAUDE.md) call compute_ins_class_weights
directly and never touch the other two schemes at all.
"""

import numpy as np


def _normalize(raw_weights, config):
    """Rescales weights so they sum to NUM_CLASSES — shared by every
    scheme below, keeps average loss magnitude comparable across schemes."""
    total_weight = sum(raw_weights)
    if total_weight == 0:
        # every class has zero samples, degenerate edge case — fall back to
        # uniform weights rather than dividing by zero
        return np.full(config.NUM_CLASSES, 1.0, dtype=np.float32)
    return np.array([w / total_weight * config.NUM_CLASSES for w in raw_weights], dtype=np.float32)


def _class_counts(labels, config):
    counts = [0] * config.NUM_CLASSES
    for label in labels:
        counts[label] += 1
    return counts


def compute_ins_class_weights(labels, config):
    """weight(c) = 1 / count(c), normalized to sum to NUM_CLASSES. Zero-
    count classes get weight 0 (no gradient signal exists for them
    regardless of weighting — 1/0 is undefined, not "infinitely
    important")."""
    counts = _class_counts(labels, config)
    raw_weights = [0.0 if c == 0 else 1.0 / c for c in counts]
    return _normalize(raw_weights, config)


def compute_isns_class_weights(labels, config):
    """weight(c) = 1 / sqrt(count(c)), normalized to sum to NUM_CLASSES.
    Milder than INS — a minority class still gets boosted, but not as
    aggressively as plain inverse-count does, since sqrt compresses the
    dynamic range between majority and minority counts."""
    counts = _class_counts(labels, config)
    raw_weights = [0.0 if c == 0 else 1.0 / np.sqrt(c) for c in counts]
    return _normalize(raw_weights, config)


# Cui et al., "Class-Balanced Loss Based on Effective Number of Samples"
# (CVPR 2019) introduces beta as a dataset-size-dependent hyperparameter,
# typically very close to 1 (0.99/0.999/0.9999 depending on dataset scale)
# — the paper doesn't state which beta it used for ENS, so this is a
# project-level choice, not a paper-matched value. 0.999 sits in the range
# the original ENS paper itself recommends for datasets on the order of
# 10^4-10^5 samples, which matches this repo's per-model-class routing
# counts.
ENS_BETA = 0.999


def compute_ens_class_weights(labels, config):
    """weight(c) = (1 - beta) / (1 - beta^count(c)), normalized to sum to
    NUM_CLASSES. Zero-count classes get weight 0 — the formula's own limit
    as count -> 0 is (1-beta)/(1-1) = (1-beta)/0, undefined, so this is
    handled as a special case rather than left to blow up or produce NaN."""
    counts = _class_counts(labels, config)
    raw_weights = []
    for c in counts:
        if c == 0:
            raw_weights.append(0.0)
        else:
            raw_weights.append((1.0 - ENS_BETA) / (1.0 - ENS_BETA ** c))
    return _normalize(raw_weights, config)


_SCHEME_FUNCTIONS = {
    "INS": compute_ins_class_weights,
    "ISNS": compute_isns_class_weights,
    "ENS": compute_ens_class_weights,
}


def compute_class_weights(labels, scheme, config):
    """Dispatches to whichever of the three schemes `scheme` names
    ("INS"/"ISNS"/"ENS" — see weighting_scheme.py for how a chromosome's
    gene decodes into one of these three strings)."""
    if scheme not in _SCHEME_FUNCTIONS:
        raise ValueError(f"Unknown weighting scheme '{scheme}', expected one of {list(_SCHEME_FUNCTIONS)}")
    return _SCHEME_FUNCTIONS[scheme](labels, config)
