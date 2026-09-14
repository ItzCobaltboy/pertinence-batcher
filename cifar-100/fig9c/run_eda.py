"""
Fig. 9(c) ground-truth EDA — entry point.

Thin: wires this variant's config into the shared EDA code (../../eda/).
All actual analysis logic lives there.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eda"))

import config
from ground_truth_eda import run_eda, _print_split_report

if __name__ == "__main__":
    # covers config.TRAIN_GROUND_TRUTH_CSV (paper's "training set") and
    # config.VAL_GROUND_TRUTH_CSV (paper's "test set")
    run_eda(config)

    # the carved-out held-out slice (paper's "validation set") isn't part
    # of the shared function's train/val assumption — report it the same
    # way here
    _print_split_report("final_val", config.FINAL_VAL_GROUND_TRUTH_CSV, config)
