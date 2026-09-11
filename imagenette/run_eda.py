"""
ImageNette ground-truth EDA — entry point.

Thin: only wires this track's config (config.py, same directory) into the
shared EDA code (../eda/). All actual analysis logic lives there.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda"))

import config
from ground_truth_eda import run_eda


if __name__ == "__main__":
    run_eda(config)
