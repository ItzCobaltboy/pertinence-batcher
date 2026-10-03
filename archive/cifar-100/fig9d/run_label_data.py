"""
Fig. 9(d) labeling entry point — "Inputs dispatched to either
shufflenetv2_x0_5, mobilenetv2_x0_75, or repvgg_a2."

Thin: wires this variant's config (config.py, same directory) into the
shared labeling code one directory up (../label_data.py). All actual
labeling logic — downloading/dumping CIFAR-100, carving the test/validation
split, computing per-model correctness — lives there.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from label_data import run_labeling

if __name__ == "__main__":
    run_labeling(config)
