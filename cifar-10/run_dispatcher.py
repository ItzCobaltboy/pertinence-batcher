"""
CIFAR-10 dispatcher NSGA-II search — entry point.

Thin: only wires this track's config (config.py, same directory) into the
shared search code (../dispatcher/). All actual search logic lives there.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dispatcher"))

import config
from nsga2_search import run_nsga2


if __name__ == "__main__":
    run_nsga2(config)
    print("\nDone.")
