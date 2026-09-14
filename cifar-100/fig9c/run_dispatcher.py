"""
Fig. 9(c) dispatcher NSGA-II search — entry point.

Thin: only wires this variant's config (config.py, same directory) into the
shared search code (../../dispatcher/). All actual search logic lives
there.

config.VAL_GROUND_TRUTH_CSV/VAL_EMBEDDINGS_NPZ point at the test-fitness
split here, not the held-out validation slice — this is the paper's "test
set" role (per-individual fitness during the search), not its "validation
set" role. See config.py's "Split naming" comment.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dispatcher"))

import config
from nsga2_search import run_nsga2


if __name__ == "__main__":
    run_nsga2(config)
    print("\nDone.")
