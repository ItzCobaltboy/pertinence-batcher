"""
NSGA-II search for the YOLOv8 dispatcher, entry point. Thin: wires this
track's config.py into the shared ../dispatcher/ search code.

config.VAL_* points at the test split (paper's "test set", NSGA-II fitness);
the final_val split is only touched by run_dispatcher_analysis.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dispatcher"))

import config
from nsga2_search import run_nsga2


if __name__ == "__main__":
    if not config.COSTS_MEASURED:
        sys.exit(f"{config.MODEL_COSTS_JSON} missing: run measure_costs.py first "
                 f"(fitness needs the measured dispatcher overhead)")
    if config.SMOKE:
        print("SMOKE run: tiny GA budget, results go to", config.RESULTS_DIR)
    run_nsga2(config)
    print("\nDone.")
