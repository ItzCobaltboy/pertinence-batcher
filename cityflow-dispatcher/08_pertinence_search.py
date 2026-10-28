# NSGA-II search for the PERTINENCE dispatcher (pop 50, 50 generations, 13-gene chromosome), using the
# vendored original code in pertinence/search. Inputs come from prepare_pertinence_data.py.
# Env: PERT_MODE (s05|train), CF_TAG, PERTINENCE_SMOKE=1 for a tiny end-to-end check.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pertinence", "search"))
sys.path.insert(0, HERE)

import pertinence_config as config
from nsga2_search import run_nsga2

if __name__ == "__main__":
    for path in [config.TRAIN_GROUND_TRUTH_CSV, config.VAL_GROUND_TRUTH_CSV, config.TRAIN_EMBEDDINGS_NPZ,
                 config.VAL_EMBEDDINGS_NPZ]:
        if not os.path.exists(path):
            sys.exit(f"{path} missing: run prepare_pertinence_data.py first")
    print("results ->", config.RESULTS_DIR)
    run_nsga2(config)
    print("\nDone.")
