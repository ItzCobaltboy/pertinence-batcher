# Config module for the vendored PERTINENCE code in pertinence/ (same contract as
# archive/yolo-dispatcher/config.py, see pertinence/search/README_original.md). Everything the search and the
# analysis need is a plain attribute here. Hyperparameters are the paper values the COCO run used.
#
# Environment variables (all optional):
#   PERT_MODE   s05   (default) head trained on S05 cameras fold A, fitness on S05 fold B, final = S02
#               train head trained on the train split (S01,S03,S04), fitness on S05, final = S02
#   CF_TAG      "" or e.g. "_v2": which fine-tuned pool the labels come from (see common/paths.py)
#   PERTINENCE_SMOKE=1   tiny GA budget (pop 8, 2 generations, 2 FC epochs), results go to a *_smoke folder
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("CF_TAG", "")
MODE = os.environ.get("PERT_MODE", "s05")
SMOKE = os.environ.get("PERTINENCE_SMOKE", "0") == "1"
if MODE not in ("s05", "train"):
    raise SystemExit("PERT_MODE must be s05 or train")

RUN_NAME = "pertinence" + TAG + "_" + MODE

# ---------------------------------------------------------------------------
# Model pool (cost-ascending). The correctness definition lives in prepare_pertinence_data.py
# (recall vs yolov8l >= config.RECALL_THRESHOLD), the same rule as the COCO run.
# ---------------------------------------------------------------------------
MODEL_NAMES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
NUM_CLASSES = len(MODEL_NAMES)
NO_CORRECT_MODEL_LABEL = NUM_CLASSES - 1

# ---------------------------------------------------------------------------
# Data: three roles, as in the paper. "train" trains the FC heads, "test" is what NSGA-II fitness is
# measured on every generation (VAL_* in the code), "final_val" is reported once (here always S02).
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data", RUN_NAME)
TRAIN_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "train_ground_truth.csv")
VAL_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "test_ground_truth.csv")
FINAL_VAL_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "final_val_ground_truth.csv")
TRAIN_EMBEDDINGS_NPZ = os.path.join(DATA_DIR, "train_embeddings.npz")
VAL_EMBEDDINGS_NPZ = os.path.join(DATA_DIR, "test_embeddings.npz")
FINAL_VAL_EMBEDDINGS_NPZ = os.path.join(DATA_DIR, "final_val_embeddings.npz")

RESULTS_DIR = os.path.join(ROOT, "results", RUN_NAME + ("_smoke" if SMOKE else ""))
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
NSGA2_DIR = os.path.join(RESULTS_DIR, "nsga2")
PARETO_FRONT_CSV = os.path.join(NSGA2_DIR, "pareto_front.csv")
EVAL_DIR = os.path.join(RESULTS_DIR, "eval")
MODEL_CACHE_DIR = os.path.join(EVAL_DIR, "model_cache")
PREDICTIONS_DIR = os.path.join(EVAL_DIR, "predictions")
TRAIN_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "train_predictions.csv")
VAL_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "test_predictions.csv")
FINAL_VAL_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "final_val_predictions.csv")
TRAIN_SUMMARY_CSV = os.path.join(EVAL_DIR, "train_summary.csv")
VAL_SUMMARY_CSV = os.path.join(EVAL_DIR, "test_summary.csv")
FINAL_VAL_SUMMARY_CSV = os.path.join(EVAL_DIR, "final_val_summary.csv")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")

# ---------------------------------------------------------------------------
# Cost in GFLOPs per frame at 640 (measured earlier by 05_build_dispatcher.py, models/dispatcher/costs.json).
# Dispatcher overhead = frozen yolov8n extractor + the FC head, added to every routed frame (paper Eq. 4).
# ---------------------------------------------------------------------------
MODEL_COST_UNIT = "GFLOPs"
EMBEDDING_DIM = 448
_COSTS_JSON = os.path.join(ROOT, "models", "dispatcher", "costs.json")
with open(_COSTS_JSON) as f:
    _COSTS = json.load(f)
MODEL_COST = [float(_COSTS["model_gflops"][letter]) for letter in ["n", "s", "m", "l"]]
FC_GFLOPS = 2 * EMBEDDING_DIM * NUM_CLASSES / 1e9
DISPATCHER_OVERHEAD_COST = float(_COSTS["extractor_gflops"]) + FC_GFLOPS
COSTS_MEASURED = True

# Embeddings are always cached by prepare_pertinence_data.py before the search starts; if one is missing
# the original code would try to compute it, which this config does not support.
EMBEDDING_NUM_WORKERS = 0
DATASET_DIR = ""
IMAGE_TRANSFORM = None


def build_feature_extractor(device):
    raise RuntimeError("embeddings must be cached by prepare_pertinence_data.py first")


# ---------------------------------------------------------------------------
# Chromosome: 4^2 - 4 = 12 off-diagonal penalty genes + 1 weighting-scheme gene, then the paper's GA settings
# (same values as archive/yolo-dispatcher/config.py)
# ---------------------------------------------------------------------------
N_GENES = NUM_CLASSES ** 2 - NUM_CLASSES + 1
PENALTY_LOWER_BOUND = 0.0
PENALTY_UPPER_BOUND = 100.0
SCHEME_GENE_LOWER_BOUND = 0.0
SCHEME_GENE_UPPER_BOUND = 3.0

FC_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
POPULATION_SIZE = 50
GENERATIONS = 50
SBX_ETA = 20
SBX_CROSSOVER_PROBABILITY = 0.9
MUTATION_ETA = 25
CHECKPOINT_EVERY_N_GENERATIONS = 5
NSGA2_SEED = 42
if SMOKE:
    FC_EPOCHS = 2
    POPULATION_SIZE = 8
    GENERATIONS = 2
    CHECKPOINT_EVERY_N_GENERATIONS = 1
