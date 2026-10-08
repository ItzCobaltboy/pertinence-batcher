"""
All paths, hyperparameters, and pool constants for the PERTINENCE dispatcher
over the YOLOv8 n/s/m/l detection pool on COCO. Passed in explicitly as
`config` to the shared ../dispatcher/ and ../dispatcher_analysis/ code, same
contract as archive/cifar-100/fig9c/config.py (see dispatcher/README.md).

Importable without torch: everything torch/ultralytics-dependent lives in
yolo_backbone.py and is imported lazily, so label_data.py runs on a machine
with only numpy/pandas.

Overrides via environment variables (all optional):
  PERTINENCE_COCO_DIR    where COCO images live / get downloaded
                         (default: ../yolo-analysis/dataset, shared with the
                         benchmark, so a server that already ran it reuses them)
  PERTINENCE_SMOKE=1     tiny GA budget (pop 8, gen 2, 2 FC epochs) writing to
                         results_smoke/, to check the whole pipeline end to end
                         before committing hours of GPU time
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))     # yolo-dispatcher/
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
YOLO_ANALYSIS_DIR = os.path.join(REPO_ROOT, "yolo-analysis")

SMOKE = os.environ.get("PERTINENCE_SMOKE", "0") == "1"

# ---------------------------------------------------------------------------
# Inputs: the committed yolo-analysis benchmark (per-model, per-image recall)
# ---------------------------------------------------------------------------
def benchmark_csv(split):
    return os.path.join(YOLO_ANALYSIS_DIR, "results", split, "coco_class_recall_benchmark.csv")

# Images are referenced as <split>/<image_id:012d>.jpg relative to DATASET_DIR
# (the same layout yolo-analysis/dataset/ uses).
DATASET_DIR = os.environ.get("PERTINENCE_COCO_DIR", os.path.join(YOLO_ANALYSIS_DIR, "dataset"))
COCO_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/{split}/{image_id:012d}.jpg"

# ---------------------------------------------------------------------------
# Model pool (cost-ascending) and correctness definition
# ---------------------------------------------------------------------------
MODEL_NAMES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
MODEL_FILES = [f"{name}.pt" for name in MODEL_NAMES]   # names used in the benchmark CSV's `model` column
NUM_CLASSES = len(MODEL_NAMES)

# Week4 [DECISION]: a model is "correct" on an image if its class-set recall
# is >= 0.80 (see Journel/Week4.md for the threshold sweep).
RECALL_THRESHOLD = 0.80

# Images no model gets right are kept and labelled with the largest model,
# same convention as the CIFAR-100 track's label_data.py.
NO_CORRECT_MODEL_LABEL = NUM_CLASSES - 1

# Weight files: looked up in these directories first, otherwise ultralytics
# downloads them into MODELS_DIR.
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WEIGHT_SEARCH_DIRS = [MODELS_DIR, os.path.join(YOLO_ANALYSIS_DIR, "models"), os.getcwd()]

# ---------------------------------------------------------------------------
# Splits (paper's train/test/validation; its "test set" is the one NSGA-II
# fitness touches every generation, its "validation set" is held out)
# ---------------------------------------------------------------------------
#   train      : yolo-analysis' stratified 20,000-image train2017 subset.
#                YOLOv8 was trained on train2017, but unlike the CIFAR-100
#                classifiers it does not saturate there (yolov8n correct on
#                47% of the subset vs 53% of val2017 at recall >= 0.80), so
#                the routing labels still carry signal.
#   test       : 70% of val2017, stratified by ideal label -> NSGA-II fitness
#   final_val  : 30% of val2017, stratified by ideal label -> reported once
TRAIN_SPLIT = "train2017"
HELDOUT_SPLIT = "val2017"
FINAL_VAL_FRACTION = 0.30
SPLIT_SEED = 42

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "train_ground_truth.csv")
VAL_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "test_ground_truth.csv")              # paper's "test set"
FINAL_VAL_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "final_val_ground_truth.csv")   # paper's "validation set"
MODEL_COSTS_JSON = os.path.join(DATA_DIR, "model_costs.json")                        # written by measure_costs.py

EMBEDDINGS_CACHE_DIR = os.path.join(PROJECT_ROOT, "embeddings_cache")
TRAIN_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "train_embeddings.npz")
VAL_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "test_embeddings.npz")
FINAL_VAL_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "final_val_embeddings.npz")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_smoke" if SMOKE else "results")
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
# Cost (paper: THOP MACs doubled -> FLOPs; here in GFLOPs at 640x640)
# ---------------------------------------------------------------------------
IMG_SIZE = 640
MODEL_COST_UNIT = "GFLOPs"

# Ultralytics' published YOLOv8 detect numbers at 640 (thop-based, fused), used
# only as a cross-check for measure_costs.py's own measurement.
PUBLISHED_GFLOPS = {"yolov8n": 8.7, "yolov8s": 28.6, "yolov8m": 78.9, "yolov8l": 165.2}

# When yolov8n is picked, its backbone already ran as the feature extractor,
# so only its neck + head are new work (the paper's Table V discussion). Off
# by default: Eq. 4 then counts the extractor in full on every image, the
# conservative reading.
REUSE_EXTRACTOR_FOR_SMALLEST = False


def _load_costs():
    if not os.path.exists(MODEL_COSTS_JSON):
        return None
    with open(MODEL_COSTS_JSON) as f:
        return json.load(f)


_COSTS = _load_costs()
COSTS_MEASURED = _COSTS is not None
if COSTS_MEASURED:
    MODEL_COST = [float(_COSTS["model_gflops"][name]) for name in MODEL_NAMES]
    DISPATCHER_OVERHEAD_COST = float(_COSTS["extractor_gflops"]) + float(_COSTS["fc_gflops"])
    if REUSE_EXTRACTOR_FOR_SMALLEST:
        MODEL_COST[0] -= float(_COSTS["extractor_gflops"])
else:
    # Placeholders so label_data.py can import this before measure_costs.py
    # has run; run_dispatcher.py refuses to start a search on them.
    MODEL_COST = [PUBLISHED_GFLOPS[name] for name in MODEL_NAMES]
    DISPATCHER_OVERHEAD_COST = None

# ---------------------------------------------------------------------------
# Embedding extractor: frozen yolov8n backbone (layers 0-9, through SPPF),
# global-average-pooled at P3/P4/P5 and concatenated: 64 + 128 + 256 = 448.
# ---------------------------------------------------------------------------
EXTRACTOR_MODEL = "yolov8n"
EXTRACTOR_TAP_LAYERS = (4, 6, 9)
EMBEDDING_DIM = 448
EMBEDDING_NUM_WORKERS = 8


def build_feature_extractor(device):
    """Frozen yolov8n backbone embedder, output dim EMBEDDING_DIM (asserted).
    Caller handles .to(device)/.eval()."""
    from yolo_backbone import build_backbone_embedder
    return build_backbone_embedder(EXTRACTOR_MODEL, EXTRACTOR_TAP_LAYERS, EMBEDDING_DIM, WEIGHT_SEARCH_DIRS, MODELS_DIR)


from letterbox import Letterbox  # noqa: E402  (numpy/PIL only; torch imported on call)

# Same geometry yolov8 sees at inference (aspect-preserving resize, pad 114),
# square so batches stack.
IMAGE_TRANSFORM = Letterbox(IMG_SIZE)

# ---------------------------------------------------------------------------
# Chromosome: 4^2 - 4 = 12 off-diagonal penalty genes + 1 weighting-scheme gene
# (paper: "(num. of DNNs)^2 chromosomes to encode the P matrix and an extra one
# for the weighting scheme"; the 4 diagonal genes are always 0, so not searched)
# ---------------------------------------------------------------------------
N_GENES = NUM_CLASSES ** 2 - NUM_CLASSES + 1
PENALTY_LOWER_BOUND = 0.0
PENALTY_UPPER_BOUND = 100.0
SCHEME_GENE_LOWER_BOUND = 0.0
SCHEME_GENE_UPPER_BOUND = 3.0

# FC + NSGA-II hyperparameters: paper values (pop 50, 50 generations, SBX eta
# 20 / p 0.9, PM eta 25, 20 FC epochs, penalties in [0, 100]). Batch size and
# learning rate aren't given for the FC search; same as the CIFAR-100 track.
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
