"""
Settings for fine-tuning the yolo-dispatcher heads on CityFlow (CCTV) frames.

Named cf_config, not config, because the shared code and yolo-dispatcher
already have a `config.py` on the path. yolo-dispatcher's config is loaded
under the name `yd` and reused for everything that must stay identical:
model pool, costs, extractor, letterbox, chromosome layout. It is also what
gets passed as `config` to the shared dispatcher_analysis functions.

Importable without torch.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # cityflow-dispatcher/
REPO_ROOT = os.path.dirname(HERE)
YOLO_DISPATCHER_DIR = os.path.join(REPO_ROOT, "yolo-dispatcher")
CITYFLOW_ROUTING_DIR = os.path.join(REPO_ROOT, "cityflow-routing")
DISPATCHER_ANALYSIS_DIR = os.path.join(REPO_ROOT, "dispatcher_analysis")

# yolo-dispatcher's config imports letterbox / yolo_backbone by plain name;
# dispatcher_analysis provides loss, penalty_matrix, class_weights, weighting_scheme.
for _dir in (YOLO_DISPATCHER_DIR, DISPATCHER_ANALYSIS_DIR):
    if _dir not in sys.path:
        sys.path.append(_dir)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


yd = _load("yd_config", os.path.join(YOLO_DISPATCHER_DIR, "config.py"))
cf_dataset = _load("cf_dataset", os.path.join(CITYFLOW_ROUTING_DIR, "dataset.py"))
cf_heads = _load("cf_heads", os.path.join(CITYFLOW_ROUTING_DIR, "heads.py"))

MODEL_NAMES = list(yd.MODEL_NAMES)          # yolov8n, s, m, l (cost-ascending)
NUM_CLASSES = yd.NUM_CLASSES

# ---------------------------------------------------------------------------
# Data: which footage trains, which evaluates (never the same scenario)
# ---------------------------------------------------------------------------
# CityFlowV2 official splits: train = S01, S03, S04; validation = S02, S05.
# S05 is 11.6 GB of video, so it is left out by default.
TRAIN_SCENARIOS = ["S01", "S03", "S04"]
EVAL_SCENARIOS = ["S02"]

# Every Nth frame: neighbouring frames at 10 FPS are near-duplicates.
FRAME_STRIDE = 5

# Videos are the ones cityflow-routing/extract.py already put on disk.
VIDEO_DATA_DIR = cf_dataset.DEFAULT_DATA_DIR

# ---------------------------------------------------------------------------
# Correctness on CCTV (Week6 decision): box recall against yolov8l
# ---------------------------------------------------------------------------
# A model is correct on a frame if it recovers >= 80% of yolov8l's boxes
# (same class, IoU >= 0.5). Label = cheapest correct model. yolov8l is the
# reference, so it is always correct and there is no "no model correct" case.
REFERENCE_MODEL = "yolov8l"
RECALL_THRESHOLD = 0.80
MATCH_IOU = 0.5

# Road-scene classes only (COCO ids): person, bicycle, car, motorcycle, bus, truck.
DETECT_CLASSES = [0, 1, 2, 3, 5, 7]
DETECT_CONF = 0.25                     # ultralytics predict default, same as the COCO benchmark
IMG_SIZE = yd.IMG_SIZE                 # 640
DETECT_BATCH = 32

# ---------------------------------------------------------------------------
# Fine-tuning: start from each COCO Pareto head, keep its chromosome
# (penalty matrix + weighting scheme), train a few epochs on CityFlow labels.
# ---------------------------------------------------------------------------
FT_EPOCHS = 10
FT_LEARNING_RATE = 1e-4                # 10x below the from-scratch rate: nudge, don't retrain
FT_BATCH_SIZE = 128

# COCO Pareto front to start from (on the A100: yolo-dispatcher/results/).
COCO_RESULTS_DIR = os.path.join(YOLO_DISPATCHER_DIR, "results")

# ---------------------------------------------------------------------------
# Outputs. data/ = CityFlow training/eval data (labels + embeddings), kept
# apart from cityflow-routing's routing runs. results/ mirrors
# yolo-dispatcher's layout, so cityflow-routing/route.py --yolo-results
# results/ loads the fine-tuned heads unchanged.
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(HERE, "data")
LABELS_DIR = os.path.join(DATA_DIR, "labels")            # <scenario>_<camera>.csv.gz
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")    # <scenario>_<camera>.npz
LABEL_MANIFEST = os.path.join(DATA_DIR, "manifest.json")

RESULTS_DIR = os.path.join(HERE, "results")
NSGA2_DIR = os.path.join(RESULTS_DIR, "nsga2")
PARETO_FRONT_CSV = os.path.join(NSGA2_DIR, "pareto_front.csv")
EVAL_DIR = os.path.join(RESULTS_DIR, "eval")
MODEL_CACHE_DIR = os.path.join(EVAL_DIR, "model_cache")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")


def role_of(scenario):
    if scenario in TRAIN_SCENARIOS:
        return "train"
    if scenario in EVAL_SCENARIOS:
        return "eval"
    raise ValueError(f"{scenario} is in neither TRAIN_SCENARIOS nor EVAL_SCENARIOS")


assert not set(TRAIN_SCENARIOS) & set(EVAL_SCENARIOS), "train and eval scenarios overlap"
