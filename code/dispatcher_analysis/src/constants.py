"""
All paths and pool constants for this pipeline. Every other module imports
from here — nothing is hardcoded twice.
"""

import os
from torchvision import transforms

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # code/dispatcher_analysis/

# ── Paths: inputs (copied over from code/Dispatcher/, no retraining needed —
#    the actual trained FC weights per Pareto individual were saved there) ───

PARETO_FRONT_CSV = os.path.join(PROJECT_ROOT, "data", "pareto_front.csv")
TRAIN_GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "data", "train_ground_truth.csv")
VAL_GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "data", "val_ground_truth.csv")

TRAIN_EMBEDDINGS_NPZ = os.path.join(PROJECT_ROOT, "embeddings_cache", "train_embeddings.npz")
VAL_EMBEDDINGS_NPZ = os.path.join(PROJECT_ROOT, "embeddings_cache", "val_embeddings.npz")

MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "model_cache")   # individual_<id>.npz: W, b, chromosome

# ── Paths: outputs ───────────────────────────────────────────────────────────

PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
TRAIN_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "train_predictions.csv")
VAL_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "val_predictions.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TRAIN_SUMMARY_CSV = os.path.join(RESULTS_DIR, "train_summary.csv")
VAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "val_summary.csv")

# ── Model pool ────────────────────────────────────────────────────────────────

NUM_CLASSES = 4
MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]
FLOPS_G = [1.824, 3.679, 4.134, 11.604]

# ── Image preprocessing (only used if embeddings ever need to be computed
#    fresh — normally the cached .npz files above are used directly) ────────

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
