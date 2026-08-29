"""
All paths and hyperparameters used across the dispatcher analysis pipeline.
"""

import os
from torchvision import transforms

# Paths are built from this file's own location (dispatcher_analysis/src/),
# so the pipeline works no matter what directory you run main.py from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # dispatcher_analysis/

# ── Paths: data ──────────────────────────────────────────────────────────────

TRAIN_GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "data", "train_ground_truth.csv")
VAL_GROUND_TRUTH_CSV   = os.path.join(PROJECT_ROOT, "data", "val_ground_truth.csv")
PARETO_FRONT_CSV       = os.path.join(PROJECT_ROOT, "data", "pareto_front.csv")

# ── Paths: cached embeddings (so we don't recompute the ResNet18 forward
#    pass every time we rerun the pipeline) ──────────────────────────────────

TRAIN_EMBEDDINGS_NPZ = os.path.join(PROJECT_ROOT, "embeddings_cache", "train_embeddings.npz")
VAL_EMBEDDINGS_NPZ   = os.path.join(PROJECT_ROOT, "embeddings_cache", "val_embeddings.npz")

# ── Paths: outputs ───────────────────────────────────────────────────────────

TRAIN_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "predictions", "train_predictions.csv")
VAL_PREDICTIONS_CSV   = os.path.join(PROJECT_ROOT, "predictions", "val_predictions.csv")

TRAIN_SUMMARY_CSV = os.path.join(PROJECT_ROOT, "results", "train_summary.csv")
VAL_SUMMARY_CSV   = os.path.join(PROJECT_ROOT, "results", "val_summary.csv")
TRAIN_CM_NPZ      = os.path.join(PROJECT_ROOT, "results", "train_confusion_matrices.npz")
VAL_CM_NPZ        = os.path.join(PROJECT_ROOT, "results", "val_confusion_matrices.npz")

# ── Model pool constants ─────────────────────────────────────────────────────

NUM_CLASSES = 4
MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]
FLOPS_G     = [1.824, 3.679, 4.134, 11.604]

# ── Training hyperparameters ─────────────────────────────────────────────────

FC_EPOCHS  = 30
BATCH_SIZE = 128
LEARN_RATE = 1e-3

# ── Image preprocessing (must match what the ResNet18 pool model expects) ────

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
