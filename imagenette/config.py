"""
All paths, hyperparameters, and pool constants for the ImageNette track —
used by both the shared dispatcher/ (NSGA-II search) and dispatcher_analysis/
(Pareto-front evaluation) code. Passed in explicitly as `config` to every
function that needs it (dependency injection, see dispatcher/README.md) —
the shared modules never import this file implicitly.
"""

import os
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))   # imagenette/

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "train_ground_truth.csv")
VAL_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "val_ground_truth.csv")

EMBEDDINGS_CACHE_DIR = os.path.join(PROJECT_ROOT, "embeddings_cache")
TRAIN_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "train_embeddings.npz")
VAL_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "val_embeddings.npz")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
NSGA2_DIR = os.path.join(RESULTS_DIR, "nsga2")
PARETO_FRONT_CSV = os.path.join(NSGA2_DIR, "pareto_front.csv")

EVAL_DIR = os.path.join(RESULTS_DIR, "eval")
MODEL_CACHE_DIR = os.path.join(EVAL_DIR, "model_cache")
PREDICTIONS_DIR = os.path.join(EVAL_DIR, "predictions")
TRAIN_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "train_predictions.csv")
VAL_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "val_predictions.csv")
TRAIN_SUMMARY_CSV = os.path.join(EVAL_DIR, "train_summary.csv")
VAL_SUMMARY_CSV = os.path.join(EVAL_DIR, "val_summary.csv")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")

# Model pool
NUM_CLASSES = 4
MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]
MODEL_COST = [1.824, 3.679, 4.134, 11.604]
MODEL_COST_UNIT = "GFLOPs"

# Dispatcher's own fixed overhead — the embedding extractor's (ResNet18,
# head stripped) forward pass plus the FC head's (Linear(512, 4)) forward
# pass, measured via thop the same way archive/model_analysis measured the
# pool models' own MODEL_COST figures above. Added on top of the dispatched
# model's cost in dispatcher/fitness.py and dispatcher_analysis/summarize.py
# — see fitness.py's module docstring for why (the paper's Eq. 4 defines
# its cost objective as inclusive of this overhead, not just the selected
# model). Near-identical to resnet18's own MODEL_COST entry above since the
# extractor IS a resnet18 backbone with a trivial FC head swapped in for
# its original 1000-class classifier.
DISPATCHER_OVERHEAD_COST = 1.824

# ImageNette folder names are ImageNet synset IDs, not 0-9 class indices —
# label_data.py remaps them to the real 1000-class ImageNet index every
# pretrained pool model here expects (otherwise correctness looks like
# random-guess ~9-10% for every model, since predictions and folder-derived
# 0-9 labels would live in different label spaces entirely).
IMAGENETTE_LABEL_MAP = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

# Embedding extractor: ResNet18 (cheapest pool model), classifier head
# stripped, ImageNet-pretrained weights pulled via torchvision.
EMBEDDING_DIM = 512
EMBEDDING_NUM_WORKERS = 0


def build_feature_extractor(device):
    """Returns a frozen ResNet18 with its final FC classifier layer
    removed, output dim EMBEDDING_DIM. Caller is responsible for
    .to(device)/.eval() — this only builds the module."""
    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    return nn.Sequential(*list(backbone.children())[:-1])


# Chromosome: 12 genes = the off-diagonal entries of the 4x4 penalty matrix.
# Class weighting is fixed to INS (applied in the loss) — no weighting-
# scheme gene is searched.
N_GENES = 12
PENALTY_LOWER_BOUND = 0.0
PENALTY_UPPER_BOUND = 5.0

# FC training hyperparameters
FC_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# NSGA-II hyperparameters
POPULATION_SIZE = 50
GENERATIONS = 30
SBX_ETA = 20
SBX_CROSSOVER_PROBABILITY = 0.9
MUTATION_ETA = 25
CHECKPOINT_EVERY_N_GENERATIONS = 5

# Image preprocessing — only used when embeddings need to be computed fresh;
# normally the cached .npz files above are used directly.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
