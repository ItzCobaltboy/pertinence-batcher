"""
All paths, hyperparameters, and pool constants for the CIFAR-10 track —
used by both the shared dispatcher/ (NSGA-II search) and dispatcher_analysis/
(Pareto-front evaluation) code. Passed in explicitly as `config` to every
function that needs it (dependency injection, see dispatcher/README.md) —
the shared modules never import this file implicitly.
"""

import os
import sys
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))   # cifar-10/
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")            # checkpoints + model_loader.py

# One unavoidable sys.path insert to reach this track's own model loader —
# see dispatcher/README.md for why this is the one place it's needed.
sys.path.insert(0, MODELS_DIR)
from model_loader import load_model, strip_classifier_head  # noqa: E402

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

# Model pool — CIFAR-10 reproduction track (chenyaofo/pytorch-cifar-models),
# ordered by MAdds ascending. Cost metric is MAdds (millions), NOT GFLOPs
# like the ImageNette track's pool — MODEL_COST_UNIT makes this explicit
# wherever cost gets printed or plotted, so it can't be silently mixed up
# between tracks.
NUM_CLASSES = 4
MODEL_NAMES = ["shufflenetv2_x0_5", "resnet20", "resnet32", "vgg11_bn"]
MODEL_COST = [10.90, 40.81, 69.12, 153.29]
MODEL_COST_UNIT = "MAdds-M"

# Embedding extractor: shufflenetv2_x0_5 (cheapest pool model), final
# classifier head stripped. Output dim measured empirically (dummy 32x32
# forward pass) — NOT assumed to be 512 like the ImageNette track's
# ResNet18 extractor.
EMBEDDING_DIM = 1024
EMBEDDING_NUM_WORKERS = 32


def build_feature_extractor(device):
    """Returns a frozen shufflenetv2_x0_5 with its final FC classifier layer
    removed, output dim EMBEDDING_DIM (asserted, not assumed). Caller is
    responsible for .to(device)/.eval() — this only builds the module."""
    backbone = load_model("shufflenetv2_x0_5", device=None)  # keep on CPU until stripped
    extractor, measured_dim = strip_classifier_head(backbone, "shufflenetv2_x0_5")
    assert measured_dim == EMBEDDING_DIM, (
        f"shufflenetv2_x0_5 embedding dim changed: measured {measured_dim}, config.py says {EMBEDDING_DIM}"
    )
    return extractor


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

# Image preprocessing — only used when embeddings need to be computed
# fresh. CIFAR-10 native 32x32, no resize/crop. Normalization stats are
# chenyaofo's actual training config (verified against the released
# training log for cifar10_resnet20, not assumed from memory) — NOT
# ImageNet mean/std.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.2023, 0.1994, 0.2010]),
])
