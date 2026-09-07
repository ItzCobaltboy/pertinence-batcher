"""
All paths and pool constants for this pipeline. Every other module imports
from here — nothing is hardcoded twice.
"""

import os
from torchvision import transforms

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # code/CIFAR_10_Implementation/dispatcher_analysis/
MODELS_DIR_SHARED = os.path.join(os.path.dirname(PROJECT_ROOT), "models")  # checkpoints + model_loader.py

# Inputs (copied over from code/CIFAR_10_Implementation/dispatcher/)
PARETO_FRONT_CSV = os.path.join(PROJECT_ROOT, "data", "pareto_front.csv")
TRAIN_GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "data", "train_ground_truth.csv")
VAL_GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "data", "val_ground_truth.csv")

TRAIN_EMBEDDINGS_NPZ = os.path.join(PROJECT_ROOT, "embeddings_cache", "train_embeddings.npz")
VAL_EMBEDDINGS_NPZ = os.path.join(PROJECT_ROOT, "embeddings_cache", "val_embeddings.npz")

# individual_<id>.npz: W, b, chromosome. Rebuilt from pareto_front.csv's
# chromosomes on every run (see build_models.py) rather than trusted as a
# standing cache — retraining is cheap (~3-5s/individual), so this folder is
# always guaranteed to match the current pareto_front.csv.
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "model_cache")

# Outputs
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
TRAIN_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "train_predictions.csv")
VAL_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "val_predictions.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TRAIN_SUMMARY_CSV = os.path.join(RESULTS_DIR, "train_summary.csv")
VAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "val_summary.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Model pool — CIFAR-10 reproduction track, ordered by MAdds ascending.
# Cost metric is MAdds (millions); kept as MADDS_M (not FLOPS_G) so the unit
# difference from the ImageNette pipeline can't be silently mixed up.
NUM_CLASSES = 4
MODEL_NAMES = ["resnet20", "resnet32", "shufflenetv2_x2_0", "vgg16_bn"]
MADDS_M = [40.81, 69.12, 187.81, 313.73]

# Embedding extractor: resnet20, classifier head stripped. Output dim
# measured empirically (dummy 32x32 forward pass), not assumed.
EMBEDDING_DIM = 64

# FC training hyperparameters — must match dispatcher/src/constants.py
# exactly, since that's what actually produced pareto_front.csv.
FC_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Image preprocessing — only used if embeddings ever need to be computed
# fresh; normally the cached .npz files above are used directly. CIFAR-10
# native 32x32, chenyaofo's actual training normalization stats (verified
# against the released training log, not ImageNet mean/std).
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.2023, 0.1994, 0.2010]),
])
