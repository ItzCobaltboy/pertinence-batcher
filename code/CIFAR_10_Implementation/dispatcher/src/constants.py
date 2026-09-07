"""
All paths, hyperparameters, and pool constants for the NSGA-II dispatcher
search. Every other module imports from here — nothing is hardcoded twice.
"""

import os
from torchvision import transforms

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # code/CIFAR_10_Implementation/dispatcher/
MODELS_DIR_SHARED = os.path.join(os.path.dirname(PROJECT_ROOT), "models")  # checkpoints + model_loader.py

# Train-only ground truth. This pipeline only searches (see dispatcher_analysis/
# for train+val evaluation) so no val paths live here.
TRAIN_GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "data", "train_ground_truth.csv")

# Backbone is frozen, so embeddings are identical across every individual in
# the search — computed once, cached, reused for the whole run.
TRAIN_EMBEDDINGS_NPZ = os.path.join(PROJECT_ROOT, "embeddings_cache", "train_embeddings.npz")

# Outputs
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NSGA2_DIR = os.path.join(RESULTS_DIR, "nsga2")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PARETO_FRONT_CSV = os.path.join(NSGA2_DIR, "pareto_front.csv")
MODELS_DIR = os.path.join(NSGA2_DIR, "models")

# Model pool — CIFAR-10 reproduction track (chenyaofo/pytorch-cifar-models),
# ordered by MAdds ascending. Cost metric is MAdds (millions), NOT FLOPs_G
# like the ImageNette pipeline's pool — kept as a separate constant name
# (MADDS_M) rather than reusing FLOPS_G so the unit difference can't be
# silently mixed up.
NUM_CLASSES = 4
MODEL_NAMES = ["resnet20", "resnet32", "shufflenetv2_x2_0", "vgg16_bn"]
MADDS_M = [40.81, 69.12, 187.81, 313.73]

# Embedding extractor: resnet20 (cheapest pool model), classifier head
# stripped. Output dim measured empirically (dummy 32x32 forward pass) in
# code/CIFAR_10_Implementation/models/model_loader.py — NOT assumed to be
# 512 like the ImageNette pipeline's ResNet18 extractor.
EMBEDDING_DIM = 64

# Chromosome: 12 genes = the off-diagonal entries of the 4x4 penalty matrix.
# Class weighting is fixed to INS (applied in the loss) — no weighting-scheme
# gene is searched.
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

# Image preprocessing — only used when embeddings need to be computed fresh.
# CIFAR-10 native 32x32, no resize/crop. Normalization stats are chenyaofo's
# actual training config (verified against the released training log for
# cifar10_resnet20, not assumed from memory) — NOT ImageNet mean/std.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.2023, 0.1994, 0.2010]),
])
