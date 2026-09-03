"""
All paths, hyperparameters, and pool constants for the NSGA-II dispatcher
search. Every other module imports from here — nothing is hardcoded twice.
"""

import os
from torchvision import transforms

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # code/Dispatcher/

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

# Model pool
NUM_CLASSES = 4
MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]
FLOPS_G = [1.824, 3.679, 4.134, 11.604]

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

# Image preprocessing — only used when embeddings need to be computed fresh
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
