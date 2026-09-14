"""
All paths, hyperparameters, and pool constants for reproducing Fig. 9(c) of
the PERTINENCE paper: "Inputs dispatched to either shufflenetv2_x0_5,
mobilenetv2_x0_75, or repvgg_a2" (CIFAR-100, page 8's Fig. 9 caption, read
directly off the figure — this exact three-model wording is what's being
matched here, not the full six-model Fig. 4b pool an earlier version of
this track built a dispatcher against). Passed in explicitly as `config` to
every function in the shared dispatcher/, dispatcher_analysis/, and eda/
packages, plus the sibling label_data.py one directory up — none of those
import this file implicitly, see dispatcher/README.md for the contract.

Sibling variant: ../fig9d/config.py (Fig. 9(d)'s three-model subset).
Shared across both variants (see the top-level cifar-100/README.md): the
raw dataset images and the model checkpoints/loader. NOT shared: ground-
truth CSVs, embeddings cache, or results — each variant's routing labels
depend on its own model subset, so a shared embeddings cache would leak one
variant's labels into the other's training. Every path below is scoped to
this variant's own subfolder except DATASET_DIR/MODELS_DIR.
"""

import os
import sys
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))                    # cifar-100/fig9c/
CIFAR100_ROOT = os.path.dirname(PROJECT_ROOT)                                # cifar-100/
MODELS_DIR = os.path.join(CIFAR100_ROOT, "models")                           # shared checkpoints + loader

# One unavoidable sys.path insert to reach the shared model loader (and, a
# directory up, the shared label_data.py) — see dispatcher/README.md for
# why this is the one place it's needed.
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, CIFAR100_ROOT)
from model_loader import load_model, strip_classifier_head, MFLOPS_M_BY_NAME  # noqa: E402

DATASET_DIR = os.path.join(CIFAR100_ROOT, "dataset")   # shared raw CIFAR-100 images, all variants
DATA_DIR = os.path.join(PROJECT_ROOT, "data")           # this variant's own ground-truth CSVs

# Split naming — see cifar-100/label_data.py's module docstring for the
# full rationale (three-way train/test/validation, paper's own backwards
# terminology, why the validation slice has to come from official test
# rather than official train). VAL_* is wired to the paper's "test set"
# role (NSGA-II fitness during the search); FINAL_VAL_* is the paper's
# actual "validation set" role, touched only by run_dispatcher_analysis.py's
# second pass.
TRAIN_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "train_ground_truth.csv")           # 50,000, official train, untouched (paper's "training set")
VAL_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "test_ground_truth.csv")              # 7,000, stratified from official test (paper's "test set")
FINAL_VAL_GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "final_val_ground_truth.csv")   # 3,000, stratified from official test (paper's "validation set")

EMBEDDINGS_CACHE_DIR = os.path.join(PROJECT_ROOT, "embeddings_cache")   # not shared, see module docstring
TRAIN_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "train_embeddings.npz")
VAL_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "test_embeddings.npz")
FINAL_VAL_EMBEDDINGS_NPZ = os.path.join(EMBEDDINGS_CACHE_DIR, "final_val_embeddings.npz")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
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

# Model pool — Fig. 9(c)'s exact three-model subset, read directly off the
# figure caption, MFLOPs-ascending. mobilenetv2_x0_75 is NOT one of Fig.
# 4b's six plotted models — it's a real, separately published chenyaofo
# checkpoint the paper uses specifically for this subset (and Fig. 9(a)'s),
# confirmed against the chenyaofo repo's full CIFAR-100 model list before
# downloading it, not a typo for mobilenetv2_x0_5.
NUM_CLASSES = 3
MODEL_NAMES = ["shufflenetv2_x0_5", "mobilenetv2_x0_75", "repvgg_a2"]
MODEL_COST = [MFLOPS_M_BY_NAME[name] for name in MODEL_NAMES]
MODEL_COST_UNIT = "MFLOPs-M"

# Dispatcher's own fixed overhead — shufflenetv2_x0_5 (head stripped)
# forward pass plus this variant's FC head (Linear(1024, 3)) forward pass,
# measured locally via thop (MACs doubled to FLOPs — see model_loader.py's
# module docstring for why doubling, and why this is a fresh local
# measurement rather than a published table figure). Added on top of the
# dispatched model's cost in the shared dispatcher/fitness.py and
# dispatcher_analysis/summarize.py, matching the paper's Eq. 4 (the
# objective is inclusive of extractor+FC cost) and page 7 (that inclusive
# figure is what decides Pareto dominance during the search, not just what
# gets reported afterward). This number (23.91) is within 1% of the
# paper's own published dispatcher-overhead figure for CIFAR-100/
# ShuffleNetV2 (Table 5: 24.17 MFLOPS) — strong evidence the measurement
# methodology (thop, doubled, 32x32 input) matches what the paper itself
# used, not just a coincidence of similar magnitude.
DISPATCHER_OVERHEAD_COST = 23.91

# Embedding extractor: shufflenetv2_x0_5, matching the paper's explicit
# statement for CIFAR-100 (page 8) and identical across every variant of
# this track — Fig. 9's subsets change which models get dispatched to, not
# which model extracts features. Output dim measured empirically, asserted
# against the hardcoded value below.
EMBEDDING_DIM = 1024
EMBEDDING_NUM_WORKERS = 8


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


# Chromosome: num_models^2 - num_models = 6 off-diagonal penalty-matrix
# genes, PLUS one weighting-scheme selector gene — matching the paper's own
# chromosome exactly ("we need (num. of DNNs)^2 chromosomes to encode the P
# matrix and an extra one for the weighting scheme"). This is a reversal of
# an earlier, settled decision (INS hardcoded, no scheme gene) that this
# track shared with ImageNette/CIFAR-10 — ImageNette and CIFAR-10 keep that
# decision, only this track's two variants search the scheme now. See
# dispatcher/weighting_scheme.py: whether a track searches the scheme is
# inferred from N_GENES itself (penalty-count vs. penalty-count+1), not a
# separate flag — this line is what actually turns the feature on.
N_GENES = 7

# Paper's penalty range (Section II-C, page 6-7): [0, 100]. The paper also
# describes a discretized step size in [0.5, 1]; pymoo's SBX crossover +
# polynomial mutation (what the shared dispatcher/nsga2_search.py wires in)
# operate over continuous reals with no native step-size/discretization
# knob, so this searches the same [0, 100] range as a continuous interval
# instead — an approximation of the paper's discretized search space.
PENALTY_LOWER_BOUND = 0.0
PENALTY_UPPER_BOUND = 100.0

# The chromosome's last gene selects a weighting scheme (INS/ISNS/ENS, see
# dispatcher/class_weights.py) rather than a penalty value, so it needs its
# own range, not [0, 100] — [0, 3) so it splits evenly into 3 bins of width
# 1, one per scheme, via dispatcher/weighting_scheme.py's decode_scheme().
# The paper doesn't specify how it encodes a discrete 3-way choice as a
# continuous gene for its own MOEA implementation (pymoo's SBX/polynomial
# mutation only ever produce continuous reals) — this bucketing is a
# reasonable, simple choice to fill that gap, not a value read out of the
# paper.
SCHEME_GENE_LOWER_BOUND = 0.0
SCHEME_GENE_UPPER_BOUND = 3.0

# FC training hyperparameters. FC_EPOCHS=20 matches the paper's stated
# value exactly. BATCH_SIZE and LEARNING_RATE aren't specified anywhere in
# the paper; kept at the same values every other track/variant already
# uses since there's nothing paper-specific to match here.
FC_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# NSGA-II hyperparameters — all four of these match the paper's stated
# values exactly (Section II-C, page 7): population 50, 50 generations, SBX
# eta=20 with crossover probability 0.9, polynomial mutation eta=25. With
# only N_GENES=7 to search (vs. the abandoned full-pool attempt's 30), this
# budget is a much closer match to what the paper itself actually ran per
# subset than a 6-model dispatcher would have been.
POPULATION_SIZE = 50
GENERATIONS = 50
SBX_ETA = 20
SBX_CROSSOVER_PROBABILITY = 0.9
MUTATION_ETA = 25
CHECKPOINT_EVERY_N_GENERATIONS = 5

# Image preprocessing — only used when embeddings need to be computed
# fresh. CIFAR-100 native 32x32, no resize/crop, no augmentation (nothing
# here ever trains a pool model — they're pretrained and frozen). Stats are
# chenyaofo's actual training config for CIFAR-100
# (image-classification-codebase's conf/cifar100.conf), not assumed from
# memory.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070, 0.4865, 0.4409],
                         std =[0.2673, 0.2564, 0.2761]),
])

# Test/validation carve: paper doesn't specify a size or mechanism for
# splitting its "test set" from its "validation set". Official CIFAR-100
# test has exactly 100 images/class — split class-stratified 70/30, drawn
# once with a fixed seed: 7,000 for NSGA-II fitness, 3,000 held out as the
# real final validation. Shared with fig9d/config.py so both variants carve
# the identical split (same seed, same fraction) even though each computes
# its own correctness columns against it independently.
FINAL_VAL_FRACTION_OF_TEST = 0.30
FINAL_VAL_SEED = 42
