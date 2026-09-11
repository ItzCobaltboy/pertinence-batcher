import os
from torchvision import transforms

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # "archive/quantization_experiments/"

# Archived 2026-09: this pipeline used to live at "code/quantization_experiments/"
# with the dataset one level up at "code/dataset/". Both moved (this pipeline ->
# archive/, dataset -> imagenette/dataset/) — repointed here so this still runs if
# revisited, without otherwise restructuring this archived pipeline.
DATASET_PATH = os.path.join(PROJECT_ROOT, "..", "..", "imagenette", "dataset")
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "model_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]

IMAGENETTE_LABEL_MAP = {
    'n01440764': 0,
    'n02102040': 217,
    'n02979186': 482,
    'n03000684': 491,
    'n03028079': 497,
    'n03394916': 566,
    'n03417042': 569,
    'n03425413': 571,
    'n03445777': 574,
    'n03888257': 701,
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Calibration: how many batches of train data to run through for PTQ
CALIB_BATCH_SIZE = 8
CALIB_NUM_BATCHES = 64  # 512 images total

LATENCY_WARMUP_RUNS = 10
LATENCY_TIMED_RUNS = 100
