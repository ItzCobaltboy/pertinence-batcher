"""
All paths and settings used across the model_analysis benchmark pipeline.
"""

import os
from torchvision import transforms

# Paths are built from this file's own location, so the pipeline works no
# matter what directory you run main.py from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # "model_analysis/"

DATASET_PATH = os.path.join(PROJECT_ROOT, "..", "dataset")
RESNET_MODELS_DIR = os.path.join(PROJECT_ROOT, "ResnetModels")
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "model_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

BENCHMARK_RESULTS_CSV = os.path.join(RESULTS_DIR, "torch_tensorrt_benchmark.csv")
EAGER_BASELINE_CSV = os.path.join(RESULTS_DIR, "eager_baseline.csv")

# ── Model pool ────────────────────────────────────────────────────────────────

MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet152"]
PRECISIONS = ["fp32", "fp16", "int8", "fp8"] # 5070ti
# PRECISIONS = ["fp32", "fp16", "int8"] # fp8 is not supported in A100 

# ── Benchmark settings ───────────────────────────────────────────────────────

# Torch-TensorRT engines below are compiled for a fixed batch size of 1
# (COMPILE_INPUT_SHAPE), so the val loader must also use batch size 1 -
# a compiled engine rejects any other input shape outright.
VAL_BATCH_SIZE = 1
LATENCY_WARMUP_RUNS = 10
LATENCY_TIMED_RUNS = 100
COMPILE_INPUT_SHAPE = [1, 3, 224, 224]

# ImageNette folder names are ImageNet synset IDs. ImageFolder assigns labels
# 0-9 alphabetically, but pretrained ResNets output 1000-class ImageNet
# logits, so we remap folder labels to the real ImageNet class indices.
IMAGENETTE_LABEL_MAP = {
    'n01440764': 0,    # tench
    'n02102040': 217,  # English springer
    'n02979186': 482,  # cassette player
    'n03000684': 491,  # chain saw
    'n03028079': 497,  # church
    'n03394916': 566,  # French horn
    'n03417042': 569,  # garbage truck
    'n03425413': 571,  # gas pump
    'n03445777': 574,  # golf ball
    'n03888257': 701,  # parachute
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
