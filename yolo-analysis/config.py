"""
Config for the YOLOv8 / COCO multi-label class-recall benchmark, run over
both val2017 and train2017.

This is prep work for redefining "accuracy" in the PERTINENCE dispatcher for a
detection-based model pool (see repo root CLAUDE.md). Single top-level config,
same config-injection convention as `cifar-100/*/config.py` -- every path and
hyperparameter lives here, nothing hardcoded in the scripts.

Self-contained: this whole `yolo-analysis/` folder is meant to be copied to a
server and run there standalone (own requirements.txt, no imports from the rest
of the repo).

Split-parameterized: every path that differs per split (images dir, ground
truth json, raw prediction cache, output csv) is a function of `split`
instead of a fixed constant, so `run_benchmark.py`/`plot_results.py` iterate
over SPLITS rather than each split needing its own copy of the pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Paths (all relative to this file, so the folder works from any location it's
# copied to)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
# Both official COCO 2017 detection splits. train2017 is included on top of
# the original val2017-only pass so the dispatcher's eventual labeling step
# has train-side ground truth to train on, not just the held-out val numbers.
SPLITS = ["val2017", "train2017"]

# Expected image count per split -- ensure_coco_split()/ensure_train_subset()
# use this to decide whether an existing images dir is actually complete or a
# stale partial extraction/download that needs finishing.
# train2017 is NOT the full 118,287-image split -- ran out of local disk quota
# on the target server for the full ~18GB download, so train2017 here means a
# stratified subset instead (see build_train_subset.py). Kept the full-split
# expected count around, commented, purely as a reference point for how far
# short of the real split this subset is.
# EXPECTED_IMAGE_COUNT["train2017"] if it were the full split: 118287
EXPECTED_IMAGE_COUNT = {
    "val2017": 5000,
    "train2017": None,  # set from TRAIN_SUBSET_SIZE below
}

# Official COCO download URLs (cocodataset.org). Both instances_train2017.json
# and instances_val2017.json ship inside the same annotations zip.
COCO_IMAGES_URL = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# ---------------------------------------------------------------------------
# train2017 stratified subset
# ---------------------------------------------------------------------------
# Full train2017 (118,287 images, ~18GB) doesn't fit the target server's disk
# budget. Sample a stratified subset instead -- proportional-to-frequency
# per category rather than uniform-random over images, since COCO's category
# distribution is heavily skewed (a blind uniform sample underrepresents rare
# classes like "toaster" or "hair drier"). Images are fetched individually
# (not via the full train2017.zip) from:
#   http://images.cocodataset.org/train2017/<image_id zero-padded to 12 digits>.jpg
TRAIN_SUBSET_SIZE = 20000  # ~3.1GB at COCO's ~156KB/image train2017 average, comfortably under a 5GB pull budget
TRAIN_SUBSET_SEED = 42
TRAIN_SUBSET_IMAGE_IDS_PATH = os.path.join(DATASET_DIR, "train_subset_image_ids.txt")
COCO_TRAIN_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/train2017/{image_id:012d}.jpg"

EXPECTED_IMAGE_COUNT["train2017"] = TRAIN_SUBSET_SIZE


def images_dir(split):
    return os.path.join(DATASET_DIR, split)


def annotations_json(split):
    return os.path.join(ANNOTATIONS_DIR, f"instances_{split}.json")


def raw_predictions_dir(split):
    return os.path.join(RESULTS_DIR, split, "raw_predictions")


def output_csv(split):
    return os.path.join(RESULTS_DIR, split, "coco_class_recall_benchmark.csv")


def results_dir(split):
    return os.path.join(RESULTS_DIR, split)


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------
# Pretrained COCO checkpoints, auto-downloaded by the `ultralytics` package on
# first load (cached under MODELS_DIR via YOLO(...) weights path below).
MODEL_POOL = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Significant VRAM headroom available -- batch inference per model rather than
# looping image-by-image. Lower this if a model OOMs on the target GPU.
INFERENCE_BATCH_SIZE = 512

IMG_SIZE = 640  # ultralytics default inference resolution
CONF_THRESHOLD = 0.25  # ultralytics default detection confidence threshold
DEVICE = "cuda:0"  # falls back to "cpu" automatically in run_benchmark.py if unavailable

# ---------------------------------------------------------------------------
# Output format note
# ---------------------------------------------------------------------------
# gt_classes / pred_classes columns in the output CSV are pipe-separated COCO
# category *names* (not numeric ids) -- easier to eyeball/spot-check per-image
# disagreement between models downstream without cross-referencing a category
# id table every time.
CLASS_LIST_SEPARATOR = "|"
MINIMUM_RECALL_THRESHOLD = 0.80