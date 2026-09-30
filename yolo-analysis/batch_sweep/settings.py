"""
Every default for the batched-inference timing sweep lives here. Nothing is
hardcoded in the other files; the CLI in sweep.py can override most of it.

Paths, the model pool, image size and the confidence threshold come from
yolo-analysis/config.py, so this sweep uses the exact same dataset folders,
checkpoints and preprocessing size as the recall benchmark. The repo is
assumed to be cloned in full, so plain relative paths from this file work.

This module imports no torch, so the aggregation side can run on a laptop.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # yolo-analysis/batch_sweep
YOLO_DIR = os.path.dirname(HERE)                         # yolo-analysis
REPO_ROOT = os.path.dirname(YOLO_DIR)

if YOLO_DIR not in sys.path:
    sys.path.insert(0, YOLO_DIR)
import config as yolo_config  # noqa: E402  (yolo-analysis/config.py)

# ---------------------------------------------------------------------------
# What gets measured
# ---------------------------------------------------------------------------
MODELS = list(yolo_config.MODEL_POOL)          # yolov8n/s/m/l .pt
BATCH_SIZES = [1, 2, 4, 8, 12, 16, 32, 48]

# eager_fp32: plain PyTorch, default torch settings (same as yolo-analysis)
# trt_fp32:   torch_tensorrt.compile(ir="dynamo"), FP32 only
# trt_fp16:   same, FP16 enabled
VARIANTS = ["eager_fp32", "trt_fp32", "trt_fp16"]
TRT_VARIANTS = ["trt_fp32", "trt_fp16"]

WARMUP_BATCHES = 20      # untimed forwards per run, before the timed loop
TIMED_BATCHES = 100      # timed forwards per run (one timing each)
RUNS = 5                 # independent runs per (variant, model, batch size)

# ---------------------------------------------------------------------------
# Data (real COCO images, loaded + letterboxed before any timing)
# ---------------------------------------------------------------------------
SPLITS = list(yolo_config.SPLITS)   # val2017 + train2017 subset
POOL_PER_SPLIT = 128                # images sampled per split into the pool
MAX_DISTINCT_BATCHES = 8            # pre-built GPU batches per run, cycled
IMG_SIZE = yolo_config.IMG_SIZE     # 640
CONF_THRESHOLD = yolo_config.CONF_THRESHOLD  # 0.25, only used by the NMS column
IOU_THRESHOLD = 0.7                 # ultralytics predict default
MAX_DET = 300                       # ultralytics predict default
SEED = 0

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
# BATCH_SWEEP_OUT_DIR is for CPU tests only, so they never write into the real results folder.
OUT_DIR = os.environ.get("BATCH_SWEEP_OUT_DIR") or os.path.join(YOLO_DIR, "results", "batch_sweep")
RAW_DIR = os.path.join(OUT_DIR, "raw")            # one JSON per run, never overwritten
SUMMARY_DIR = os.path.join(OUT_DIR, "summary")    # regenerated from raw/ by `aggregate`
LOG_DIR = os.path.join(OUT_DIR, "logs")
GPU_LOG_DIR = os.path.join(OUT_DIR, "gpu_logs")
ENGINE_DIR = os.path.join(OUT_DIR, "engines")     # cached TRT modules, git-ignored, not copied back
PROFILE_DIR = os.path.join(REPO_ROOT, "scheduler-sim", "profiles")
PROFILE_HARDWARE_TAG = "a100"                     # -> yolov8_a100_<variant>.csv

DEFAULT_SESSION = "main"

# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
COMPILE_TIMEOUT_S = 3600   # one TRT compile, killed and recorded as failed past this
RUN_TIMEOUT_S = 1800       # one timed run
MAX_ATTEMPTS = 2           # a failed job is retried on a later invocation until this many failures

# ---------------------------------------------------------------------------
# Sanity flags
# ---------------------------------------------------------------------------
STD_FLAG_FRACTION = 0.05       # std across runs above 5% of the mean
CLOCK_SAG_FRACTION = 0.90      # SM clock at run end below 90% of max SM clock
HOT_TEMPERATURE_C = 80


def images_dir(split, dataset_root=None):
    """Same folder yolo-analysis reads; --dataset-root swaps the parent dir."""
    if dataset_root is None:
        return yolo_config.images_dir(split)
    return os.path.join(dataset_root, split)


def weights_candidates(model_name):
    """Where yolo-analysis may have left a checkpoint. Last resort: the bare
    name, which ultralytics downloads (or, for a .yaml, builds untrained)."""
    return [
        os.path.join(yolo_config.MODELS_DIR, model_name),
        os.path.join(YOLO_DIR, model_name),
        model_name,
    ]


def model_stem(model_name):
    return os.path.splitext(os.path.basename(model_name))[0]
