# All the knobs for the power sweep live here.

import os

HERE = os.path.dirname(os.path.abspath(__file__))       # yolo-analysis-a100/power_analysis
YOLO_DIR = os.path.dirname(HERE)                         # yolo-analysis-a100

# what to measure
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
BATCH_SIZES = [1] + list(range(2, 33, 2)) + [40, 48, 56, 64]   # same grid as batch_sweep_DENSE
IMG_SIZE = 640
SEED = 0                    # for the random order of the combos and the image picks

# timing of each phase (seconds)
BASELINE_S = 20.0           # idle power with nothing loaded, at the start of the sweep
IDLE_S = BASELINE_S         # idle power with the model loaded, same length as the baseline
NUM_RUN_PHASES = 5          # how many run phases per (model, batch)
FORWARDS_PER_PHASE = 200    # a run phase = this many forwards of the batch, however long that takes
                            # (on the A100: ~1 s for yolov8n batch 1, ~45 s for yolov8l batch 64)
BUFFER_S = 3.0              # sleep between run phases
END_TAIL_S = 10.0           # keep sampling after the last combo: nvidia-smi writes its file in ~4 KB
                            # chunks (~6 s of samples), and the last chunk can be lost when it is stopped

# power sampling (nvidia-smi)
NVIDIA_SMI = ["nvidia-smi"]
GPU_INDEX = "0"             # which GPU nvidia-smi reads; match it to CUDA_VISIBLE_DEVICES
SAMPLE_MS = 50
# power.draw is always there, but on the A100 it is a ~1 s average.
# If your driver supports it, "power.draw.instant" is the real instantaneous reading.
POWER_FIELD = "power.draw"

# where things live
IMAGES_DIR = os.path.join(YOLO_DIR, "dataset", "val2017")
MODELS_DIR = os.path.join(YOLO_DIR, "models")
OUT_DIR = os.path.join(YOLO_DIR, "results", "power_analysis")
WORKER_TIMEOUT_S = 900      # a stuck worker is killed after this long
