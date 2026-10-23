# Every default for edge-runtime lives here. Scripts take command line flags that
# override these; nothing else in the folder hardcodes a path, a size or a duration.
# No torch import here: config is read by analyze.py and the tests too.

import os

HERE = os.path.dirname(os.path.abspath(__file__))           # edge-runtime/
REPO_ROOT = os.path.dirname(HERE)
YOLO_DISPATCHER_DIR = os.path.join(REPO_ROOT, "yolo-dispatcher")
CITYFLOW_ROUTING_DIR = os.path.join(REPO_ROOT, "cityflow-routing")

# ---------------------------------------------------------------------------
# Model pool (cost-ascending, the order Promote uses: dst must come later)
# ---------------------------------------------------------------------------
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
IMG_SIZE = 640

# COCO val2017 routing split at recall >= 0.80 (CLAUDE.md, scheduler-sim)
ROUTING_SPLIT = [0.528, 0.149, 0.083, 0.240]

# ---------------------------------------------------------------------------
# Calibration (T_i(b) on this device)
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 2, 4, 8, 16]      # add 32 with --batch-sizes if memory allows
B_MAX = 16                          # largest batch any policy may run
CALIB_WARMUP = 20
CALIB_ITERS = 100
SWITCH_ITERS = 100                  # n, l, n, l, ... at b = 1 vs back-to-back
PROFILE_STAT = "p50"                # which statistic policies plan with (--profile-stat)

# NMS settings: ultralytics predict() defaults
NMS = True
NMS_CONF = 0.25
NMS_IOU = 0.7

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
CAMERAS = 8
CAMERA_FPS = 10.0
STICKY_STAY = 0.9
LOAD_FRACTIONS = [0.2, 0.5, 0.8]    # of the device's capacity (calibrated)
SLOS_MS = [50, 100, 200]
TRACE_SEED = 0

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
FRAME_POOL_SIZE = 256
TENSOR_POOL_SIZE = 64               # pre-made tensors kept on the GPU in tensor mode
FRAMES_DIR = os.path.join(HERE, "data", "frames")
TRACES_DIR = os.path.join(HERE, "data", "traces")

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
RUN_DURATION_S = 120.0              # of trace; the drain comes on top
WARMUP_S = 10.0                     # dropped from metrics
IDLE_LOADED_S = 5.0                 # per run: power with models loaded, nothing running
DRAIN_TIMEOUT_S = 120.0             # give up draining after this long
MAX_QUEUED = 300                    # frames-mode jobs hold a GPU tensor each (~4.9 MB);
                                    # past this many queued jobs the run is marked collapsed

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
COOLDOWN_S = 15.0
REPEATS = 1
RUN_OVERHEAD_S = 40.0               # subprocess start + model load + idle phase + tail, for estimates
TEMP_WARN_C = 10.0                  # warn when a run starts this much hotter than the first
# visit order (section 12): the most informative policies first
VISIT_POLICIES = [
    ("lqf", {}),
    ("triton", {"max_queue_delay_ms": 2}),
    ("triton", {"max_queue_delay_ms": 10}),
    ("edgeserving", {}),
    ("clockwork", {}),
    ("symphony", {}),
    ("fcfs_nobatch", {}),
]
TRITON_DELAYS_MS = [0, 2, 5, 10, 20, 50]

# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------
SAMPLE_MS = 50
NVIDIA_SMI_TAIL_S = 10.0            # nvidia-smi writes its -f file in ~4 KB chunks
TEGRASTATS_TAIL_S = 1.0
POWER_BASELINE_S = 30.0             # power_profile: nothing loaded
POWER_IDLE_S = 20.0                 # power_profile: model loaded, idle
POWER_RUN_PHASES = 5
POWER_PHASE_MIN_S = 10.0            # a run phase lasts at least this long ...
POWER_PHASE_MIN_FORWARDS = 50       # ... and at least this many forwards
POWER_GAP_S = 3.0

RESULTS_DIR = os.path.join(HERE, "results")


def device_dir(device_tag):
    return os.path.join(RESULTS_DIR, device_tag)
