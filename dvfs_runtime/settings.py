# Every constant of the DVFS profiling: hardcoded frequency list, models, batches, timings, safety limits.
# There are no command line flags for these; edit here. Python 3.8 only (the board).
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_TAG = os.environ.get("DEVICE_TAG", "agx_xavier_jp511")
RESULTS_DIR = os.path.join(HERE, "results", DEVICE_TAG)
MODELS_DIR = os.path.join(HERE, "models")          # copy sched-harness/models/*.pt here to skip the download
INPUTS_DIR = os.path.join(HERE, "data", "inputs")  # optional images; else the two ultralytics samples are used

# ---- device ----
GPU_DEVFREQ = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b"
GPU_3D_SCALING = "/sys/devices/17000000.gv11b/enable_3d_scaling"
# python processes whose command line contains one of these block a run (system daemons like networkd-dispatcher do not); tegrastats always blocks
BUSY_PATTERNS = ("dvfs", "sched-harness", "jetson-power", "00_preflight", "01_calibrate", "02_run", "03_plan", "01_power_sweep",
                 "ultralytics", "yolo", "torch", "benchmark")
# nvpmodel mode 4 = MODE_30W_6CORE (the Oct 30 runs, GPU cap 905.25 MHz); mode 0 = MAXN (8 cores, GPU cap 1377 MHz). This tool never changes the mode.
# DVFS_EXPECTED_MODE=0 (or "any") to run in another mode; the mode and the clock cap are recorded in the run's meta file either way.
_EM = os.environ.get("DVFS_EXPECTED_MODE", "4")
EXPECTED_MODE_ID = None if _EM.lower() == "any" else int(_EM)
RUN_TAG = os.environ.get("DVFS_RUN_TAG", "")      # appended to the models_<stamp> folder name, e.g. "maxn"


# GPU clocks to profile, in Hz. Every entry must be in available_frequencies and not above the mode's max_freq,
# else the run refuses to start (checked before any clock is written). Run `python -m dvfs_runtime.board_info` for the real list.
# Board list (AGX Xavier, mode 4 cap 905.25 MHz): 114.75 216.75 318.75 420.75 522.75 624.75 675.75 828.75 905.25 (higher entries are
# outside the cap). Left out: 114.75 (8x slower than the cap, runs take many minutes), 624.75 and 828.75 (near a neighbour).
FREQS_HZ = [216750000, 318750000, 420750000, 522750000, 675750000, 905250000]
# Override without editing, in MHz, comma separated:  DVFS_FREQS_MHZ=420.75,522.75,...,1377   (still checked against available_frequencies and the cap)
if os.environ.get("DVFS_FREQS_MHZ"):
    FREQS_HZ = [int(round(float(x) * 1e6)) for x in os.environ["DVFS_FREQS_MHZ"].split(",")]

# ---- models and pipeline (same as sched-harness/config.py) ----
# FAMILY picks what is profiled:
#   "yolo"   the four YOLOv8 nets, NMS after the forward (the Oct 30 run; the default)
#   "resnet" torchvision ResNet classifiers with random weights (timing and power do not depend on the weight values),
#            no post-processing, 224x224 input
# Switch without editing this file:   DVFS_FAMILY=resnet STEPS=models sudo -E bash run_all.sh
FAMILY = os.environ.get("DVFS_FAMILY", "yolo")
if FAMILY == "yolo":
    QUEUES = ["n", "s", "m", "l"]
    MODELS = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "l": "yolov8l.pt"}
    LABELS = {"n": "yolov8n", "s": "yolov8s", "m": "yolov8m", "l": "yolov8l"}
    IMGSZ = 640
    EST_BATCH_S = 0.5           # rough mean batch time over the grid; only used for the --dry-run time estimate
    MIN_SUBRUN_S = 0.0          # a subrun runs SUBRUN_BATCHES batches, and at least this many seconds (0 = just the batches, as on Oct 30)
elif FAMILY == "resnet":
    QUEUES = ["r18", "r50", "r152"]          # add "r34" and/or "r101" here to run them as well
    TORCHVISION_NAMES = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50", "r101": "resnet101", "r152": "resnet152"}
    LABELS = dict(TORCHVISION_NAMES)
    IMGSZ = 224
    EST_BATCH_S = 0.1
    MIN_SUBRUN_S = 5.0          # ResNet batches are short: 20 of them would give a window with almost no power samples
    RANDOM_IMAGES = 10          # with no files in INPUTS_DIR the pool is this many seeded random-noise images (a classifier's compute ignores content)
else:
    raise ValueError("DVFS_FAMILY must be yolo or resnet, got %r" % FAMILY)
B_MAX = 16
INPUT_POOL_SIZE = 64
NMS_CONF = 0.25
NMS_IOU = 0.45
WARMUP_BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]     # run twice per model once at start, not timed

# ---- profile_models ----
BATCHES = [1, 2, 3, 4, 6, 8, 12, 16]
SUBRUNS = 4                     # subruns per (freq, model, batch) run
SUBRUN_BATCHES = 20             # every subrun runs exactly this many back-to-back batches, whatever they take
# EST_BATCH_S and MIN_SUBRUN_S are set per FAMILY above
SUBRUN_GAP_S = 10               # idle between subruns
RUN_GAP_S = 20                  # idle after the last subrun of a run (cooldown)
PIN_SETTLE_S = 1.0              # wait after pinning, before the warm-up
WARMUP_S = 2.0                  # discarded batches at the new clock: at least this long ...
WARMUP_MIN_BATCHES = 5          # ... and at least this many
GAP_SKIP_S = 2.0                # first seconds of a gap are left out of gap_power_w (power still decaying)
POWER_TAIL_S = 0.6              # wait after the last batch so tegrastats has a sample past the window
SEED = 0                        # shuffles the (freq, model, batch) order

# ---- profile_switch ----
SWITCH_REPEATS = 5              # per ordered pair of FREQS_HZ
SWITCH_WAIT_S = 5               # settle after pinning the "from" clock, and watch time after the switch
SWITCH_POLL_TIMEOUT_S = 2.0     # give up waiting for cur_freq to reach the target

# ---- frequency check ----
TEGRA_MATCH_TOL_MHZ = 1.5       # tegrastats GR3D_FREQ prints whole MHz; 599250000 Hz shows as 599 (gr3d_ok column)

# ---- power ----
RAILS = ["GPU", "CPU", "SOC", "CV", "VDDRQ", "SYS5V"]
TEGRASTATS_INTERVAL_MS = 50     # sched-harness got about 8.25 Hz at 100; `dvfs_runtime.board_info` measures 100 / 50 / 20 on this board
KEEP_SAMPLES_S = 120            # sampler samples older than this are dropped from RAM after each run (the gz log keeps all)
PRIMARY_POWER = "total_excl_sys5v"   # key of tegrastats.powers(); also reported: "total" and "gpu"
POWER_KEYS = ["total_excl_sys5v", "total", "gpu"]

# ---- parallel measurement of the held clock and of throttle messages (recorded, never acted on) ----
CUR_FREQ_INTERVAL_S = 0.02     # the sampler reads sysfs cur_freq this often
TEMP_IGNORE = ("PMIC",)         # temperatures are only recorded, never acted on; tegrastats prints PMIC@100C constantly on Xavier
DMESG_INTERVAL_S = 1.0          # how often the sampler reads dmesg for new lines
DMESG_PATTERN = r"over[- ]?current|throttl|thermal|overheat|emergency"
