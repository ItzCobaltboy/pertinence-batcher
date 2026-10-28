# Settings for the Jetson power sweep: models, batch grid, durations, thresholds, paths.
# Everything the three scripts share lives here so a number is changed in one place.
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_TAG = os.environ.get("DEVICE_TAG", "agx_xavier_jp511")
RESULTS_DIR = os.path.join(HERE, "results", DEVICE_TAG)
MODELS_DIR = os.path.join(HERE, "models")

# short name -> stock COCO weights (downloaded by ultralytics into MODELS_DIR)
MODELS = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "l": "yolov8l.pt"}
IMGSZ = 640

BATCHES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

# ---- sweep settings (01_power_sweep.py reads these; edit here, there are no command line flags) ----
SMOKE = False               # True: current mode only, yolov8n, b=1 and 4, short. False = the full sweep.
# nvpmodel mode names (or ids); None = every mode in nvpmodel.conf. Low-power modes only for now: the
# supply on this board is non-standard and a MAXN run (~36 W) took the board down once.
SWEEP_MODES = ["MODE_10W", "MODE_15W", "MODE_15W_DESKTOP"]
SWEEP_MODELS = ["n", "s", "m", "l"]
LOCK_CLOCKS = True          # False = DVFS variant of the same sweep
BUDGET_MIN = None           # e.g. 120 shrinks the grid (batches/reps) to fit; None = use BATCHES and REPS as is
# Resume is always on: finished cells in status.csv are skipped. To start over, delete the results folder.

# power rails printed by tegrastats (only when run as root). No total rail exists.
RAILS = ["GPU", "CPU", "SOC", "CV", "VDDRQ", "SYS5V"]
TEGRASTATS_INTERVAL_MS = 100
SAMPLER_TAIL_S = 3          # keep sampling this long after the last mark (output is buffered)

# measurement
WARMUP_ITERS = 10
REPS = 3
MIN_SECONDS = 10            # per repetition
MIN_ITERS = 30              # per repetition
CHUNK_ITERS = 10            # sync every this many iterations to check the clock
GAP_S = 3                   # pause between repetitions
IDLE_S = 20                 # idle baseline length (no model, and model loaded)
MODE_SETTLE_S = 10

# thermal guard (degrees C)
THERMAL_PAUSE_ABOVE = 80
THERMAL_RESUME_BELOW = 65

# a cell is flagged in the analysis when its repetitions differ by more than this
REP_SPREAD_FLAG = 0.10

NVPMODEL_CONF = "/etc/nvpmodel.conf"
