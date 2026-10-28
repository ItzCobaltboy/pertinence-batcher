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
EXPECTED_MODE_ID = 7            # nvpmodel mode 7 = MODE_15W_DESKTOP; this tool never changes the mode. None = accept any.

# GPU clocks to profile, in Hz. Every entry must be in available_frequencies and not above the mode's max_freq,
# else the run refuses to start (checked before any clock is written). Run `python -m dvfs.inspect` for the real list.
FREQS_HZ = [306000000, 408000000, 510000000, 599250000]

# ---- models and pipeline (same as sched-harness/config.py) ----
QUEUES = ["n", "s", "m", "l"]
MODELS = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "l": "yolov8l.pt"}
IMGSZ = 640
B_MAX = 16
INPUT_POOL_SIZE = 64
NMS_CONF = 0.25
NMS_IOU = 0.45
WARMUP_BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]     # run twice per model once at start, not timed

# ---- profile_models ----
BATCHES = [1, 2, 3, 4, 6, 8, 12, 16]
SUBRUNS = 4                     # subruns per (freq, model, batch) run
SUBRUN_BATCHES = 20             # every subrun runs exactly this many back-to-back batches, whatever they take
EST_BATCH_S = 0.5               # rough mean batch time over the grid; only used for the --dry-run time estimate
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
TEGRA_MATCH_TOL_MHZ = 1.5       # tegrastats GR3D_FREQ prints whole MHz; 599250000 Hz shows as 599
REQUIRE_TEGRASTATS_MATCH = True # stop if tegrastats GR3D_FREQ disagrees with the pinned clock under load

# ---- power ----
RAILS = ["GPU", "CPU", "SOC", "CV", "VDDRQ", "SYS5V"]
TEGRASTATS_INTERVAL_MS = 50     # sched-harness got about 8.25 Hz at 100; `dvfs.inspect` measures 100 / 50 / 20 on this board
KEEP_SAMPLES_S = 120            # sampler samples older than this are dropped from RAM after each run (the gz log keeps all)
PRIMARY_POWER = "total_excl_sys5v"   # key of tegrastats.powers(); also reported: "total" and "gpu"
POWER_KEYS = ["total_excl_sys5v", "total", "gpu"]

# ---- safety: only two run-time checks, the pinned clock is held and dmesg stays clean ----
TEMP_IGNORE = ("PMIC",)         # temperatures are only recorded, never acted on; tegrastats prints PMIC@100C constantly on Xavier
GUARD_PERIOD_S = 5              # how often dmesg is looked at (also once at the end of every subrun)
DMESG_PATTERN = r"over[- ]?current|throttl|thermal|overheat|emergency"
