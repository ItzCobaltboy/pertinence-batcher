# Every constant of the scheduling harness: paths, models, calibration, traffic, run phases, plan, thresholds.
# ResNet classifiers only (the YOLO version of this harness is in ../archive/sched-harness-yolo).
# There are no command line flags for these; edit here. Python 3.8 only (the board).
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_TAG = os.environ.get("DEVICE_TAG", "agx_xavier_jp511_resnet")
RESULTS_DIR = os.path.join(HERE, "results", DEVICE_TAG)
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PROFILE_PATH = os.path.join(RESULTS_DIR, "profile.json")
MODELS_DIR = os.path.join(HERE, "models")          # only used for chown after a session; torchvision ResNets need no files

# ---- device state ----
# The harness never switches the nvpmodel mode: set MAXN by hand first (sudo nvpmodel -m 0), as for the dvfs_runtime table.
# run_resnet.sh checks it. None = leave the mode as it is.
SESSION_MODE = None
LOCK_CLOCKS = False             # jetson_clocks would pin the GPU at max; the DVFS policy owns the GPU clock
MODE_SETTLE_S = 10
NVPMODEL_CONF = "/etc/nvpmodel.conf"

# ---- models and pipeline ----
QUEUES = ["r18", "r50", "r152"]     # queue i is served only by model i
MODELS = {"r18": "resnet18", "r50": "resnet50", "r152": "resnet152"}      # torchvision names, random weights, FP16
IMGSZ = 224
B_MAX = 16
INPUT_POOL_SIZE = 128           # random CIFAR-10 test images, upscaled to IMGSZ once at start-up (a job picks one at random)
CIFAR_NPZ = os.path.join(HERE, "data", "cifar10_128.npz")      # built once by make_cifar128.py, copy it to the board
WARMUP_BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]     # run twice per model before a run, not timed

# ---- calibration (01_calibrate.py), done with the GPU pinned at the top clock ----
CALIB_BATCHES = [1, 2, 3, 4, 6, 8, 12, 16]          # the largest must equal B_MAX
CALIB_WARMUP = 20
CALIB_TIMED = 100
SWITCH_BATCHES = 200
CALIB_IDLE_S = 20
PROFILE_STAT = "p50"            # which statistic of the profile the policies and the capacity use

# ---- traffic ----
MIXES = {"balanced3": [1 / 3.0, 1 / 3.0, 1 / 3.0], "light3": [0.6, 0.3, 0.1]}
MIX_ORDER = ["light3", "balanced3"]
QUEUE_CAP = 512
PIN_CORES = True             # arrival process on the last online core, main process on the others

# ---- one run ----
WARMUP_S = 10
DURATION_S = int(os.environ.get("DURATION_S", "1800"))      # measured window: 30 min per run, a long job sequence (env DURATION_S for quick tests)
DRAIN_TIMEOUT_S = 60            # a drain that takes longer than this is cut and the run is flagged
SMOKE_DURATION_S = 20
SMOKE_LAM = 60.0                # req/s, total
SMOKE_MIX = "balanced3"
SMOKE_POLICY = "lqf"            # dvfs_alpha needs params: --policy dvfs_alpha --param alpha=0.5 --param slo_ms=300
LATENESS_P99_FLAG_MS = 5.0

# ---- power ----
RAILS = ["GPU", "CPU", "SOC", "CV", "VDDRQ", "SYS5V"]
TEGRASTATS_INTERVAL_MS = 100
SAMPLER_TAIL_S = 3
# the three board-power definitions the sweep uses; column name in power.csv
POWER_COLS = {"total": "total_mw", "total_excl_sys5v": "total_excl_sys5v_mw", "gpu": "gpu_mw"}
PRIMARY_POWER = "total_excl_sys5v"   # until I say otherwise (SYS5V overlap is unchecked on the board)

# ---- thermal guard (degrees C) ----
THERMAL_PAUSE_ABOVE = 80
THERMAL_RESUME_BELOW = 65

# ---- plan (03_plan.py) ----
COOLDOWN_S = 20
BUDGET_MIN = None               # e.g. 90 drops the lowest-priority runs to fit; None = run everything
SETUP_S_ESTIMATE = 45           # per run: process start, loading the nets, warming them
DRAIN_S_ESTIMATE = 10
ACTIVE_TIERS = [1]
# SLO deadlines in ms; every run reports the share of requests above each of them (metrics viol_pct_<slo>).
SLO_MS_GRID = [150, 300, 600, 900, 1200, 1500, 2400]

# ---- DVFS study: GPU clock policy ----
DVFS_SWITCH_MS = 88.0           # measured in mode 4 (30 W); NOT measured in MAXN, remeasure before trusting it
DVFS_SLO_MS = 900               # default deadline allowance D; the sweep overrides it with DVFS_SLOS
DVFS_TABLE = os.environ.get("DVFS_TABLE", os.path.join(
    HERE, "..", "dvfs_runtime", "results", "agx_xavier_jp511", "models_20261031_100223_resnet", "run_summary.csv"))
# Hard ceiling for the GPU clock: the original PSU cannot carry MAXN at 1377 MHz with the CPU busy (the board powered off in the
# stress test), so the study tops out at 1237 MHz (1339 only saves about 1 W over 1377; 1237 saves about 4 W). The policy's table is cut to this, and Clock refuses anything above it.
DVFS_MAX_MHZ = int(os.environ.get("DVFS_MAX_MHZ", "1237"))
DVFS_BASE_MHZ = [829, 1033, 1198]       # fixed-clock baselines (dvfs_fixed), run at every SLO: same queue rule, same dropping of expired jobs
DVFS_DROP_EXPIRED = os.environ.get("DVFS_DROP_EXPIRED", "1") != "0"      # dvfs_alpha cancels jobs whose deadline has passed


def _words(name, default, cast):
    return [cast(x) for x in os.environ.get(name, default).split()]


def _num(x):
    return float(x) if "." in x else int(x)


# Sweep axes, overridable from the shell (space separated), e.g. DVFS_SLOS="300 600" DVFS_SEEDS="0".
# Lambda_cap at the 1237 MHz ceiling from the table (balanced3, b = 16, p50) is about 200 req/s; the default loads (100 200 300) are for the light3 mix, whose Lambda_cap is about 327 (31%, 61%, 92%).
DVFS_ALPHAS = _words("DVFS_ALPHAS", "0 0.25 0.5 0.75 1", float)
DVFS_SLOS = _words("DVFS_SLOS", "900 300", int)
DVFS_LAMS = _words("DVFS_LAMS", "100 200 300", _num)
DVFS_SEEDS = _words("DVFS_SEEDS", "0 1 2", int)

# Run order is tier, then seed, mix, load, policy, so a stopped plan always leaves whole seeds behind. Resumable.
TIER1 = {"mixes": ["light3"], "lams": {"light3": DVFS_LAMS}, "seeds": DVFS_SEEDS,       # light3 = 60/30/10 over r18/r50/r152
         "policies": ([("dvfs_alpha", {"alpha": a, "slo_ms": slo}) for slo in DVFS_SLOS for a in DVFS_ALPHAS]
                      + [("dvfs_fixed", {"freq_mhz": f, "slo_ms": slo}) for slo in DVFS_SLOS for f in DVFS_BASE_MHZ])}
TIERS = {1: TIER1}

# ---- validation thresholds (metrics.py) ----
CHECK_SERVICE_TOL = 0.10
CHECK_LITTLE_TOL = 0.05
CHECK_RATE_TOL = 0.05
CHECK_CADENCE_TOL = 0.20
CHECK_GAP_SAMPLES = 3
CHECK_ENERGY_TOL = 0.01

# ---- analysis ----
MIX_SWEEP_LAMS = [100, 200, 300]
PLOT_LAM_ENERGY_BAR = 200
PLOT_LAM_QUEUE_P99 = 200
TIMELINE_S = 10
