# Every constant of the scheduling harness: paths, pool, calibration, traffic, run phases, plan, thresholds.
# There are no command line flags for these; edit here. Python 3.8 only (the board).
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_TAG = os.environ.get("DEVICE_TAG", "agx_xavier_jp511")
RESULTS_DIR = os.path.join(HERE, "results", DEVICE_TAG)
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PROFILE_PATH = os.path.join(RESULTS_DIR, "profile.json")
MODELS_DIR = os.path.join(HERE, "models")          # copy jetson-power/models/*.pt here to skip the download
INPUTS_DIR = os.path.join(HERE, "data", "inputs")  # optional images; else the two ultralytics samples are used

# ---- device state ----
# nvpmodel mode (name or id) for the whole session. None = do not change the mode, use the current one.
# The sweep note says a MAXN run took the board down once, so this is deliberately left for me to set.
SESSION_MODE = None
LOCK_CLOCKS = True              # jetson_clocks for the session, restored at the end
MODE_SETTLE_S = 10
NVPMODEL_CONF = "/etc/nvpmodel.conf"

# ---- models and pipeline ----
QUEUES = ["n", "s", "m", "l"]   # queue i is served only by model i
MODELS = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "l": "yolov8l.pt"}
IMGSZ = 640
B_MAX = 16
INPUT_POOL_SIZE = 64
NMS_CONF = 0.25
NMS_IOU = 0.45
WARMUP_BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]     # run twice per model before a run, not timed

# ---- calibration (01_calibrate.py) ----
CALIB_BATCHES = [1, 2, 3, 4, 6, 8, 12, 16]          # the largest must equal B_MAX
CALIB_WARMUP = 20
CALIB_TIMED = 100
SWITCH_BATCHES = 200
CALIB_IDLE_S = 20
PROFILE_STAT = "p50"            # which statistic of the profile the policies and the capacity use
NMS_DOMINATES_SHARE = 0.5       # flag when NMS is above this share of yolov8n service time

# ---- traffic ----
MIXES = {"balanced": [0.25, 0.25, 0.25, 0.25],
         "skewed": [0.528, 0.149, 0.083, 0.240],
         # skew ladder: share of the cheap model (n) or the heavy model (l), the rest split equally
         "n40": [0.40, 0.20, 0.20, 0.20], "n55": [0.55, 0.15, 0.15, 0.15],
         "n70": [0.70, 0.10, 0.10, 0.10], "n85": [0.85, 0.05, 0.05, 0.05],
         "l55": [0.15, 0.15, 0.15, 0.55], "l85": [0.05, 0.05, 0.05, 0.85]}
# order used by the mix-sweep plot: heavy-model dominated -> balanced -> cheap-model dominated
MIX_ORDER = ["l85", "l55", "balanced", "n40", "skewed", "n55", "n70", "n85"]
QUEUE_CAP = 512
PIN_CORES = True             # arrival process on the last online core, main process on the others

# ---- one run ----
WARMUP_S = 15
DURATION_S = 120
DRAIN_TIMEOUT_S = 60            # a drain that takes longer than this is cut and the run is flagged
SMOKE_DURATION_S = 20
SMOKE_LAM = 11.0                # req/s, total (balanced mix capacity is about 22)
SMOKE_MIX = "balanced"
SMOKE_POLICY = "lqf"
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
SETUP_S_ESTIMATE = 45           # per run: process start, loading four nets, warming them
DRAIN_S_ESTIMATE = 10
ACTIVE_TIERS = [6, 7]           # 6 = seeds 1 and 2 on the skewed mix, 7 = skew ladder (3 seeds); 3-5 = the seed-0 skewed sweeps (done)

TRITON_DELAYS = [0, 2, 5, 10, 20, 50]
# SLO deadlines in ms, tight to lenient. b=1 takes 24 / 36 / 72 / 110 ms for n / s / m / l, so 150 ms is nearly unattainable
# once anything waits. EdgeServing uses one as its deadline (--param slo_ms=...); every run also reports the share of requests
# above each of these (metrics viol_pct_<slo>), so all policies are compared against the same deadlines.
SLO_MS_GRID = [150, 300, 600, 1200, 2400]
# Offered load is a hardcoded total arrival rate Lambda in req/s per mix (Lambda_cap is only printed for reference).
# Skewed Lambda_cap was 27.4 req/s on the AGX Xavier, balanced 22.1: keep every value below its mix capacity.
TIER1 = {"mixes": ["skewed"], "lams": {"skewed": [8, 16, 24]}, "seeds": [0],
         "policies": [("fcfs_nobatch", {}), ("fcfs_batch", {}), ("lqf", {}),
                      ("triton", {"delay_ms": 0}), ("triton", {"delay_ms": 5}), ("triton", {"delay_ms": 20})]}
TIER2 = {"mixes": ["skewed", "balanced"], "seeds": [0, 1, 2],
         "lams": {"skewed": [6, 11, 16, 22, 25], "balanced": [4, 9, 13, 18, 20]},     # proposed, not reviewed
         "policies": ([("fcfs_nobatch", {}), ("fcfs_batch", {}), ("lqf", {})]
                      + [("triton", {"delay_ms": d}) for d in TRITON_DELAYS]
                      + [("fixed_batch_b", {"b": b}) for b in [2, 4, 8]])}
# Tier 3: EdgeServing at every SLO in SLO_MS_GRID (15 runs). Needs only itself, not tier 1.
TIER3 = {"mixes": ["skewed"], "lams": {"skewed": [8, 16, 24]}, "seeds": [0],
         "policies": [("edgeserving", {"slo_ms": s}) for s in SLO_MS_GRID]}
# Tier 4: Clockwork at every SLO (15 runs). Its cancelled requests are logged as dropped = 2 and count as deadline misses.
TIER4 = {"mixes": ["skewed"], "lams": {"skewed": [8, 16, 24]}, "seeds": [0],
         "policies": [("clockwork", {"slo_ms": s}) for s in SLO_MS_GRID]}
# Tier 5: Symphony at every SLO (15 runs). Not work-conserving: it holds batches back on purpose, so its latency sits near the SLO.
TIER5 = {"mixes": ["skewed"], "lams": {"skewed": [8, 16, 24]}, "seeds": [0],
         "policies": [("symphony", {"slo_ms": s}) for s in SLO_MS_GRID]}
# Tiers 6 and 7: the three paper policies only (no fcfs / fixed-batch / no-batch), at SLO 150 / 600 / 2400 ms.
# Run order is tier, then seed, mix, load, policy, so a stopped plan always leaves whole seeds behind. Resumable.
PAPER_POLICIES = [(p, {"slo_ms": s}) for p in ("edgeserving", "clockwork", "symphony") for s in (150, 600, 2400)]
# Tier 6: more seeds on the skewed mix (seed 0 came from tiers 3-5). 54 runs.
TIER6 = {"mixes": ["skewed"], "lams": {"skewed": [8, 16, 24]}, "seeds": [1, 2], "policies": PAPER_POLICIES}
# Tier 7: the skew ladder, the SAME loads for every mix (req/s), so mixes can be compared at each load.
# Lambda_cap in req/s: balanced 22.4, skewed 27.5, n40 25.8, n55 30.3, n70 36.8, n85 46.9, l55 16.7, l85 13.3,
# so 24 and 36 overload the heavy-model mixes (l55, l85) and 8 is light load for the cheap-model mixes.
# 8 mixes x 4 loads x 9 policies x 3 seeds = 864 runs (27 already done from tiers 3-5).
LADDER_LAMS = [8, 16, 24, 36]
TIER7 = {"mixes": ["skewed", "balanced", "n40", "n55", "n70", "n85", "l55", "l85"],
         "lams": {m: LADDER_LAMS for m in ["skewed", "balanced", "n40", "n55", "n70", "n85", "l55", "l85"]},
         "seeds": [0, 1, 2], "policies": PAPER_POLICIES}
TIERS = {1: TIER1, 2: TIER2, 3: TIER3, 4: TIER4, 5: TIER5, 6: TIER6, 7: TIER7}

# ---- validation thresholds (metrics.py) ----
CHECK_SERVICE_TOL = 0.10
CHECK_LITTLE_TOL = 0.05
CHECK_RATE_TOL = 0.05
CHECK_CADENCE_TOL = 0.20
CHECK_GAP_SAMPLES = 3
CHECK_ENERGY_TOL = 0.01

# ---- analysis ----
MIX_SWEEP_LAMS = [16, 24, 36]   # loads (req/s) that get a mix_sweep plot
PLOT_LAM_ENERGY_BAR = 16      # req/s, nearest available value is used
PLOT_LAM_QUEUE_P99 = 22
TIMELINE_S = 10
