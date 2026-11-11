# sched-harness

Real-time multi-queue scheduling harness for the Jetson AGX Xavier, ResNet classifiers only (r18, r50, r152; torchvision, random
weights, FP16, 224x224). One queue per model, seeded Poisson arrivals, one batch on the GPU at a time. A pluggable policy decides what
to run next and, for the DVFS policies, at which GPU clock. Logs latency, deadline misses, power, energy and clock switches.
The YOLO version of this harness (tiers 1 to 7, the Week 8 baselines) is kept in `../archive/sched-harness-yolo/`.
Python 3.8 only (the board). Constants are all in `config.py`.

## Run

1. Set MAXN by hand: `sudo nvpmodel -m 0`, wait about 30 s. The harness never switches modes.
2. In tmux, from this folder: `sudo -E bash run_resnet.sh` prints the plan size and stops; `GO=1 sudo -E bash run_resnet.sh` runs
   preflight, calibration (skipped if `profile.json` exists), two smoke runs and the plan. It resumes after a crash.
3. Grid overrides (space separated): `DVFS_ALPHAS`, `DVFS_SLOS`, `DVFS_LAMS`, `DVFS_SEEDS`.
   Pilot: `DVFS_SLOS="300" DVFS_SEEDS="0" GO=1 sudo -E bash run_resnet.sh`.

The DVFS policies read a latency/energy table (`config.DVFS_TABLE`, from `dvfs_runtime`). Policy notes: `common/dvfs.py`; tests:
`python3 tests/test_dvfs.py` and `python3 tests/test_policies.py` (no board needed).
