# sched-harness

Real-time multi-queue scheduling harness for the Jetson AGX Xavier. One queue per YOLOv8 model (n, s, m, l),
seeded Poisson arrivals, eager FP16, all four nets resident, one batch on the GPU at a time. A pluggable policy
decides what to run next; the harness records mean and p99 latency, total energy and energy per request.
Spec: `research/spec_harness.md`. Python 3.8 only (the board).

Written on the laptop and never run there; every check below runs on the board.

## Setup

Copy this folder to the board. To skip the weight download, copy `jetson-power/models/*.pt` to `sched-harness/models/`.
Optional: put images in `sched-harness/data/inputs/` (else the two ultralytics samples are used, with a notice).
Set `SESSION_MODE` in `config.py` (None = leave the power mode as it is). All commands run from `sched-harness/` with sudo
(tegrastats rails need root):

## Commands, in order

| step | command | wall time |
|---|---|---|
| preflight | `sudo python3 00_preflight.py` | < 1 min plus model load |
| calibrate | `sudo python3 01_calibrate.py` | about 10 to 15 min |
| smoke run | `sudo python3 02_run.py --smoke` | about 1.5 min |
| tier 1 plan | `sudo python3 03_plan.py` | 18 runs x about 3.5 min = about 1 h 5 min |
| analysis | `python3 04_analyze.py` (laptop or board) | under a minute |

**One command for the whole chain:** inside `tmux`, `sudo bash run_all.sh` runs preflight, calibration, the smoke run and (only if it
prints `SMOKE PASS`) the Tier 1 plan, stopping at the first failure. Logs go to `logs/`.

`02_run.py` alone takes `--policy`, `--param key=value`, `--mix`, `--lam` (total req/s), `--seed`, `--out`. `03_plan.py` runs the
tiers in `config.ACTIVE_TIERS` (tier 1 by default; add 2 once tier 1 looks sane) and drops runs from the end when
`BUDGET_MIN` is set. It is resumable: finished runs (`run.json` with `status = "done"`) are skipped.
Mode and clocks are put back at the end of every script, also on error or Ctrl+C.

Copy results back: `results/<device_tag>/` (keep `profile.json`, `preflight.json`, `runs/`, `analysis/`).
Raw power logs are gitignored.

## What gets measured

- Service time of a batch = stack + H2D + convert, forward, NMS (each CUDA-synchronised). Calibration measures
  exactly this pipeline, so profile and runtime agree.
- Latency = `t_batch_end - t_arrival_target` (arrival lateness is charged to the system).
- Total energy = board power integrated from the first arrival of the measurement window to the completion of the
  last job of that window (drain included), idle time included. Primary power definition is the sum of rails
  excluding SYS5V (`PRIMARY_POWER`); the other two are in `summary.csv`.
- Busy/idle split: power is sampled every 100 ms and a batch lasts 10 to 100 ms, so busy energy is a coarse
  attribution (interpolated between samples). Read it as a trend, the total is the solid number.
- Offered load is a hardcoded total arrival rate `--lam` in req/s per mix. `Lambda_cap` (computed from full `B_MAX` batches) is printed for reference only, so loads stay fixed when the profile is redone. The measured GPU busy fraction is in the summary.

## Validation checks (`analysis/validation.md`, claim 3)

1. Every generated job is finished or dropped (count equality).
2. Batches never overlap on the GPU.
3. Mean in-run service time per (queue, b) is within 10% of the calibration p50 (worst offender reported). Only (queue, b) with at least 5 batches.
4. Little's law: time-average jobs in system over the window (from the log) vs throughput x mean latency, within 5%.
5. Measured arrival rate per queue vs lambda_i, within 5% or 3 sigma of Poisson counting noise, whichever is larger.
6. Arrival lateness p99 below 5 ms (else the run is flagged).
7. tegrastats cadence within 20% of the setting, no gap over 3 sample intervals, trace covers `[t_first, t_last]`.
8. Energy bookkeeping: busy + idle = total within 1%, and `power.csv` agrees with a recomputation from `power_raw.log.gz` within 1%.
9. Same arrival list and job set across policies for the same mix, lam and seed (arrivals hash). Analysis only.

`02_run.py --smoke` runs checks 1 to 8 on one 20 s `lqf` run (lam 11 req/s, balanced) and prints PASS or FAIL.

## Policies (`common/policies.py`)

`fcfs_nobatch`, `fcfs_batch`, `lqf`, `triton` (param `delay_ms`), `fixed_batch_b` (param `b`). Stubs that raise until
built: `symphony`, `clockwork`, `edgeserving` (these need an `slo_ms` run parameter first) and `index_T` (my policy,
formula not specified yet). A policy is a class with `decide(state)` returning `Serve(queue, b)` or `Wait(until)`.
`python3 tests/test_policies.py` checks the decision logic on hand-built states (no torch).

## Faithfulness notes

- `common/tegrastats.py` and `common/powermode.py` are copies of `jetson-power/common/` with no logic changes
  (validated on the board there). `jetson-power/` is untracked in git so there is no commit hash to record; copied 2026-10-24.
- `fixed_batch_b` flushes the leftover jobs once arrivals have stopped (otherwise the drain would never end).
- Triton-style here means per-queue timer + full batch cap, no preferred batch sizes and no multiple instances.
- A drain longer than `DRAIN_TIMEOUT_S` (60 s) is cut and the run is flagged.
- Two processes: the job maker (`ArrivalProcess`, no torch, pinned to the last online core) sends jobs through a pipe; the
  scheduler, runtime and GPU work stay in the main process on the other cores. This replaced an arrival thread that was
  up to 150 ms late under the GIL on the board (smoke checks 6). It is forked before any CUDA call.
- NMS runs through ultralytics. The board's torchvision wheel does not match NVIDIA's torch (its C++ ops do not load), so
  `models.setup_nms()` probes `torchvision.ops.nms` and, if it fails, swaps in `nms_plain_torch` (pairwise IoU on the GPU,
  greedy pass on the CPU). The backend used is saved as `nms_backend` in `run.json` and `profile.json`.
