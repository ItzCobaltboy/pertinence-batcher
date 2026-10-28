# Spec: `sched-harness/`, a real-time multi-queue scheduling harness for the Jetson AGX Xavier

For Claude Code on my laptop. You write the code, I copy the folder to the Jetson and run it there. You cannot
run it, so the code has to be right by reading, and every check runs on the board (`00_preflight.py`, `--smoke`).
Read this whole file first. It replaces `research/spec_edge_runtime.md` as the thing to build this week
(that spec is bigger, see section 12 for what to take from it and what to leave).

## 0. Goal, and what this harness must prove

I need to hand over one defendable piece: "this scheduling problem exists on real edge hardware, here is the
measured proof". The harness produces that proof. Real YOLOv8 models run on the real GPU under live Poisson
traffic, a pluggable policy decides what to run next, and I measure **mean and p99 latency, total energy and
energy per request** for each policy. The goal this serves: show a policy that serves the same jobs for less energy.

It must back three claims with numbers:

1. **Batching changes latency and energy a lot on this board**, so the scheduling decision matters.
   (Exhibit: `01_calibrate.py` service-time curves plus total energy and energy per request vs load.)
2. **Standard policies leave something on the table here.** Concretely: for the same load, show where
   LQF / Triton-style batching lose against each other or against the best policy in the grid (latency, energy,
   or a queue that starves). (Exhibit: latency vs energy Pareto across policies, per-queue latency.)
3. **The harness itself is trustworthy.** Measured in-run service times match calibration; queueing numbers
   obey Little's law; no job is lost. Without this the other two claims are not defendable. (Exhibit:
   `validation.md` produced by the analysis, section 9.)

### What the sweep already shows (partial, MAXN, yolov8n, p50 batch latency; read from `jetson-power/results`)

| batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| latency ms | 12.5 | 15.5 | 28.2 | 53.7 |
| energy per frame mJ | 292 | 235 | 224 | 222 |

So on this board batch time is **not** size-independent: it is roughly `a + c*b` (a launch-bound floor, then
linear growth). Batching saves energy mostly by amortising the floor, and the saving is gone by b=4 for n. The
harness must therefore never assume a cost model; policies read the measured table (section 5).

## 1. Rules

- **Hard stop (my standing rule, laptop side):** never touch C drive system areas, Program Files or the Windows
  folder. Work only in the project folder.
- Python **3.8.10 only** (the board). No `list[int]`/`dict[str, int]` annotations, no `X | Y`, no `match`, no
  `removeprefix`, no `dict | dict`, no parenthesised multi-line `with`.
- Style: same rules as section 1b of `research/spec_jetson_power.md` (plain readable Python, one idea per
  function, no decorators/inheritance/`**kwargs` pass-through, comments say why). Policies are small separate
  classes with a `decide(state)` method, nothing inherits from anything. pandas only in the analysis script.
- No em dashes anywhere you write.
- **No fake testing.** No mocked models, no mocked power. The only allowed checks without the board are plain
  asserts on policy decision logic over hand-built queue states (no torch needed) and the tegrastats parser on
  text lines of real output (already done in the copied module). Everything else is checked on the board.
- **Flag, do not guess.** An unknown fact goes under Flags in your summary and the code stops with a clear
  message instead of continuing on a guess.
- Do not edit `Journel/` or `CLAUDE.md`. Put suggested lines for CLAUDE.md in your summary instead.
- Do not import from `archive/`. Section 12 lists what to read there.
- On the board sudo needs a password, so `sudo -n` fails and I start the process with sudo myself. tegrastats
  rails need root. The code must restore the original power mode and clocks at the end, also on error or Ctrl+C
  (`try/finally`).

## 2. Decisions already made (do not reopen)

| topic | decision |
|---|---|
| classes / queues | **One queue per model**: n, s, m, l. A job in queue i can only be served by model i, batches never mix queues. This is the Xia et al. / Chen and Wang setting (incompatible classes). No promotion, no routing in this version. |
| arrivals | **Synthetic Poisson per queue**, rate lambda_i = Lambda * w_i, seeded. Same seed gives the identical arrival list for every policy. |
| models | YOLOv8 n/s/m/l, stock COCO weights, **eager PyTorch FP16**, fused, as in `jetson-power/common/bench.py:load_net`. No TensorRT. |
| residency | **All four models resident on the GPU for the whole run.** No load/unload decisions. |
| inputs / preprocessing | **Nothing camera or video specific.** A job is "one input for model i". Inputs come from a small pool of ready 640x640 uint8 arrays built once before the run (section 4); there is no live decode or letterbox. The scheduler is generic. |
| GPU concurrency | One batch on the GPU at a time. No streams, no MPS. |
| metrics | Mean latency, p99 latency (also p50, p95), **total energy** (first arrival to last completion, section 9), **energy per request** (= total energy / requests served). No deadlines or SLO in this version. Per-job lifecycle energy is NOT in this version. |
| device state | One power mode per session, clocks locked with `jetson_clocks` (same as the sweep), recorded in `run.json`. Default mode: whichever the sweep showed as stable and I name in `config.py` (see Flags). |

## 3. Layout

```
sched-harness/
  README.md                 how to run, visit procedure, faithfulness notes
  config.py                 every constant (paths, pool, loads, mixes, durations, thresholds)
  00_preflight.py           environment + one real forward + a 3 s parsed power sample
  01_calibrate.py           service-time table T_i(b) for the whole serving pipeline -> profile.json
  02_run.py                 ONE run: one policy, one load, one mix, one seed
  03_plan.py                grid of runs, resumable, subprocess per run, time budget
  04_analyze.py             metrics, plots, validation.md (laptop or board, no torch)
  common/
    models.py               load 4 nets (copy load_net from jetson-power), forward, NMS
    inputs.py               ready input pool, batch assembly (stack, H2D, convert)
    traffic.py              Poisson arrival lists per queue, capacity and load maths
    runtime.py              arrival thread, queues, executor loop, logging
    policies.py             policies (section 6)
    power.py                COPY of jetson-power/common/tegrastats.py and powermode.py, logic unchanged
  results/<device_tag>/     see section 10
  tests/test_policies.py    plain asserts on hand-built states
```

`common/power.py` is a copy because `jetson-power` has its own `common` and `config`, which would clash on
import. Do not change the parsing or energy logic (validated on the board); note in README which commit it came
from.

## 4. Input pool and batch pipeline (`common/inputs.py`)

The scheduler is generic, so a job is just "one input for model i". There is no camera, video or decode step.

- **Input pool:** 64 ready arrays, each `uint8` 640x640x3, built **once before the run** and kept in RAM. Source:
  images in `sched-harness/data/inputs/` if the folder has any (letterboxed once with
  `jetson-power/common/bench.py:letterbox`), else the two ultralytics sample images (`bench.py:sample_images`)
  cycled to 64 entries, with a loud notice. Record which in `run.json`. Preparing the pool is not part of the
  run and is not timed. Content matters only for NMS time (box count), so say which source was used.
- **Per job at arrival** (arrival thread): the Job just holds a reference to `pool[pool_index]`. No copy, no
  CPU work beyond creating the Job and enqueueing it.
- **In the executor, as part of the batch**: stack the b referenced arrays into a pre-allocated pinned `uint8`
  buffer (one per model, size B_max), copy to the GPU, BGR to RGB, to FP16, /255, forward, then NMS. Record
  `t_h2d_ms` (stack + copy + convert), `t_forward_ms` (CUDA-synchronised), `t_nms_ms` separately.
  **Batch service time = h2d + forward + nms.** The calibration in section 5 measures exactly this pipeline, so
  profile and runtime agree.
- NMS: `ultralytics.utils.ops.non_max_suppression` with conf 0.25, iou 0.45, on the raw output. The result is
  discarded after timing, but the call must really run (it is the model's own post-processing).
- The arrival thread is a hidden cost source anyway (Python sleep jitter and GIL contention with the executor).
  Log `arrival_lateness_ms` (actual enqueue time minus target). If its p99 exceeds 5 ms in a run, the run is
  flagged in `run.json`.

## 5. Calibration (`01_calibrate.py`)

Measured on the board, never copied from the A100 or from the power sweep.
- For each model n/s/m/l and each b in `[1, 2, 3, 4, 6, 8, 12, 16]` (configurable): 20 warmup batches, then 100
  timed batches of the **full pipeline** from section 4 (h2d + forward + NMS, CUDA-synchronised), batches built
  from the input pool. Save mean, std, p50, p95, p99 of the service time and of its three parts.
- **Switch cost:** alternate models at b=1 (n, l, n, l, ... 200 batches) versus 200 back-to-back same-model
  batches; report the difference. All models are resident, so this measures cache and launch effects only.
- **Idle power with all four models loaded**, 20 s, with tegrastats running. Saved as `p_idle_loaded_mw`
  (board sum of rails, and the sum excluding SYS5V, both).
- **Fit** `T_i(b) = a_i + c_i * b` by least squares per model; report the fit residual per b. This is the
  evidence for or against the size-independent assumption. Print a plain-language line: "batch time grows by
  X ms per frame, floor Y ms".
- Output `profile.json`: `{model: {"b": {p50_ms, p95_ms, mean_ms, parts...}, "fit": {a, c, max_rel_err}}}` plus
  device metadata and `p_idle_loaded_mw`. Policies use p50 by default (`PROFILE_STAT` in config).
- Lookup for a b that was not profiled: linear interpolation between the neighbours, never extrapolate.

## 6. Policies (`common/policies.py`)

Interface: `decide(state) -> Serve(queue, b) | Wait(until_time)`. `state` is a read-only snapshot taken by the
executor under the lock: `now` (monotonic), for each queue the number of jobs and the list of arrival times of
its jobs (oldest first), the profile lookup `service_ms(queue, b)`, `B_max`. A policy may keep its own state
between calls. `Wait(until)` means: sleep until that time or until the next arrival, whichever is first. If all
queues are empty the executor blocks until an arrival (no policy call). b must satisfy `1 <= b <= min(len, B_max)`.

First set (build all of these):

1. `fcfs_nobatch`: queue holding the oldest job, b=1.
2. `fcfs_batch`: queue holding the oldest job, b=min(len, B_max).
3. `lqf`: longest queue (tie: oldest head), b=min(len, B_max). Xia et al. 2002.
4. `triton`: per-queue timer. A queue is ready when `len >= B_max` or its oldest job has waited
   `>= max_queue_delay_ms`. Among ready queues serve the one with the oldest head, b=min(len, B_max). If none
   is ready, `Wait` until the earliest timeout. Plan grid: `max_queue_delay_ms` in `[0, 2, 5, 10, 20, 50]`.
5. `fixed_batch_b`: serve the queue with the oldest job once it holds `>= b` jobs (parameter b), else Wait for
   an arrival. Sanity baseline for "what does a fixed target batch size cost".

Second set (stub the interface now, implement only after the first set has produced results):
`symphony` (deferred batching), `clockwork` (INFER scheduling), `edgeserving`. These need deadlines; add an
`slo_ms` run parameter then, and take details from the papers or from the archived implementations (section 12).

My own policy goes in later through the same interface. Reserve the name `index_T`: serve the (queue, b) that
maximises a service-time-aware index built from the calibrated T_i(b). **Do not invent the formula.** I will
specify it after seeing the data.

Policy tests (`tests/test_policies.py`, plain asserts, no torch): for each policy, a few hand-built states with
the known correct decision (Triton timer fires and not before, LQF tie-break, fcfs picks the right queue,
`fixed_batch_b` waits below b).

## 7. Runtime (`common/runtime.py`)

- **Arrival schedule:** built before the run from `traffic.py` (section 8) as a list of `(t_arrival_s, queue,
  pool_index)`, sorted. Same list for every policy given the seed.
- **Arrival thread:** sleeps to each absolute target using `time.monotonic()` (a target time, not cumulative
  sleeps), creates the Job (a reference into the input pool), appends it to its queue, notifies the
  executor. Cap each queue at `QUEUE_CAP` (default 512); a job arriving at a full queue is counted as an
  `overflow` drop and logged (it should not happen at loads <= 0.9; if it does the run is flagged).
- **Executor thread** (owns the GPU, one batch at a time): ask the policy, validate the decision (raise on an
  invalid one), pop the b oldest jobs, run the pipeline, stamp `t_start` before h2d and `t_end` after NMS
  (CUDA-synchronised), mark jobs done. Time every `decide()` call and log it.
- Threads share one lock and one condition variable. Python `threading` only.
- **Run phases:** warmup `WARMUP_S` (default 15 s, jobs run but are excluded from metrics) -> measurement
  `DURATION_S` (default 120 s) -> arrivals stop -> drain with the same policy -> end. The drain is excluded from
  rate metrics but its jobs still count for latency if they arrived in the measurement window.
- **Power:** start `Sampler` (from `common/power.py`, interval `TEGRASTATS_INTERVAL_MS`, default 100) before the
  warmup and stop `SAMPLER_TAIL_S` after the drain. All timestamps are `time.monotonic()`, which is also what
  the Sampler stamps, so power and job logs share one clock.
- **Thermal guard** between runs as in the sweep (pause above 80 C, resume below 65 C), and log the board
  temperature at the start and end of each run.
- On any exception or Ctrl+C: stop the sampler, restore clocks and mode, write what has been logged so far with
  `status = "aborted"` in `run.json`.

## 8. Traffic and load (`common/traffic.py`)

- Mixes (shares w_i of total rate Lambda over n, s, m, l):
  `balanced = [0.25, 0.25, 0.25, 0.25]`, `skewed = [0.528, 0.149, 0.083, 0.240]` (the COCO routing split from the
  earlier dispatcher work). Configurable.
- **Capacity** for a mix, from the calibrated table with full batches `B_max`:
  `Lambda_cap = 1 / sum_i( w_i * service_s(i, B_max) / B_max )` requests per second. **Load is a hardcoded total
  arrival rate `Lambda` in req/s (`--lam`), listed per mix in `config.py`; it is not derived from rho (changed Oct 2026).**
  Print Lambda_cap, the percentage of it that `Lambda` is, and the per-queue rates before each run. Note: with
  `B_max` batches the real saturation can be lower (queues need time to fill), so the percentage is nominal, and
  the analysis also reports measured GPU busy fraction.
- Per queue: exponential inter-arrival times from `numpy.random.RandomState(seed * 100 + queue_index)`.
  `pool_index` drawn from the same RandomState.
- Arrival list is saved to the run folder (`arrivals.csv`) so any run can be audited.

## 9. Logs, metrics, validation

Per run folder:
- `requests.csv`: job_id, queue, pool_index, t_arrival_target, t_enqueued, t_batch_start, t_batch_end, batch_id,
  batch_size, arrival_lateness_ms, in_measurement_window, dropped (0/1).
- `batches.csv`: batch_id, queue, b, t_start, t_end, h2d_ms, forward_ms, nms_ms, decide_ms, queue_len_before.
- `waits.csv`: idle intervals (t_start, t_end, reason).
- `power_raw.log.gz` (the Sampler's raw lines) and `power.csv` (parsed rails per sample).
- `run.json`: policy and parameters, mix, Lambda, nominal load, seed, durations, profile hash, input source (data/inputs or
  ultralytics samples), power mode, clocks locked, board temp start/end, python/torch/ultralytics versions, git commit,
  `status` and any flags.

`04_analyze.py` (pandas, no torch) computes, over the measurement window only:
- **Latency** = `t_batch_end - t_arrival_target` (so that arrival lateness is charged to the system, not hidden). Mean, p50, p95, p99, max, overall and per queue.
  Also the split: queueing wait (enqueue to batch start) and batch service.
- **Total energy** (the headline energy metric): energy the board consumes from the **first arrival of the
  measurement window to the completion of the last job of that window**, idle time included.
  - `t_first` = the earliest `t_arrival_target` among jobs that arrive at or after `WARMUP_S`.
  - `t_last` = the latest `t_batch_end` among those same jobs (this includes the drain).
  - Total energy = trapezoid integral of board power over `[t_first, t_last]` (`energy_mj` from
    `common/power.py`), in joules. Report the span `t_last - t_first` and the mean power next to it.
  - Every policy gets the identical arrival list for a given mix, Lambda and seed, so `t_first` and the set of jobs
    are the same across policies; only `t_last` and the energy differ. That is what makes totals comparable. A
    policy that defers jobs pays for it here through a later `t_last` (more idle floor) or gains through lower
    busy energy (bigger batches); the table shows the net effect.
  - Report with all three board-power definitions the sweep uses (sum of rails, sum excluding SYS5V, GPU rail).
    The sum excluding SYS5V is the primary one in plots unless I say otherwise (see Flags).
  - Split the total into **busy energy** (power integrated over the intervals when a batch is on the GPU, from
    `batches.csv` `t_start` to `t_end`) and **idle energy** (the rest of the span, including waits), as joules
    and as a share of the total.
  - Also report the **dynamic** version: total minus `p_idle_loaded_mw * span`, for context only.
- **Energy per request** = total energy / number of jobs in the window (same denominator for every policy).
- Throughput, drop count, batch-size histogram per queue, GPU busy fraction, mean decide() time.
- **Summary table** `summary.csv` (one row per run) and `summary.md`.
- **Plots** (matplotlib, one idea each): mean and p99 latency vs Lambda per policy; total energy vs Lambda per policy
  (and energy per request vs Lambda); a stacked bar of busy vs idle energy per policy at Lambda 16; latency vs energy scatter with the Pareto front (Triton's delay grid appears as a curve); per-queue
  p99 at Lambda 22; batch-size histogram; one run timeline (queue lengths, batch bars, power trace on one time
  axis).
- **`validation.md`** (claim 3), computed per run and summarised across runs, with PASS/FAIL per check:
  1. every generated job is finished or dropped (count equality);
  2. batches do not overlap (sorted by t_start, each t_start >= previous t_end);
  3. in-run mean service time per (queue, b) is within 10% of the calibration p50 (report the worst offender);
  4. **Little's law**: time-average number of jobs in system (from the request log) equals throughput times
     mean latency within 5% over the measurement window;
  5. measured arrival rate per queue is within 5% of lambda_i;
  6. arrival lateness p99 below 5 ms (else flag);
  7. tegrastats cadence within 20% of the configured interval, and the power trace covers `[t_first, t_last]`
     with no gap longer than 3 sample intervals;
  8. energy bookkeeping: busy energy + idle energy equals total energy within 1%, and total energy computed from
     `power.csv` agrees with a recomputation from `power_raw.log.gz` within 1%;
  9. the job set and `t_first` are identical across policies for the same mix, Lambda and seed (compare the
     `arrivals.csv` hash recorded in `run.json`).

## 10. Plan runner (`03_plan.py`) and outputs

- A plan is a Python dict in `config.py` (no YAML): policies with parameter grids, mixes, Lambda levels per mix, seeds,
  durations. Each run is a subprocess of `02_run.py` (a crash or OOM kills one run, not the plan). A run
  directory containing a complete `run.json` with `status = "done"` is skipped (resumable). Cooldown between
  runs `COOLDOWN_S` (default 20).
- Before starting, print the number of runs, the estimated wall time (runs x (WARMUP+DURATION+drain+cooldown)),
  and with `BUDGET_MIN` set, drop the lowest-priority runs and say which.
- **Tier 1** (about 1 hour, the minimum for the meeting): mix `skewed`, Lambda `[8, 16, 24]` req/s, one seed,
  policies `fcfs_nobatch`, `fcfs_batch`, `lqf`, `triton` with delay `[0, 5, 20]`. 18 runs.
- **Tier 2** (add after Tier 1 looks sane): both mixes, Lambda per mix (skewed `[6, 11, 16, 22, 25]`, balanced `[4, 9, 13, 18, 20]`, proposed), 3 seeds, the full
  Triton delay grid, `fixed_batch_b` with b in `[2, 4, 8]`. Mark the priority order in the plan so the budget
  cut drops it last-first.
- Output tree:

```
sched-harness/results/<device_tag>/
  preflight.json  profile.json
  runs/<mix>_lam<..>_<policy>_<params>_s<seed>/   requests.csv batches.csv waits.csv power_raw.log.gz power.csv arrivals.csv run.json
  analysis/  summary.csv summary.md validation.md plots/
```
`device_tag` is read from the environment variable `DEVICE_TAG`, default `agx_xavier_jp511` (as in jetson-power).
Large raw power logs are gitignored; everything else is committed.

## 11. Preflight and smoke (`00_preflight.py`, `--smoke`)

`00_preflight.py` (under a minute) prints and saves `preflight.json`: board model, L4T, Python, torch with
`cuda.is_available()`, ultralytics, opencv; `nvpmodel -q` and the current clocks; root check and a 3 s parsed
tegrastats sample (rails found, board power); free RAM and disk; input pool built (source and count); all four
nets load and fit in memory together, one real forward at b=1 and b=8 with timing and NMS; finishes with a
clear `READY` or `NOT READY: <what>`.

`02_run.py --smoke` = 20 s warmup-free run, `lqf`, Lambda 11 req/s, balanced, then writes the logs and runs the validation
checks 1 to 8 on that single run (check 9 compares policies, so it is skipped) and prints PASS/FAIL. This is the single command I run first on the board.

## 12. What to take from the archive (read only, do not import)

`archive/edge-runtime/` is a larger earlier build (about 3,700 lines) of a related harness, never run on the
board. Read it for ideas on: `runtime.py` (thread and condition-variable structure), `policies.py` (Clockwork,
Symphony, EdgeServing details for the second policy set), `traces.py`. Leave behind: the nvidia-smi backend,
tensor-input mode, CityFlow trace replay, Promote/Drop decisions, the multi-subprocess power profile (already
done in `jetson-power`), and the `visit` mode. I want a smaller, plainer harness that I can read in one sitting.

## 13. Flags (stop and tell me instead of guessing)

- Power mode for the session: the sweep run so far is in `MAXN`, and a note in `jetson-power/config.py` says a
  MAXN run took the board down once. Do not pick the mode yourself: read the mode from `config.py`
  (`SESSION_MODE`, leave it `None` meaning "do not change the mode, use the current one") and print it at
  startup. I will set it.
- If all four nets do not fit on the GPU together, or `B_max=16` does not fit for l.
- If NMS time is large enough that it dominates yolov8n service time (print its share in calibration).
- If the arrival thread cannot keep p99 lateness under 5 ms at the highest planned total rate. Say what rate
  that is; do not silently lower the loads.
- If `ultralytics` NMS import fails on 8.3.186.
- Which board-power definition is primary for total energy. The sweep reports three and whether SYS5V overlaps
  the other rails is still unchecked on the board. Until I answer, use the sum excluding SYS5V as primary and
  print the other two beside it.
- If `t_last` differs between policies by more than a few percent only because of the drain tail, say so in the
  analysis summary; do not trim it.
- Anything in this spec that contradicts the measured data (for example the fit `a + c*b` is poor).

## 14. Done when

- All files in section 3 exist and compile under Python 3.8 (`python3.8 -m py_compile` equivalent by reading).
- `tests/test_policies.py` passes with plain `python tests/test_policies.py` (no torch needed).
- README gives, in order: the exact commands for preflight, calibration, the smoke run, the Tier 1 plan and the
  analysis; expected wall times; how to copy the folder and `results/` back; what each validation check means.
- Your summary lists: assumptions you had to make, deviations from this spec, Flags, suggested lines for
  CLAUDE.md, and the exact first three commands I type on the board.
