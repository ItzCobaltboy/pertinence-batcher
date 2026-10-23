# Spec: `edge-runtime/`, a real-hardware multi-model scheduling harness

Hand-off spec for Claude Code. Read all of it before writing code. Where this spec says "verify
against the paper", do that before implementing; where it says "flag", stop and tell me rather
than guessing.

---

## 0. Goal and non-goals

**Goal.** Run real YOLOv8 models on real hardware (Jetson first, laptop RTX 5070 Ti and the A100
for development) under realistic multi-camera traffic, with a pluggable scheduler deciding what
runs next, and measure latency and energy for each scheduling policy. The point is to catch the
behaviour a simulator hides: CPU preprocessing cost, Python and launch overhead, timing jitter,
thermal and clock effects, CPU/GPU contention on unified memory, model switching cost.

**Paper direction this serves.** A generic multi-model scheduler for one edge accelerator, with
better latency plus power saving as the goal. The first deliverable is reimplementing existing
policies and seeing how they actually behave on the device; my own policy comes later and plugs
into the same interface.

**Non-goals.**
- No discrete event simulator. `scheduler-sim/` is not used, imported or extended.
- No TensorRT. Eager PyTorch only (the torch_tensorrt YOLOv8 export gives wrong outputs, see
  `CLAUDE.md` "TRT engines broken").
- No MPS or concurrent kernels. One batch on the GPU at a time.
- No retraining of anything.

---

## 1. Repo context (read these first)

Repo: `pertinence-batcher`. Pull latest main. Read `CLAUDE.md` fully, then:
- `yolo-analysis-a100/power_analysis/`: the A100 power sweep. Reuse its phase design (baseline,
  idle with model loaded, run phases, cut one continuous trace with `time.time()` marks). The
  Jetson power profile in this spec is a port of it.
- `yolo-analysis-a100/batch_timing_analysis/`: how T_i(b) was measured (warmup, sync, real COCO
  batches). Reuse the measurement discipline, not the code.
- `yolo-dispatcher/config.py`, `yolo_backbone.py`, `letterbox.py`: model pool, weight lookup,
  Letterbox(640) preprocessing. Import, don't copy.
- `cityflow-routing/`: per-frame routing CSVs (`results/<run>/raw/routing_<S>_<cam>.csv.gz`)
  with `synced_time_s` and `model_p<id>` columns. These become real arrival traces (section 5).
- `cityflow-dispatcher/`: fine-tuned heads (not run yet). Traces can be regenerated from its
  routing later; nothing here depends on it.

Repo conventions (hard rules):
- New top-level folder `edge-runtime/` with its own `README.md` and `requirements.txt`.
- Do not touch `Journel/`. Edit `CLAUDE.md` in place on top of the current version (never write
  it back from an older copy) and keep its CRLF line endings.
- No em dashes anywhere you write (code comments, README, CLAUDE.md).
- Plain functions and small classes; see section 1b for the full style rules. Previous
  over-engineered versions were thrown away for being hard to trust.
- Work on a branch, open a PR at the end.

---

## 1b. Code style (applies to every file)

Depth is not the issue: the code can be as long or as short as the problem needs. What matters is
that I can read any function top to bottom without stopping to decode Python. Write it so a
C++/Java programmer follows it on first read.

**Do:**
- Explicit `for` loops and `if` / `else` with early returns. One idea per line.
- Plain classes with an explicit `__init__` for things with state (Job, Queue, each policy);
  plain functions for everything else.
- Descriptive names (`oldest_job`, `batch_start_time`), no one-letter names outside short loops.
- Each function starts with a one or two line docstring saying what it does; comments explain
  *why* something non-obvious is done, not what the line says.
- Plain lists and dicts in the runtime. pandas only in `analyze.py` and `traces.py`.
- `print` plus writing to a log file for logging. `argparse` with a handful of flags per script.
- Constants in `config.py`, imported by name.

**Don't (no matter how idiomatic):**
- Nested or multi-clause comprehensions, long chained one-liners, `lambda` chains, the walrus
  operator, `map`/`filter`/`reduce` tricks, `functools` / `itertools` cleverness.
- Decorators other than `@staticmethod` if really needed. No metaclasses, no descriptors, no
  dynamic `getattr`/`setattr`, no monkeypatching, no operator overloading.
- `*args` / `**kwargs` pass-through, generator pipelines where a list is fine, custom context
  managers (use `try` / `finally`), `asyncio`.
- Inheritance hierarchies or ABCs. Policies are separate small classes that each have a
  `decide(state)` method; nothing inherits from anything.
- Type-hint gymnastics (`TypeVar`, `Protocol`, generics). Simple hints like `b: int` are fine.

**Not verbose either:**
- No getters / setters, wrapper functions that only call one other function, or config objects
  wrapping config objects.
- No defensive checks for things that cannot happen; validate at the boundaries (policy
  decisions, input files) and trust the rest.
- No logging framework, plugin system, CLI framework or factory.

When in doubt between short-and-clever and plain-and-a-bit-longer, pick plain.

## 2. Platforms and environment constraints

| Platform | Role | GPU power source | Constraints |
|---|---|---|---|
| **Jetson AGX Xavier** (confirmed; Volta GPU, 512 CUDA cores, 16 or 32 GB shared memory) | The real measurements | `tegrastats` | JetPack 4.x or 5.x (exact version from preflight); JetPack 5.1 is the last for Xavier, so CUDA 11.4 and **Python 3.8** at best. If preflight finds JetPack 4.x (Ubuntu 18.04, Python 3.6, torch 1.x), ultralytics and this code will not run as specced: preflight must say so loudly, and I decide whether the board gets reflashed to JetPack 5. torch comes from NVIDIA's JetPack wheel: **never pip install or upgrade torch/torchvision there**. `nvpmodel` / `jetson_clocks` need sudo, which I may not have. Visits may be short (an hour or two). |
| Laptop RTX 5070 Ti (WSL2, Blackwell sm_120) | Development and debugging | `nvidia-smi` | Numbers are not data. Power-limited laptop. |
| A100 server | Optional extra runs | `nvidia-smi` | Frozen env: torch 2.3.1+cu121, numpy 2.2.6, pandas 2.3.3, ultralytics 8.4.x. Install nothing. |

Code must run on **Python 3.8** (the AGX Xavier's ceiling). That rules out 3.9+ syntax: no `list[int]` / `dict[str, float]` hints (use plain names or `typing.List`), no `dict1 | dict2`, no `match`, no `str.removeprefix`, no parenthesised multi-item `with`. Check every file against this. Dependencies: numpy, pandas,
matplotlib, torch (preinstalled), opencv (for real frames), ultralytics (for loading YOLOv8
weights). Plain JSON for configs, no YAML. If ultralytics is missing on the Jetson, the preflight
reports it and prints the install command; it does not install anything itself.

---

## 3. Architecture

```
edge-runtime/
  README.md, requirements.txt
  config.py          every default (paths, pool, batch grid, SLOs, loads, durations, budgets)
  preflight.py       environment dump, in seconds (section 12)
  power.py           power backends: nvidia-smi, tegrastats (section 4)
  models.py          load the 4 YOLOv8 models, warmup, forward(batch) with sync
  inputs.py          tensor mode and real-frame mode (section 6)
  calibrate.py       T_i(b) profile on this device -> profile.json (section 7)
  power_profile.py   Jetson port of power_analysis (section 8)
  traces.py          build arrival traces: synthetic + CityFlow replay (section 5)
  runtime.py         arrival thread, queues, executor loop, logging (section 9)
  policies.py        every policy, plain functions/classes (section 10)
  plan.py            experiment runner with time budget and resume (section 13)
  analyze.py         metrics + plots from run logs, no torch (section 14)
  tests/
```

One run = one trace replayed in real time against one policy, producing a request log, a batch
log and a power trace.

---

## 4. Power backends (`power.py`)

Two backends, one interface: `start(path)`, `stop()`, and `read(path) -> DataFrame[t, power_w, rails..., extra...]`.
All timestamps are host `time.time()`; everything else in the harness records the same clock.

**nvidia-smi** (laptop, A100): same as `power_analysis`: one
`nvidia-smi -i <idx> --query-gpu=timestamp,power.draw[.instant],clocks.sm,temperature.gpu,
utilization.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits -lms 50 -f <file>`
for the whole session. Probe whether `power.draw.instant` exists and record which field was used.
Known gotcha: it writes in ~4 KB chunks, so keep sampling >= 10 s past the last mark.

**tegrastats** (Jetson): run `tegrastats --interval 50` with stdout piped, read lines in a thread
and stamp each line with `time.time()` on read (do not trust tegrastats' own timestamp format,
it differs by version). Check the effective cadence after the run and record it.
- Parse every power rail present generically: tokens like `NAME current_mW/avg_mW` (names differ
  by board: `VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC`, `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0`, ...).
  Store every rail as its own column.
- **Board power** = `VDD_IN` if present, else the sum of all rails, and record which. The AGX
  Xavier has no `VDD_IN`: it reports rails like `GPU`, `CPU`, `SOC`, `CV`, `VDDRQ`, `SYS5V`
  (names differ between JetPack 4 and 5, e.g. `VDD_GPU`, `VDD_CPU`, `VDD_SOC`, `VDD_CV`,
  `VDD_DDR`, `VDD_SYS5V`). Report each rail, the GPU rail on its own, and a sum. Whether `SYS5V`
  already includes the others is board-specific: flag it and report both "sum of rails" and
  "sum excluding SYS5V" until checked.
- Also parse `GR3D_FREQ` (GPU utilisation / clock), CPU loads, `EMC_FREQ`, temperatures.
- Unknown tokens are kept raw in a text column, never a crash.
- If tegrastats needs sudo for some fields on this image, run it without sudo and record what was
  missing.

Sampler cleanup: always stopped in `try/finally` and `atexit`. A killed run must not leave a
sampler process behind. No more than that (no multi-layer cleanup machinery).

---

## 5. Workload: arrival traces (`traces.py`)

A trace is a JSON-lines or CSV file of jobs, sorted by arrival:
`job_id, camera, t_arrival_s (relative to trace start), queue (model name), slo_ms, frame_ref`.

Deadline = arrival + SLO. SLO is a per-run parameter (same for all jobs in a run) unless the trace
sets it per camera.

**Sources:**
1. **Periodic multi-camera** (the main synthetic case): C cameras at F FPS each (default 8 x 10
   FPS), each camera with a random phase offset (not all at t=0). Each frame routed to a model by
   a fixed split (default the COCO val2017 routing split `[0.528, 0.149, 0.083, 0.240]`) or by a
   sticky Markov chain (stay probability p, default 0.9).
2. **Poisson** per queue at a given total rate, for comparison with the literature.
3. **CityFlow replay**: `traces.py from-cityflow --run <cityflow-routing results dir> --scenario
   S02 --point <id>`: one camera per CityFlow camera, arrival = `synced_time_s`, queue =
   `model_p<id>`. Writes a compact trace so the Jetson never needs the CityFlow results or videos.
   Optional `--cameras` and `--duration` to cut it down.

**Load levels**: express load as a fraction of the device's capacity, measured by calibration
(section 7): capacity = frames/s the device sustains with max batching on the trace's split.
Synthetic traces scale cameras or FPS to hit 20 / 50 / 80% (configurable). CityFlow traces are
what they are; report their load fraction.

Traces are generated offline (laptop) and copied to the device. A trace has a fixed seed and is
identical for every policy.

---

## 6. Inputs (`inputs.py`)

Two modes behind a flag, both required:
- **tensor**: a pool of pre-made 3x640x640 float tensors already on the GPU, cycled. Pure GPU
  behaviour, no preprocessing.
- **frames**: a pool of N real JPEG frames (default 256, exported from CityFlow on the laptop by
  `inputs.py export-frames`, stored as JPEGs, ~50 MB). On each job: decode JPEG, letterbox with
  yolo-dispatcher's `IMAGE_TRANSFORM`, move to GPU. Where this happens is a flag:
  - `--prep arrival` (default): done in the arrival path as the frame "arrives", like a camera
    pipeline; its time is logged per job but is not part of the GPU batch.
  - `--prep executor`: done by the executor when the batch is formed, so it adds to batch time.

Log decode and preprocessing time per job in both cases.

---

## 7. Calibration: T_i(b) on this device (`calibrate.py`)

Policies need a profile. Measured on the device, never copied from the A100.
- Models n/s/m/l, batch sizes default `[1, 2, 4, 8, 16]` (configurable, add 32 if memory allows).
- Per (model, b): warmup 20 forwards, then 100 timed forwards with `torch.cuda.synchronize()`
  before and after each, real input pool. Record p50, p95, p99, mean, std.
- Also record: model load time, first-forward time, GPU memory per model, and whether all four
  models fit in memory at once (they should: ~340 MB FP32 total). If they don't, flag it.
- Also measure model switch cost: alternate models (n, l, n, l, ...) at b=1 and compare to
  same-model back-to-back. Hidden cost #1 to look for.
- Output `profile.json`: `{model: {b: {p50_ms, p95_ms, ...}}}` plus device metadata.
- Runtime uses p50 by default (`--profile-stat p95` option).

---

## 8. Power profile on the Jetson (`power_profile.py`)

A port of `yolo-analysis-a100/power_analysis/` with the same phase structure:
- 30 s baseline, nothing loaded (process started, no CUDA context).
- Per (model, b), random order, each in its own subprocess: load model + real batch, one warmup,
  20 s idle with the model loaded, then 5 run phases of back-to-back forwards (fixed count or
  fixed 10 s, whichever is longer), 3 s gaps, sync at the end of each phase.
- One continuous power trace for the whole sweep; cut it afterwards with the recorded marks.
- Output per (model, b): mean board power, mean of each rail, energy per batch, energy per frame
  (total and dynamic = (P - P_idle_loaded) x T / b), GPU clock and temperature during the phase.
- Record power mode: `nvpmodel -q` and, if permitted, `jetson_clocks --show`. Run under whatever
  mode the board is in. `--set-mode N` and `--lock-clocks` options that only act if sudo works,
  and record the result either way.
- `--modes` option to repeat the sweep under several nvpmodel modes when sudo works (this is the
  knob the A100 could not show).
- Time-budgeted: `--budget-min 45` picks a batch grid and phase length that fit, and prints the
  estimate before starting.

---

## 9. Runtime (`runtime.py`)

**Threads.** Main thread loads everything, then starts:
- **Arrival thread**: sleeps until each job's arrival time (absolute `time.monotonic()` target,
  not cumulative sleeps), does preprocessing if `--prep arrival`, appends the job to its model's
  queue with `t_enqueued`, signals the executor. Log how late each arrival actually was vs its
  target (hidden cost: Python sleep jitter).
- **Executor loop** (one thread, owns the GPU): whenever the GPU is idle, call the policy; carry
  out its decision; repeat. Never runs two batches at once.

Use `threading` with a lock and a condition variable. Queues are per model, FIFO, holding Job
objects (`job_id, camera, queue, t_arrival, deadline, frame_ref, t_enqueued`).

**Decisions** (the only things a policy can return):
- `Serve(queue, b)`: take the b oldest jobs of that queue (b <= len), run one batch, record
  `t_start` / `t_end` (sync after the forward), mark jobs done. Postprocessing (NMS) is on by
  default and timed separately; `--no-nms` to drop it.
- `Wait(until)`: sleep until `until` or until a new arrival, whichever is first, then ask again.
  The time spent waiting is logged as an idle interval (it costs idle power).
- `Promote(src, dst, k)`: move the k oldest jobs of queue src into queue dst (dst must be a
  larger model), then the policy is asked again. Implement and test it in the executor; no
  initial policy uses it. Log promotions.
- `Drop(queue, job_ids)`: remove jobs (for policies that drop late requests). Logged as drops.

The executor validates every decision (queue exists, b <= len, dst larger than src, etc.) and
raises on an invalid one, the same contract as the old simulator.

**Policy call cost.** Time every `decide()` call and log it: scheduler overhead on a weak ARM CPU
is one of the hidden costs.

**End of run.** Stop arrivals at trace end, then drain the queues with the same policy, then stop.
Report the drain separately. Discard the first `--warmup-s` (default 10 s) from metrics.

**Logs** (per run directory):
- `requests.csv`: one row per job: arrival target, actual arrival, enqueue, batch start, batch
  end, deadline, model run (after any promotion), batch id, preprocessing time, dropped flag.
- `batches.csv`: batch id, model, b, t_start, t_end, forward ms, NMS ms, decide() ms before it.
- `waits.csv`: idle intervals.
- `power.*`: the backend's trace for the run.
- `run.json`: policy and its parameters, trace file and hash, SLO, load, input mode, prep mode,
  device metadata, power mode, versions, git commit.

---

## 10. Policies (`policies.py`)

Signature: `decide(state) -> Serve | Wait | Promote | Drop`, where `state` gives read-only
access to: `now`, per-queue lists of jobs (arrival, deadline), the profile (T_i(b) table and the
allowed batch sizes), `B_max`, and the policy's own parameters. A policy may keep internal state
between calls (for example Triton's timers). Batch size b is limited to the profiled sizes plus
"whatever is in the queue" up to B_max (profile lookup interpolates linearly between profiled
sizes).

**Baselines:**
1. `fcfs_nobatch`: serve the queue holding the oldest job, b = 1.
2. `fcfs_batch`: serve the queue holding the oldest job, b = min(|Q|, B_max).
3. `lqf`: serve the longest queue (ties: oldest head), b = min(|Q|, B_max).

**Existing approaches** (reimplement their scheduling policy only, in this setting):

4. `triton`: NVIDIA Triton dynamic batcher, per model. Parameters: `max_queue_delay_ms` (per
   queue or global), `max_batch` (= B_max), optional `preferred_batch_sizes`. A queue is ready
   when |Q| >= max_batch (or reaches a preferred size) or its oldest job has waited
   >= max_queue_delay. Among ready queues serve the one with the oldest head, b = min(|Q|,
   max_batch). If none is ready, `Wait` until the earliest timeout. Triton has no cross-model
   coordination; that is the point of the baseline. The plan runner grid-searches
   `max_queue_delay_ms` (default `[0, 2, 5, 10, 20, 50]`).

5. `edgeserving`: Cao et al., arXiv 2605.05527. **Verify every detail against the paper's
   scheduling section before implementing** (fetch the arXiv PDF). Known shape: per-model queues,
   one batch at a time, never waits (work-conserving), batch = min(|Q_m|, B_max) with B_max = 10
   in their setup, and the queue is chosen to minimise a system-wide stability score over all
   waiting jobs with urgency f(w) = min(exp(w / tau - 1), C), tau tied to the job's SLO. Compute
   the score as it would be after the candidate batch finishes (waits of the jobs left behind,
   evaluated at now + T_m(b)). No early exits (we have none); say so in the README. Their
   accuracy/exit choice is out of scope.

6. `clockwork`: Gujarati et al., OSDI 2020, the controller's INFER scheduling only (no LOAD,
   no memory management, one executor). Verify against the paper's scheduler section. Known
   shape: for each model and each allowed batch size b <= |Q|, a strategy (model, b) is valid if
   the b oldest jobs can all finish by their deadlines when started now (now + T(b) <= earliest
   of their deadlines). Each strategy has a latest start = earliest deadline - T(b). Choose the
   valid strategy with the earliest latest start (most urgent), then grow b for that model to the
   largest still-valid size. Jobs that can no longer meet their deadline even at b = 1 are
   dropped (Clockwork's behaviour; `--no-drop` option serves them late instead). If nothing is
   valid and nothing is droppable, serve the oldest job at b = 1. Clockwork's ~5 ms commit-ahead
   exists to hide network/RPC latency; with a local executor it is off by default
   (`commit_ahead_ms` option).

7. `symphony`: Chen et al., arXiv 2308.07470, deferred batch scheduling. Read Sec 3.1 to 3.4
   and **verify before implementing**. Known shape: each model's batch is deliberately held back
   to grow, but launched no later than the point where waiting for one more request would make
   the head request miss its deadline: launch when now >= d_head - l_m(|Q_m| + 1) (the
   "frontrun" point of the window [d - l(b+1), d - l(b)]) or when |Q_m| reaches B_max. A central
   scheduler picks among models that are due (earliest deadline first). Otherwise `Wait` until
   the earliest due time. Requests that would miss their SLO are dropped, per the paper
   (`--no-drop` option).

8. (stretch, after the rest works) `cost_v1`: my policy v1 from `CLAUDE.md` ("Policy v1
   defined"): Cost(i, b) = beta * sum over jobs left waiting of u_lambda(age at completion) +
   (1 - beta) * T_i(b), u_lambda(J) = (exp(lambda J) - 1) / lambda (lambda -> 0: plain sum),
   argmin over (queue, b), no Wait option yet. Parameters beta, lambda.

Not now: Xu et al. SMDP thresholds (needs offline value iteration), EdgeBatch (learned).

For each reimplemented policy, the README gets a short table: what is faithful, what was
simplified, and why.

---

## 11. Metrics (`analyze.py`, no torch, runs on the laptop)

Per run, after dropping the warmup window:
- Latency (finish - arrival): mean, p50, p95, p99, max; per model queue and overall. Separately:
  queueing wait, preprocessing, batch run time.
- SLO: miss rate, drop rate, "goodput" (frames finished within SLO per second).
- Batching: batch-size histogram per model, mean b, batches per second, promotions.
- Energy (from the power trace, integrated over the measured window):
  - total energy per finished frame;
  - dynamic energy per frame = (P - P_idle_loaded) integrated / frames;
  - energy split into: busy (inside batches) vs idle (waits and gaps);
  - mean board power, GPU busy fraction (sum of batch time / window).
- Hidden costs: arrival lateness distribution, decide() time distribution, preprocessing time,
  model switch count and the measured switch penalty, clock/temperature drift over the run.

Plots:
- latency CDF per policy (same load, same SLO);
- p95 latency vs load, one line per policy;
- energy per frame vs load, one line per policy;
- latency vs energy scatter / Pareto front across all runs (Triton's timeout grid becomes a
  curve, the others points);
- batch-size histograms per policy;
- one run timeline: queue lengths per model, batches as bars, power trace below, same time axis.

`summary.csv` (one row per run) + `summary.md`.

---

## 12. Preflight and visit mode (`preflight.py`)

Jetson time may be one short visit, so everything must work on the first try.
`python preflight.py` (under a minute) reports and saves `preflight.json`:
- board model (`/proc/device-tree/model`), JetPack / L4T (`/etc/nv_tegra_release`), CUDA,
  Python, torch and its CUDA availability, ultralytics, opencv, numpy, pandas, matplotlib;
- `nvpmodel -q`, sudo availability (non-interactive check only, never prompts), tegrastats
  present and a 3 s parsed sample (rails found, board power value);
- free disk and RAM, GPU memory;
- weights found for n/s/m/l (yolo-dispatcher's lookup), loads each and runs one forward at b=1
  and b=8 with timing;
- traces and frame pool present.
Ends with a clear "ready / not ready" line and what is missing.

`python plan.py visit --budget-min 90` runs, in order and within the budget: preflight,
calibration, a reduced power profile, then as many policy runs as fit (priority order:
`lqf`, `triton` at 2 delays, `edgeserving`, `clockwork`, `symphony`, `fcfs_nobatch`), and
finally `copyout`: one `tar.gz` of all results, its path and size printed at the end.

---

## 13. Experiment runner (`plan.py`)

A plan is JSON: list of policies (with parameter grids), traces, loads, SLOs (default
`[50, 100, 200]` ms, flag if these look wrong once the device profile is known), input mode,
prep mode, run duration (default 120 s of trace + drain), repeats (default 1), cooldown between
runs (default 15 s), and an optional global time budget.
- Each run executes in its own subprocess (a crash or OOM kills one run, not the plan). Models
  load per subprocess; that load time is logged, not counted.
- Resumable: a run whose directory has a complete `run.json` + logs is skipped.
- Before starting, print the run count and estimated wall time; with a budget, drop
  lowest-priority runs and say which.
- Record board temperature at each run's start; warn if it is far above the first run's.

---

## 14. Outputs

```
edge-runtime/results/<device_tag>/
  preflight.json
  profile.json                 calibration
  power_profile/               per (model, b) + raw trace + plots
  runs/<run_id>/               requests.csv, batches.csv, waits.csv, power.csv, run.json
  analysis/                    summary.csv, summary.md, plots/
```
`device_tag` from preflight (for example `agx_xavier_MAXN`, `agx_xavier_30W`, `rtx5070ti`, `a100`). Results are
committed except raw power traces over a size limit (add `.gitignore` lines).

---

## 15. Tests (`tests/`, on the laptop GPU with real models)

No CPU-only testing, no fake models, no fake power. Everything is validated on real hardware: the
laptop RTX 5070 Ti with the real YOLOv8 weights and real `nvidia-smi` sampling. You (Claude Code)
cannot run these; write them so I can run them with one command and read a clear pass/fail.

- **Smoke run per policy**: a 30 s periodic trace at low load on the laptop GPU, real models,
  both input modes. Assert complete logs, every job accounted for (finished or dropped), no
  overlapping batches, batch times consistent with the calibration profile.
- **Calibration sanity**: T_i(b) increases with b and with model size; p95 / p50 reported; model
  switch penalty measured.
- **Power**: the nvidia-smi trace covers the whole run, its cadence is recorded, and idle vs
  running power differ visibly; per-batch energy sums to the window energy within 1%.
- **Policy decisions**: each policy checked on hand-built queue states with known correct
  decisions (Triton timeout fires, Clockwork picks the most urgent valid strategy and grows it,
  Symphony waits until its frontrun point then launches, EdgeServing picks the score-minimising
  queue, drops happen when they should). These are plain asserts on the decision logic, run in
  the same test command.
- **Executor**: rejects invalid decisions; Promote moves the right jobs; Wait wakes on arrival.
- **tegrastats parser**: line fixtures for AGX Xavier on JetPack 4 and JetPack 5 (the target), plus
  one Orin-style line (from NVIDIA's
  documented format), unknown tokens, missing rails. The real check happens in preflight on the
  Jetson, which prints the parsed rails from a live sample.
- **Sampler cleanup**: kill a run mid-way, assert no sampler process left.
- **Trace builder**: periodic phases, split proportions within tolerance, CityFlow replay from a
  small fixture CSV preserves synced times and queues.

## 16. Done when

- Tests written for the laptop GPU (section 15), runnable with one command; you can't run them,
  so tell me the exact command and what a pass looks like.
- On the laptop GPU: preflight, calibration, a 2-minute run per policy and the analysis all work
  end to end with real models, in both input modes (make it ready and tell me the exact commands).
- README: what it measures, the visit procedure for the Jetson step by step (copy repo + traces +
  frame pool, preflight, visit plan, copyout), expected runtimes, the faithfulness table per
  policy, known limits.
- `CLAUDE.md`: a short `edge-runtime/` bullet under "Repo layout" and one line in the scheduler
  open thread saying the DES is no longer the evaluation path.
- PR opened. At the end give me: the exact laptop commands, the exact Jetson visit commands, and
  anything you had to assume or simplify (especially in sections 10.5 to 10.7).

---

## 17. Things to flag instead of guessing

- Any EdgeServing / Clockwork / Symphony detail that the paper states differently from this spec.
- Anything that needs root on the Jetson that you can't make optional.
- If all four models don't fit in memory together on the AGX Xavier at FP32 (they should).
- If the installed ultralytics / torch on the Xavier's Python 3.8 can't load YOLOv8 weights: report
  the versions and the exact error, and propose the smallest fix (I install nothing without asking).
- If the default SLOs or loads make no sense once the profile exists (for example every policy
  misses everything at 50 ms).
