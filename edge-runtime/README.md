# edge-runtime

Real models on real hardware under multi-camera traffic, with a pluggable scheduler
choosing what runs next, measuring latency and energy per scheduling policy. It exists
to catch what a simulator hides: CPU preprocessing, Python and launch overhead, timing
jitter, thermal and clock effects, CPU/GPU contention on unified memory, model switch
cost. `scheduler-sim/` is not used.

YOLOv8 n/s/m/l, eager PyTorch FP32 (no TensorRT: see `CLAUDE.md`, "TRT engines
broken"), one batch on the GPU at a time (no MPS, no concurrent kernels), nothing
retrained. Targets: a Jetson for the real measurements; the laptop RTX 5070 Ti and the
A100 for development (their numbers are not data); any CPU machine for the tests.

## Files

| file | what |
|---|---|
| `config.py` | every default: pool, batch grid, SLOs, loads, durations, budgets, paths |
| `common.py` | profile table `Profile` (T_i(b), interpolated), `Job`, JSON/env helpers, device tag |
| `power.py` | power backends `nvidia-smi`, `tegrastats`, `fake`, `none`; tegrastats parser; energy integration |
| `models.py` | loads the 4 models with yolo-dispatcher's lookup; `Runner.run_batch` (sync, forward, sync, NMS); `FakeRunner` |
| `inputs.py` | tensor mode, frames mode, `export-frames` |
| `traces.py` | arrival traces: periodic multi-camera, sticky routing, Poisson, CityFlow replay |
| `calibrate.py` | T_i(b) on this device -> `profile.json` |
| `power_profile.py` | E_i(b): port of `yolo-analysis-a100/power_analysis/` with tegrastats and a time budget |
| `runtime.py` | one run: arrival thread + executor loop + logs |
| `policies.py` | every policy (baselines + Triton, EdgeServing, Clockwork, Symphony, cost_v1) |
| `plan.py` | experiment runner (subprocess per run, resumable, budget), `visit`, `copyout` |
| `analyze.py` | metrics + plots from run logs (no torch) |
| `preflight.py` | environment check before a visit |
| `tests/` | 43 CPU tests, no GPU, no weights |

Imports from the rest of the repo: yolo-dispatcher's `config.py` (weight lookup,
`IMAGE_TRANSFORM` = Letterbox(640)) and `yolo_backbone.load_detection_model`, and
cityflow-routing's `dataset.py` for `export-frames`. yolo-dispatcher's config is loaded
under another module name because this folder has its own `config.py`.

## What one run measures

`runtime.py` replays one trace in real time against one policy:

- **Arrival thread**: sleeps until each job's absolute arrival time (monotonic target,
  not cumulative sleeps), preprocesses it if `--prep arrival`, appends it to its
  model's FIFO queue, wakes the executor. How late each arrival actually was is logged.
- **Executor** (main thread, owns the GPU): whenever the GPU is idle and a queue is
  non-empty, asks the policy, validates the decision (raises on an invalid one) and
  carries it out:
  - `Serve(queue, b[, job_ids])`: run one batch of the b oldest jobs (or the listed
    ones), sync after forward, NMS timed separately (`--no-nms` drops it);
  - `Wait(until)`: idle until `until` or the next arrival (logged as an idle interval);
  - `Promote(src, dst, k)`: move the k oldest jobs of src into a larger model's queue
    (merged by arrival). Implemented and tested; no policy uses it yet;
  - `Drop(queue, job_ids)`.
- Every `decide()` call is timed. At trace end arrivals stop, the queues drain with the
  same policy (cut after `--drain-timeout`), then the run stops. Before the trace,
  `--idle-s` seconds of idle with the models loaded give the run's own idle power.
- Latency = finish - arrival, where arrival is when the camera produced the frame
  (trace time): arrival lateness and preprocessing are inside it. Deadline = arrival + SLO.
- Collapse guard: past `--max-queued` jobs queued (default 300; in frames mode each
  holds a ~4.9 MB GPU tensor) the run stops with status `collapsed`.

Per run folder: `requests.csv` (one row per job: target and actual arrival, enqueue,
batch start/end, deadline, model run, batch id, decode/prep ms, dropped),
`batches.csv` (model, b, start, end, prep, assemble, forward, NMS, decide() ms, planned
ms), `waits.csv`, `decisions.csv`, `promotions.csv`, `power.csv.raw.gz` (+ `.meta.json`),
`stdout.log`, and `run.json` (policy and parameters, trace + hash, SLO, load, input and
prep modes, device, power mode, versions, git commit; written last, so its presence
marks a complete run).

All timestamps are host `time.time()` (taken as `t0_wall + perf_counter - t0_perf`, so
they never jump), the same clock the power samplers stamp with.

## Workload

`traces.py` writes CSV + `.meta.json`: `job_id, camera, t_arrival_s, queue, slo_ms,
frame_ref`. Same seed, same trace, for every policy.

- **periodic** (main synthetic case): C cameras at F FPS, each with a random phase.
  Routing by a fixed split (default COCO val2017 `[0.528, 0.149, 0.083, 0.240]`) or a
  sticky Markov chain (`--sticky 0.9`: stay with probability p, else redraw from the
  split, so the long-run split is unchanged).
- **poisson** per queue at a total rate.
- **from-cityflow**: one camera per CityFlow camera, arrival = `synced_time_s`, queue =
  `model_p<id>`, read from the committed `cityflow-routing/results/all/raw` CSVs.
  `data/traces/cityflow_S01_p80.csv` and `cityflow_S02_p80.csv` (130 s) are committed.
- **Load** = offered frames/s / capacity, capacity = frames/s the device sustains with
  every queue at b = B_max and the trace's split (from `profile.json`). Synthetic traces
  scale FPS to hit 20 / 50 / 80 %; CityFlow traces report theirs.
- `python traces.py suite --profile <profile.json>` writes periodic, sticky and Poisson
  traces at every load (what `plan.py visit` does on the device after calibration).

## Inputs

- `--input tensor`: a pool of 3x640x640 tensors already on the GPU (built from the
  frame pool, or noise if there is none), cycled. Pure GPU behaviour.
- `--input frames`: a pool of real JPEGs held in RAM as bytes. Per job: decode (PIL),
  letterbox (yolo-dispatcher's `IMAGE_TRANSFORM`), copy to the GPU.
  `--prep arrival` (default) does it in the arrival path, logged per job but outside the
  batch; `--prep executor` does it when the batch is formed, so it adds to batch time.

The frame pool lives in `data/frames/` (git-ignored, ~50 MB). Make it on the laptop:

```
python inputs.py export-frames --scenario S02        # 256 frames from the extracted CityFlow videos
python inputs.py export-frames --from-coco            # or: 256 COCO val2017 images (downloads)
```

CityFlow frames are 1080p, COCO ones ~640x480: preprocessing cost depends on which.

## Calibration and power profile

`calibrate.py`: per (model, b) warmup, then timed forwards with a sync before and after
each, on the real input pool; p50 / p95 / p99 / mean / std of forward, NMS and their
sum. A combo stops at `--iters` (100) or `--max-s` (20 s), never before `--min-iters`
(10): a Jetson runs yolov8l at b = 16 in seconds. Also: load time, first-forward time,
GPU memory per model and per batch, whether all four fit together, and the switch cost
(n, l, n, l, ... at b = 1 against back-to-back). It flags default SLOs that are below
1.5x the slowest model at b = 1 and suggests some (`suggested_slos_ms`). Policies plan
with the `total` table (forward + NMS) at `--profile-stat` (p50 default).

`power_profile.py`: same phases as `power_analysis` (30 s baseline with nothing loaded;
per (model, b) in random order, its own subprocess: load, warmup, 20 s idle with the
model loaded, 5 run phases of at least 50 forwards and 10 s, 3 s gaps; one continuous
trace cut with `time.time()` marks). Per (model, b): board power and every rail, energy
per batch, energy per frame (total, and dynamic = (P - P_idle_loaded) x T / b), GPU
clock, temperature. Records `nvpmodel -q` and `jetson_clocks --show`; `--set-mode N`,
`--lock-clocks` and `--modes 0,2` act only if `sudo -n` works, and record the outcome.
`--budget-min` shrinks the grid and phases to fit and prints the estimate first.

Power backends: `nvidia-smi` probes `power.draw.instant` (else `power.draw`, a ~1 s
average on some GPUs), SM clock, temperature, utilisation, throttle reasons, samples
every 50 ms into one file (kept running >= 10 s past the last mark: it writes in ~4 KB
chunks). `tegrastats --interval 50` runs on a pseudo-terminal (so it line-buffers) and
each line is stamped with `time.time()` when read; every rail becomes a column; board
power = `VDD_IN` (or `POM_5V_IN` on older boards), else the sum of all rails, and
`power_source` says which; the effective cadence is recorded. Samplers stop in
`try/finally`, at exit, and via `PR_SET_PDEATHSIG`, so even a SIGKILLed run leaves none
behind. Raw traces are gzipped when the sampler stops.

## Metrics (`analyze.py`)

Per run, jobs that arrived in the first `warmup_s` (10 s) are left out.
Latency mean/p50/p95/p99/max overall and per queue, split into queueing, preprocessing
and batch time; SLO miss rate (late + dropped + never served), drop rate, goodput
(frames finished within SLO per second); batch-size histogram per model, mean b,
batches/s, promotions; energy over the window per finished frame, dynamic energy per
frame, busy (inside batches) vs idle energy, mean power, GPU busy fraction; hidden
costs: arrival lateness, decide() time, preprocessing time, model switches and the
measured switch penalty (run time minus profile after a switch vs after the same
model), clock and temperature drift.

Plots: latency CDF per policy; p95 latency and energy per frame vs load; latency vs
energy with the Pareto front (Triton's delay grid becomes a curve); batch-size
histograms; one timeline per run (queue lengths, batches, power on one time axis).
`summary.csv` (one row per run) and `summary.md`.

## Policies

`decide(state) -> Serve | Wait | Promote | Drop`; `state` holds `now`, per-queue job
tuples (arrival, enqueue, deadline, SLO), `t(m, b)` (profile, seconds, linear
interpolation), `b_max`, the profiled sizes. A policy may keep state between calls and
may define `on_batch_done(model, b, ms)`.

| policy | rule |
|---|---|
| `fcfs_nobatch` | queue holding the oldest job, b = 1 |
| `fcfs_batch` | queue holding the oldest job, b = min(\|Q\|, B_max) |
| `lqf` | longest queue (ties: oldest head), b = min(\|Q\|, B_max) |
| `triton` | NVIDIA Triton dynamic batcher, per model (`max_queue_delay_ms`, `max_batch`, `preferred_batch_sizes`, `delays_ms`) |
| `edgeserving` | Cao et al., arXiv 2605.05527 |
| `clockwork` | Gujarati et al., OSDI 2020, INFER scheduling |
| `symphony` | Chen et al., arXiv 2308.07470, deferred batch scheduling |
| `cost_v1` | policy v1 from `CLAUDE.md`: argmin over (queue, b) of beta x sum u(age at completion of jobs left waiting) + (1 - beta) x T_i(b); ms units |

Each reimplementation was written from its paper (and, for Clockwork, its source:
`controller/concurrent_infer_and_load_scheduler.cpp` at gitlab.mpi-sws.org/cld/ml/clockwork).

### Triton (docs: user_guide/batcher.html)

| | Triton | here | why |
|---|---|---|---|
| batching | independent dynamic batcher per model | same | |
| ready when | a max-size or preferred-size batch can be formed, or the oldest request waited `max_queue_delay` | same; the delay counts from enqueue (when the request reached the scheduler) | |
| batch size | largest preferred size formable, else largest possible <= max | max_batch if \|Q\| >= max_batch, else largest preferred formable, else all | the docs do not say what happens when max_batch is not a preferred size |
| several models ready | run concurrently (instances, streams) | one GPU, ready queue with the oldest head first | no concurrency is a non-goal |
| priorities, queue policies, timeouts | optional | not implemented | defaults |

### EdgeServing (Sec. V)

| | paper | here | why |
|---|---|---|---|
| queue choice | argmin stability score S_m after the candidate batch (Eqs. 3, 4, 7) | same | |
| batch | B = min(\|Q_m\|, B_max), B_max = 10 | same, B_max = the run's (16) unless `b_max` is set | same B_max for every policy |
| urgency | f(w) = min(exp(w/tau - 1), C) | same, C = 10 | C is not given as a number; the paper clips tasks past tau(1 + ln 10), i.e. f = 10 |
| tau | one SLO | the job's SLO (one per run anyway) | |
| prediction | served jobs removed; every other job's wait + L(m, B); arrivals ignored | same | |
| latency table | P95 profile (Sec. VI-A) | p95 by default (`stat`) | faithful; the other policies use the run's stat (p50) |
| exit selection (Eq. 6) | deepest exit meeting the SLO | dropped | YOLOv8 has no early exits |
| clocks | locked at max | whatever mode the board is in | sudo may not be available; recorded |
| ties | unspecified | oldest head, then pool order | |

### Clockwork (Sec. 5.3 + source)

| | Clockwork | here | why |
|---|---|---|---|
| strategies | one per (request, supported b), latest = d - T(b), tried most urgent first | same | |
| exec deadline | arrival + SLO - 5 ms buffer | same (`buffer_ms=5`) | |
| valid strategy | finishes by d, enough requests from that one onward | same | |
| growth | to the largest supported size those requests allow; if that misses d the strategy fails (not run at the smaller b) | same | the source's choice, favours big batches |
| older requests that cannot make the batch | dropped ("skipped") | same | |
| requests that cannot finish even alone | cancelled when found | dropped at the next decision | same outcome |
| T(b) | p99 of the last 10 measured runs per (model, b), seeded by profiling | same, seeded from `profile.json` (`online=False`: profile only) | |
| batch sizes | compiled kernels 1, 2, 4, 8, 16 | profiled sizes <= B_max | same default grid |
| schedule ahead | ~5 ms (paper) / 10 ms (source) of queued work to hide RPC latency | `commit_ahead_ms`, 0 by default, pads the start time | local executor, no network |
| LOAD / UNLOAD, workers, memory manager | yes | no: all models resident, one executor | out of scope |
| short-queue reverse scan in `try_dequeue` | yes | no | only fires when every queued request is already past its deadline |
| `--no-drop` | n/a | late jobs skipped by strategies, served alone (oldest first) only when nothing is valid | option asked for |

### Symphony (Sec. 3.1-3.3, Algorithm 1)

| | paper | here | why |
|---|---|---|---|
| candidate per model | GetBatch: from the head, add while the batch still finishes by its earliest deadline | same, start = now | |
| window | exec = max(now, d - l(\|B\| + 1)) (frontrun), latest = d - l(\|B\|) | same | |
| \|B\| = B_max | not discussed (l(B_max + 1) undefined) | exec = now | the batch cannot grow |
| GPU free | the due candidate (exec < free < latest) with the smallest latest | same | |
| nothing due | model timer at exec | `Wait(earliest exec)`; any arrival re-plans | |
| l(b) | affine alpha b + beta per model | the measured table, interpolated | more accurate on curved T(b) |
| drops | requests that would miss the SLO | dropped when they cannot finish even alone | |
| hardware | many GPUs, central coordinator, autoscaling | one GPU | the setting here |
| `margin_ms` | none | 0 by default; plans against deadline - margin | extension, off |

What to expect from Symphony here: each model defers its batch to the edge of its own
window, and with one GPU the windows of different models collide (a deferred yolov8l
batch occupies the GPU exactly when n/s/m reach their frontrun points). On fake models at
11 % load it already dropped ~4 % and finished ~17 % late by under a millisecond, since
it aims at the deadline with no margin. The EdgeServing paper describes Symphony the same
way ("scheduling each model queue independently").

## Running it

### Tests (any machine, CPU)

```
cd edge-runtime
python -m pytest -q tests          # 43 tests, ~80 s
```

### Laptop (RTX 5070 Ti, WSL2): end to end, both input modes

```
cd edge-runtime
cp /path/to/yolov8{n,s,m,l}.pt ../yolo-dispatcher/models/      # or let ultralytics download them
python inputs.py export-frames --scenario S02                  # or --from-coco
python preflight.py
python calibrate.py                                            # -> results/rtx5070ti/profile.json, ~5 min
python traces.py suite --profile results/rtx5070ti/profile.json
python plan.py example > plan_laptop.json                      # edit: traces, slos_ms, input
python plan.py run plan_laptop.json --dry-run
python plan.py run plan_laptop.json
python analyze.py results/rtx5070ti
```

A single 2-minute run by hand (both input modes):

```
P=results/rtx5070ti/profile.json; T=data/traces/periodic_load50.csv
python runtime.py --policy lqf --trace $T --slo-ms 100 --profile $P --input tensor --out results/rtx5070ti/runs/manual_lqf_tensor
python runtime.py --policy lqf --trace $T --slo-ms 100 --profile $P --input frames --prep arrival --out results/rtx5070ti/runs/manual_lqf_frames_arrival
python runtime.py --policy lqf --trace $T --slo-ms 100 --profile $P --input frames --prep executor --out results/rtx5070ti/runs/manual_lqf_frames_executor
```

For a plan covering both input modes, set `"input": ["tensor", "frames"]` and
`"prep": ["arrival", "executor"]` in the plan.

### Native Windows (laptop)

Works, with three differences. The power sampler is `nvidia-smi.exe`, stopped with
`terminate()` (Windows has no SIGINT for a child), so the last ~6 s chunk of its file may
be lost; the 10 s tail covers it. The sampler-cleanup test is skipped (it needs Linux's
`PR_SET_PDEATHSIG`), so a hard-killed run can leave `nvidia-smi.exe` running: check Task
Manager. Timing: use Python >= 3.11, where `time.sleep` and condition waits use
high-resolution timers; older Windows Pythons sleep in ~15.6 ms steps, which shows up as
arrival lateness and late `Wait` wake-ups (logged, but it skews the runs). The run clock
uses `perf_counter`, so timestamps are fine either way. Same commands as above with
`python` and `%VAR%` (cmd) or `$VAR` (PowerShell) instead of the bash variables.

### Jetson visit, step by step

Before the visit (laptop):
1. `git pull` the repo; put `yolov8{n,s,m,l}.pt` in `yolo-dispatcher/models/` (the
   Jetson may not have internet).
2. `python inputs.py export-frames --scenario S02` (frame pool in `data/frames/`).
3. `python -m pytest -q tests`.
4. Copy to the Jetson: the repo folder including `yolo-dispatcher/models/*.pt` and
   `edge-runtime/data/` (frames + the committed CityFlow traces), e.g.
   `rsync -a --exclude venv --exclude '*/dataset' pertinence-batcher/ jetson:pertinence-batcher/`.

On the Jetson:
1. Do NOT pip install torch or torchvision. If ultralytics is missing:
   `pip3 install --no-deps ultralytics ultralytics-thop && pip3 install pyyaml requests scipy tqdm psutil py-cpuinfo`.
2. `cd pertinence-batcher/edge-runtime && python3 preflight.py` (under a minute) until
   it says READY. It checks torchvision's CUDA NMS too (ultralytics needs it).
3. `python3 plan.py visit --budget-min 90` runs, in order and within the budget:
   preflight, calibration (if no profile yet), synthetic traces from the fresh profile,
   a reduced power profile (15 % of the budget), then policy runs in priority order
   (`lqf`, `triton` at 2 and 10 ms, `edgeserving`, `clockwork`, `symphony`,
   `fcfs_nobatch`) on periodic traces at 50 %, 80 %, 20 % load, then sticky and Poisson
   at 50 %, then the CityFlow traces; analysis if matplotlib is there; finally `copyout`.
   Options: `--slo 100` (default `auto`: 100 ms if it is at least 1.5x the slowest
   model at b = 1, else the profile's suggestion, printed as a FLAG), `--input frames`,
   `--run-duration 60` (seconds of trace after the 10 s warmup), `--skip-power`,
   `--power-min`, `--recalibrate`, `--force`.
4. It ends with `COPYOUT: results/<tag>_<time>.tar.gz (N MB)`. Copy that file back.
   Interrupted? Run the same command again: done runs are skipped.

Back on the laptop: untar into `edge-runtime/results/`, `python analyze.py
results/<tag>`, commit `results/<tag>/`.

### Expected runtimes

| step | laptop | Jetson (estimate) |
|---|---|---|
| tests | ~80 s (CPU) | |
| preflight | < 1 min | < 1 min |
| calibration (4 models x 5 sizes) | ~3-5 min | ~8-10 min (time-capped per combo) |
| power profile | ~30-60 min full grid | 13 min reduced (visit default) |
| one run | trace + ~40 s (load, warmup, idle, tail) + cooldown | same |
| `plan.py visit --budget-min 90` | | ~25-30 policy runs of 60 s |
| analysis | < 1 min | |

## Known limits

- decide() time is that of these Python reimplementations, not of Clockwork's C++ or
  Triton's; it is still the overhead this harness's policies add on this CPU.
- Capacity (and so "load") assumes b = B_max batches; with a tight SLO the achievable
  capacity is lower, and on a Jetson T_l(16) can exceed every SLO.
- Default SLOs (50 / 100 / 200 ms) were chosen before any Jetson profile existed; on an
  Orin Nano yolov8l at b = 1 may already be near 100 ms or above (FP32 eager).
  `calibrate.py` flags this and suggests others.
- Board power on a Jetson includes the CPU, memory and everything else on `VDD_IN`;
  dynamic energy subtracts the run's idle-with-models-loaded power.
- tegrastats may not honour `--interval 50`; the effective cadence is recorded.
- Preprocessing in `--prep arrival` runs in a Python thread that shares the GIL with
  the executor: that contention is real for this harness, not for a C++ camera pipeline.
- `cudnn.benchmark` is left off (PyTorch default) everywhere.
- CityFlow point 80 sends no frames to yolov8n/s (S01: 76 % to yolov8l): the COCO heads
  barely discriminate on CCTV (Week6). Regenerate traces from `cityflow-dispatcher`
  routing once it has run.
- Results on the laptop and the A100 are for debugging, not data.
