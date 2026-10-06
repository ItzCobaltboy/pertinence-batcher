# Week 5

## Meeting notes & tasks

**Tasks**:
1. Literature search: has the same or a similar scheduling problem been solved? If yes, adapt it
   instead of rebuilding it.
2. Build a discrete event simulator (DES) for the scheduler. The scheduling problem is decoupled
   from PERTINENCE, so it is designed and validated in simulation first.
3. Once it works in simulation, prove it experimentally with PERTINENCE in the loop.
4. Mail the meeting notes to Prof. Gayathri and ask for a group chat for quick opinions between
   meetings.

Target: compute-constrained edge devices (Jetsons), not datacenter serving.

Later extensions (not this cycle): per-job deadlines, and stream priorities (some video streams
matter more than others).

Next meeting (meet 6) is in two weeks, none in between.

---

## [MEETING] Scheduler direction accepted, simulator first

Presented the formal problem definition and the YOLO correctness work (Week 4). Received well.
The literature review was presented as still to do.

**Key decision**: the scheduling problem stands on its own. The scheduler only sees queues, batch
sizes and runtimes T_i(b), not how frames were routed. So it is built and evaluated in a DES
first, then validated with PERTINENCE doing the routing.

**Scope**: edge devices (Jetsons). The constraint is limited compute on one device, which
separates this from most scheduling/serving work (datacenter scale).

**Extensions for later**: deadlines and stream priorities.

## Log

## [DECISION] Compute metric: report both compression ratio and busy time, pick the primary experimentally

Week 4's journal defined compute as busy time; the Week 4 deck used compression ratio. Settled
before the simulator, since every policy gets scored on it.

- *Compression ratio* K / I: batches run per job (= 1 / mean batch size). 1 means no batching.
- *Busy time* C = sum of T_i(b) over all batches run.

**They agree** for one affine model: T(b) = α + βb gives C = α·K + β·I, and I is fixed, so
minimising C and K is the same.

**They disagree**:
- Convex curves: K rewards slower batches. Week 2 resnet50: one batch of 32 takes 149.6 ms, four
  batches of 8 take 109.0 ms, yet K prefers the 32.
- Mixed pool: K counts a nano batch and a large batch as one unit each.

**Decision**: the simulator reports both. Run the policies on measured YOLOv8 T_i(b) and check
whether ranking by K / I and by C agree. If they agree over our batch sizes, compression ratio is
the simpler, hardware-independent choice. If not, the divergence is itself a result.

## [SETUP] scheduler-sim built: DES engine, 4 v1 policies, 4 routers, tests-first, example sweeps run

First version of the DES in `scheduler-sim/`, self-contained like `yolo-analysis/` (own
`requirements.txt` and `README.md`, no imports from the rest of the repo). Components were
swappable through a name-based registry (`scheduler_sim/registry.py`). (Replaced by the rewrite
below; kept here as the record of what was first built and measured.)

**Built, tests first**:
- `RuntimeProfile`: T_i(b) from CSV; partial batches `pad` (default, like a fixed-shape TensorRT
  engine) or `interpolate`; optional switch cost.
- `Job`, `Workload`/`Stream` with `PeriodicArrival` (+ jitter) and `PoissonArrival`, all drawing
  from a seeded `numpy.random.Generator`.
- 4 routers (`JobRouter` subclasses): `UniformRandomRouter`, `WeightedRandomRouter`, `TraceRouter` (replays a
  `queue_id` CSV, example in `scheduler-sim/data/trace_router_example.csv`), `StickyRouter`
  (per-stream Markov chain: video stays on one model across frames). Plus
  `routers/recall_labeling.py`, which derived weights from
  `yolo-analysis/results/<split>/coco_class_recall_benchmark.csv` (cheapest model with recall
  >= 0.80, recomputed from the raw CSV, not the overwritten `*_minimum_match.csv` files).
- `QueueSet` (one FIFO per model) with a pluggable `DropPolicy` (v1: drop arrivals to a full queue).
- 4 policies (`SchedulingPolicy` subclasses): `FCFSNoBatch` (batch 1, reference), `FCFSBatch` (oldest job's queue, largest batch),
  `LongestQueue`, `TimeoutBatch` (Triton-style: run on full batch or after timeout tau). Invalid
  actions (empty queue, oversized batch, busy accelerator) raise instead of being coerced.
- `Accelerator` + `Simulator`: `heapq` over `ARRIVAL`/`BATCH_DONE`/`TIMER` (+ internal
  `QUEUE_SAMPLE`), fixed tie-breaking at equal times, deterministic per seed.
- `MetricsCollector`: turnaround (mean/p50/p95/p99/max, per queue), both compute metrics,
  throughput/utilization/idle, batch-size histograms, queue length over time, a stability flag
  (trend of total queue occupancy), deadline miss rate (`Job.deadline`/`Job.priority` in the
  schema for the later extensions).

**Tests**: 29 pytest tests written before the policies: periodic arrivals exactly matching
service time, Poisson mean wait within 20% of M/D/1 (rho = 0.5, D = 5 ms, expected Wq = 2.5 ms),
conservation (every job completes, drops, queues or is in flight; none served twice; busy time <=
sim time plus one batch), same-seed determinism, pad/interpolate/switch cost, invalid-action
rejection, router labels on a tiny recall CSV. All passed.

**Sweeps**: `resnet_example` (real resnet18/resnet50 fp32 from
`archive/quantization_experiments/results/exp2_batch_size_sweep.csv`) and `synthetic_4model`
(SYNTHETIC YOLOv8 n/s/m/l: n/s affine, m/l convex, mimicking resnet18 vs resnet50). 4 policies,
5-6 loads, 3 seeds via `experiments/run_experiment.py`, 60 ms warm-up excluded (warm-up later dropped, see below):

| experiment | policy | mean turnaround (ms) | mean utilization | mean compression ratio |
|---|---|---|---|---|
| yolo (synthetic, real recall-derived routing) | fcfs_no_batch | 5.95 | 0.462 | 1.000 |
| yolo | fcfs_batch | 5.20 | 0.446 | 0.909 |
| yolo | longest_queue | 5.05 | 0.449 | 0.918 |
| yolo | timeout_batch (tau=15ms) | 17.80 | 0.382 | 0.583 |
| resnet (real profile, 50/50 routing) | fcfs_no_batch | 5997.2 (unstable past rho~0.7) | 0.816 | 1.029 |
| resnet | fcfs_batch | 43.8 | 0.765 | 0.597 |
| resnet | longest_queue | 44.5 | 0.766 | 0.598 |
| resnet | timeout_batch (tau=15ms) | 52.6 | 0.649 | 0.393 |

YOLO routing weights from val2017 at recall >= 0.80: `[0.528 n, 0.149 s, 0.083 m, 0.240 l]`, so
the nano queue carries most load. `fcfs_no_batch` goes unstable first on both profiles (no
batching, no headroom); `longest_queue` and `fcfs_batch` track each other; `timeout_batch` trades
turnaround for a much lower compression ratio (tau forces early runs). Busy time and compression
ratio did not always rank the policies the same (`ranking_agreement.png`): a first data point for
the metric decision, not conclusive on synthetic curves. Outputs: `results.csv` and plots
(`turnaround_vs_busy_time.png`, `*_vs_load.png`, `ranking_agreement.png`) per experiment in
`scheduler-sim/results/`.

**Open at that point**: synthetic YOLO numbers only test the engine; real batched YOLOv8 T_i(b)
needed (extend `yolo-analysis/` to batched timing); literature review; PERTINENCE-in-the-loop
validation.

---

## [RESULT] Literature search done: the multi-queue batch-scheduling gap looks open, the power lever is not new on its own

About 50 papers, each with a link and a "what differs from us" note, in `research.md` (edge
multi-DNN, real-time edge, batch-service queueing, incompatible job families, datacenter serving,
LLM serving, energy/DVFS).

**Closest**: EdgeServing (arXiv 2605.05527). Same setting: several models time-share one GPU,
own queue each, measured latency table, tested on Jetson Orin Nano. Differs: objective is SLO
violations (not turnaround/compute), adds an early-exit choice, and always dispatches
min(queue length, B_max), so it never waits or picks a smaller batch on purpose.

**Theory**: Xia et al. 2002, Chen & Wang 2022, Duenyas & Neale 1997 (incompatible job families,
best keyword anchor) assume batch cost independent of batch size; their proofs break once T_i(b)
is size-dependent and per model. That is the open part. Concurrent or preemptive edge systems
(BCEdge, SEEB-GPU, Pantheon, Fluid Batching, RT-mDL) break one-batch-at-a-time, so they are a
different problem.

**Power**: Camel (Jetson AGX Orin) and Nabavinejad et al. (TPDS 2022) tune batch size and GPU
frequency together for delay/energy, but for one model. Nobody does it across N heterogeneous
queues with queue choice.

**Leaning** (not decided): T_i(b) -> T_i(b, f) with measured power; policy picks (queue, batch
size, frequency), one batch at a time. MPS stays a separate thread.

**Corrections**: polling-with-set-up-costs is Duenyas & Van Oyen (1995), not Koole (1998);
Whittle index is Glazebrook, Lumley & Ansell (2003). **Unverified** (paywalled): whether Chen &
Wang's finite-capacity extension is a full proof; Nabavinejad's headline number.

---

## Idea (open, not yet decided): size-aware parallel scheduling via MPS

A possible way around "one batch at a time". CUDA MPS runs kernels from several processes
concurrently. Small-batch inference is mostly memory-bandwidth-bound, so compute sits partly idle
(which is why batching helps). If two small models fit within the bandwidth budget, co-running
their batches under MPS might cost little extra. The scheduler would then also decide whether
two batches can safely co-run or must be serialized (**size-aware parallel scheduling**).

**Not free**: nobody characterizes this cleanly. BCEdge learns an RL interference predictor;
SEEB-GPU partitions compute units per model (TPC masking) instead of sharing. The real
slowdown-vs-concurrency curve for small edge models is unmeasured.

**Needed first**: measure runtime under MPS on the target hardware (does contention, launch
overhead or SM fragmentation eat the gain?). MPS on Jetson needs JetPack 6.1 / CUDA 12.5 and has
reported container/Kubernetes issues.

Not pursued yet. It also marks "one batch at a time" as a simplifying assumption, not a hardware
limit.

---

## Idea (open, not yet decided): what "building a policy" actually means for this scheduler

**Shape**: take the state (queue lengths, wait times, T_i(b), ...), score every option (serve
queue i at size b, or wait), take the best. A greedy/myopic index policy (cμ-rule, CAW index are
classic examples): score what looks best right now, every time the accelerator frees up.

**Catch**: "wait" can't be scored as zero. Its value depends on whether jobs will arrive soon to
fill a bigger batch, which is probabilistic. Arrivals are periodic overall (a frame every Δ), but
the dispatcher routes by frame content, so per queue it is a periodic tick randomly thinned by
routing.

**Plan**: no textbook distribution (the OR papers' Poisson doesn't match). Keep a sliding window
of the last X arrivals, estimate each queue's landing probability from it, and use that to value
waiting against serving now. X needs tuning later (short = noisy, long = slow to react to routing
shifts such as a scene change).

Net: state -> score per option (with a stochastic wait term from recent history) -> pick the max.

---

## [SETUP] scheduler-sim rewritten as plain student-style OOP, same results within noise

The first version worked but I couldn't read it top to bottom (registries, ABCs, frozen view
dataclasses, decorators, import-time registration, an unused JSONL tracer and hooks). Rebuilt it
as three plain files:

- `sim.py`: `Job`, `Queue`, `Profile` (T_i(b) table), base classes `Workload` and `Scheduler`,
  `Simulator` (one heap, one loop), `compute_metrics`.
- `workloads.py`: `UniformWorkload`, `WeightedWorkload`, `StickyWorkload`, `PeriodicWorkload`
  (`PeriodicRoutedWorkload` added later, see below).
- `schedulers.py`: the same 4 policies (`fcfs_no_batch`, `fcfs_batch`, `longest_queue`,
  `timeout_batch` tau = 15 ms), same tie-breaks.

Two knobs: a Workload decides arrivals and target queue (usually only `choose_queue`); a
Scheduler decides the next queue and batch size (only `decide`). Both share the same `Queue`
objects. The Simulator still raises on invalid decisions. Metric keys and `results.csv` columns
unchanged; the `stable` heuristic still wrongly flags some very light loads (TODO).

**Removed**: registries, ABCs, hooks, JSONL tracer, deadline/priority fields, trace router,
`recall_labeling.py`, interpolate mode, switch cost, the live recall-CSV read. YOLO weights are
hardcoded from val2017 at recall >= 0.80: `[0.5278, 0.149, 0.0834, 0.2398]` for n/s/m/l (2639 /
745 / 417 / 1199 of 5000 images).

**Regression check**: one RNG now drives arrivals and routing, so no bit-for-bit match. On
`yolo_synthetic_load_sweep` every seed-averaged turnaround, utilization and compression ratio is
within 7% and rankings are identical. On `resnet_load_sweep`, loads 0.05-0.20 are within 6% with
identical rankings; at 0.25-0.30 a few cells moved 10-55% and the 3-seed ranking flipped, but
those loads run at 97-100% utilization where the old code's own seed spread was ~40%. Rerunning
old and new code on 20 fresh seeds: agreement within 1-2 standard errors, same ranking. Seed noise,
not a behaviour change.

**New** `yolo_sticky_load_sweep`: `StickyWorkload`, stay_probability = 0.9 (streams keep hitting
the same queue, like video). `longest_queue` wins at every load, everything is 5-15% slower. Only
ranking change: at load 0.25 `fcfs_no_batch` drops to last (21.2 ms vs 9.9 ms without
stickiness). Bursts hurt the policy that can't batch most.

---

## [SETUP] scheduler-sim follow-ups: warm-up dropped, raw data saved, camera-like periodic workload

- **Warm-up dropped** (commit e9b2004): the cutoff counted jobs by arrival time but batches by
  start time, so warm-up backlog inflated compression ratio on overloaded runs (`fcfs_no_batch`
  read up to 1.088 instead of 1.0). Metrics now run from t=0: one population, and the empty-start
  bias shrinks with run length. Stable cells moved under 1% mostly (max 5.5% at 97% utilization).
- **Raw data saved** (26c3d43): every run writes all jobs, batches and queue samples to a
  compressed `.npz` under `results/<exp>/raw/` (git-ignored, runs are seed-deterministic).
  `compute_metrics` reads only that. `SIMULATE = False` in `run_experiment.py` recomputes all CSVs
  and plots in ~10 s, so new metrics need no rerun.
- **`PeriodicRoutedWorkload`** (eb4b280): cameras. Each stream sends a frame every
  `period_ms = num_streams / load` (8 streams at 0.25 jobs/ms = one frame every 32 ms per camera,
  ~30 fps), routed by the recall weights. New `yolo_periodic_load_sweep`: all cameras start at
  t=0, so frames arrive in bursts of 8 and turnaround (~10 ms at every load, ~9.9-10.0 ms for
  `fcfs_batch`) mostly measures burst drain. Documented as a worst case; staggered start phases
  are a TODO.

---

## [SETUP] Batched-inference timing sweep for real YOLOv8 T_i(b), running on the A100

`yolo-analysis/batch_sweep/` (entry point `sweep.py`: preflight | measure | aggregate | profile)
replaces the synthetic YOLO profile with measured curves.

- **Configs**: YOLOv8n/s/m/l × b = 1, 2, 4, 8, 12, 16, 32, 48 × three variants: eager_fp32
  (plain PyTorch), trt_fp32 (torch_tensorrt dynamo) and trt_fp16.
- **Timed**: forward = fused DetectionModel incl. the Detect head's box decode, on a batch
  already on the GPU, synchronized before and after. NMS is a separate column.
- **Repeats**: 20 warmup + 100 timed batches per run, 5 runs per config, each run a fresh process
  with a fresh model/engine load. Rounds shuffled so drift on the shared GPU spreads evenly.
- **Data**: a seeded pool of real images from the same val2017/train2017 subset folders
  yolo-analysis uses, letterboxed like `predict()`, preloaded before timing.
- **Engines**: compiled once per (model, batch size) and cached (64 compiles).
- **Logging**: raw per-batch timings per run, plus nvidia-smi clocks, temperature, throttling and
  other processes at start and end. Resumable after a kill; failures are recorded, not fatal.

CPU testing on torch 2.3.1 caught a blocker: torch 2.3 export rejects YOLOv8 because the model
and its Detect head share one stride tensor. Cloning the model-level copy (forward never reads it)
fixes export for the full model, head included. "fp32" on the A100 is TF32 by default for eager
and TRT; kept the defaults and logged the flags.

**Status**: finished, see the `[RESULT]` entry below. Next:
aggregate, choose a variant (`sweep.py profile --variant <v>` writes
`scheduler-sim/profiles/yolov8_a100_<v>.csv`), rerun the YOLO sim sweeps on it.

---

## [RESULT] EdgeServing read in full: max batching, never waits, clocks locked

Read EdgeServing (arXiv 2605.05527) properly.

**Setup**: ResNet50/101/152 on CIFAR-100, one FIFO queue each, one batch at a time (MPS and
spatial sharing rejected as unpredictable). Offline P95 table L(model, exit, batch) for b = 1-10
(CV < 3%), GPU clocks locked at max. 4 exits per model (layer1/2/3/final; ResNet152 accuracy
7.3% at layer1 vs 78.0% at final). Poisson arrivals at 3:2:1, SLO 50 ms (100 ms on Jetson),
20 s runs. RTX 3080 main, plus GTX 1650 and Jetson Orin Nano.

**Scheduler** (one-step greedy, each time the GPU frees): batch B = min(|Q|, 10), never chosen;
deepest exit that keeps the oldest job within the SLO; for each queue, predict everyone's waits
if it is served, score sum of min(exp(w/τ - 1), C) over all waiting jobs ("stability score"),
serve the lowest. Future arrivals ignored; no wait option.

**Results**: under 1% SLO violations where All-Final and Symphony degrade. **Ablation** at
120 req/s: theirs 42.22 ms P95, EDF 44.02 ms, LQF 46.01 ms, so queue choice adds only 2-4 ms;
disabling early exit blows up. Most of the gain is early exit, not scheduling. Batch size 1 is
worse everywhere, so batching matters.

**What this means for me**:
- On flat curves (theirs: 2-3x latency for 10x batch) max batching is near optimal for
  turnaround. Waiting only pays off when each batch carries a fixed cost to spread over more jobs
  (classic threshold result for batch-service queues, Deb & Serfozo 1973). Energy is that cost.
- Max batching can still lose past the knee (T_i(b)/b stops dropping) and to head-of-line
  blocking by a big batch of a heavy model.
- Their gaps are my angle: no batch-size or wait decision, no energy objective, clocks locked at
  max (frequency as a knob is the opposite choice).
- Easy extra baseline: stability score + max batch (no early exit) in `schedulers.py`.

---

## [DECISION] Direction for meet 6: energy is the angle, power before MPS

**Decision**: pitch turnaround + energy as the objective, with the policy choosing (queue, batch
size, wait) and later frequency. Turnaround alone leaves little room over max batching (see
above); with energy in the objective, choosing b and waiting become real decisions that
EdgeServing cannot express. Power work comes before MPS; MPS stays future work (needs the Jetson
and its own slowdown study).

**Hardware reality**: no Jetson yet (ask at meet 6). A100 and 5070 Ti are both far stronger than
AGX Orin (roughly 2048 CUDA cores, ~200 GB/s vs ~9000 cores, ~900 GB/s on the 5070 Ti), so
curves there look flat and batching looks nearly free. Workarounds: also run the sweep on the
5070 Ti; imgsz 1280 on the A100 as a labelled stand-in for a weaker device (~4x work per image
fills the GPU sooner). Power draw can be read through NVML without root on the A100, so E_i(b)
(energy per batch) can be logged now; frequency control needs root, so that waits for the Jetson.

**Meet 6 target**: real curves in the sim with baselines rerun, EdgeServing comparison, the
energy argument, a definitions slide (turnaround, K / I vs C, T_i(b), load), and asks: Jetson
access, go/no-go on the energy direction. Stretch: an E_i(b) plot and the policy written as
equations (score = extra waiting + λ × energy, plus a wait score; at λ = 0 with no wait it should
reduce to roughly EdgeServing).

**Roadmap** (changes if results say so):
1. Real T_i(b) in the sim, baselines rerun.
2. Log E_i(b); write the score-based policy with a wait option; add an EdgeServing-style baseline.
3. Sweep load, workload type, hardware profile and λ; target result is a turnaround-vs-energy
   curve beating the baselines' single points.
4. Jetson: frequency as a knob, check real numbers against the sim.
5. Real workload: multi-camera surveillance traces (e.g. MOT17/20, VIRAT, WILDTRACK, AI City;
   check licences) through PERTINENCE, logged as (time, camera, model) and replayed in sim and on
   device. Adds frames-on-time / dropped-frames metrics.
6. Paper.

Rethink points: if energy barely varies with b on real curves; if the policy barely beats max
batching after step 3 (then frequency or MPS must carry it).

---

## [RESULT] Real YOLOv8 T_i(b) measured on the A100, simulator rerun on the real curves

**Sweep quality**: 480/480 runs ok, 0 failures, no other GPU processes in any run, every TRT
engine fully TensorRT (0 PyTorch fallback subgraphs). Run-to-run CV: median 0.35%, max ~7%
(yolov8s trt_fp16 b=1). Nearly all 92 sanity flags are `sw_power_cap`: the A100 hits its power
limit at big batches and SM clocks sag from 1410 to ~1320 MHz. That is real hardware behaviour,
not interference. The "per-image cost not decreasing" flags are <2% wiggles on the plateau.

**trt_fp16 T_i(b) (ms)**, now `scheduler-sim/profiles/yolov8_a100_trt_fp16.csv`:

| b | n | s | m | l |
|---|---|---|---|---|
| 1 | 1.66 | 1.88 | 2.35 | 2.94 |
| 4 | 2.24 | 2.82 | 4.25 | 6.16 |
| 8 | 2.85 | 3.77 | 7.19 | 11.03 |
| 16 | 4.04 | 6.26 | 13.18 | 20.95 |
| 32 | 6.76 | 11.31 | 24.82 | 40.28 |
| 48 | 9.62 | 16.53 | 37.09 | 60.28 |

- **Not flat, and very different per model**: per-image cost drops ~8x for n (1.66 -> 0.20 ms)
  but only ~2.3x for l (2.94 -> 1.26 ms). m and l hit their knee around b=8 (past it, bigger
  batches only add waiting); n and s keep gaining until ~32. This is the heterogeneous,
  size-dependent T_i(b) the problem assumes, visible even on an A100.
- **eager_fp32 is launch-bound**: ~5 ms floor (n is 5.0 ms at b=1 and 5.3 ms at b=8), so it is
  not a useful scheduler profile. trt_fp32 has the same shape as fp16 at ~1.5-2x the time
  (and is TF32 on the A100). Picked **trt_fp16** as the profile: it is the deployment path.
- **The synthetic profile was badly off**: l was ~3x too slow at b=32 (131 vs 40 ms) and far too
  convex; n's slope ~2x too steep. Earlier synthetic YOLO sim results don't carry over.
- **Power**: the sweep only logged one power reading at the end of each run (60-250 W depending
  on whether the GPU already idled), useless for energy. E_i(b) needs NVML sampling during the
  timed loop.

**Simulator on the real profile**: three new experiments, same setups as the synthetic ones,
loads 0.1-1.0 jobs/ms (batch-1 capacity is ~0.49 jobs/ms with these routing weights, max-batch
~1.9): `yolo_real_load_sweep`, `yolo_real_sticky_load_sweep`, `yolo_real_periodic_load_sweep`.
Mean turnaround (ms), seed-averaged:

| load | fcfs_no_batch | fcfs_batch | longest_queue | timeout_batch |
|---|---|---|---|---|
| 0.10 | 2.35 | 2.32 | 2.31 | 15.03 |
| 0.30 | 3.82 | 3.12 | 3.07 | 14.09 |
| 0.45 | 16.31 | 4.06 | 3.95 | 14.13 |
| 0.60 | unstable (5732) | 5.34 | 5.15 | 14.41 |
| 1.00 | unstable (15443) | 10.97 | 10.83 | 16.31 |

- `fcfs_no_batch` collapses right past its ~0.49 capacity, as predicted from the curves.
- `longest_queue` and `fcfs_batch` are within ~0.3 ms at every load (Poisson and sticky); on the
  periodic sweep at load 1.0, `longest_queue` is ~11% better (9.42 vs 10.63 ms).
- **The useful one: `timeout_batch` trades latency for compute.** At load 0.3 it costs ~11 ms
  more turnaround but uses 21.3 s of GPU busy time vs 34.4 s for `fcfs_batch` (~38% less) with
  K/I 0.42 vs 0.91. Waiting to fill batches really does buy compute, which is the energy argument
  in numbers.
- **Compute metric**: on the real curves, compression ratio and busy time rank the four policies
  the same way at every load in these runs. By the Week 5 `[DECISION]` rule that points to
  compression ratio as the simpler primary metric, but only four policies were compared, so this
  is not final.
- Periodic sweep: K/I sits at ~0.5 at low load (synchronized bursts give batches of ~2); still the
  t=0 worst case.

**Cleanup**: deleted everything built on the old data: `synthetic_4model.csv`, `resnet_example.csv`,
`build_profiles.py` and their four sweeps (`resnet_load_sweep`, `yolo_synthetic_load_sweep`, and the
synthetic `yolo_sticky`/`yolo_periodic` sweeps). The real-profile experiments were renamed to
`yolo_load_sweep`, `yolo_sticky_load_sweep`, `yolo_periodic_load_sweep`; numbers unchanged
(recomputed from the saved raw data). Still in git history if ever needed.

Next: E_i(b) via NVML sampling, then the score-based policy (turnaround + λ × energy with a wait
option) and an EdgeServing-style baseline on this profile.

---

## [SETUP] Dense batch-size sweep, all three variants, on the A100

The `main` sweep had only 8 batch sizes, too coarse to see where each curve bends (m/l knee near
b=8, n/s near 16-32), and the simulator pads an unmeasured size up to the next measured one (b=5
costs T(8)). Reran all three variants (eager_fp32, trt_fp32, trt_fp16) at b = 1, 2, 4, 6, ..., 32
in steps of 2 where the curves bend, then 40, 48, 56, 64 where they are already linear. Going to
64 so the simulator can use larger batches at high load. Ran it in the same session `main`: the
sweep resumes, so the 8 sizes already measured were skipped and only new sizes ran (cached
engines reused). Power is still not sampled during the timed loop (E_i(b) needs a separate
change).

**Done**: 1260/1260 runs ok (252 configs: 3 variants × 4 models × 21 sizes, 5 runs each), 0
failures, every TRT engine fully TensorRT. trt_fp16 T(b) increases monotonically at all 21 sizes
for all four models and matches the first sweep within a few percent (b=1: 1.62/1.87/2.47/2.99 ms
for n/s/m/l). Per-image cost flattens from b~8 for l (1.39 -> 1.22 ms at 64), b~10 for m (0.87 ->
0.76), and keeps falling to b~32 for n and s (0.21 and 0.35 ms), flat after. Past the knee l is
linear (~1.20 ms per extra image from 8-32 and 32-64); n bends up slightly (0.16 -> 0.18).
n and s are noisier at small b (CV 2-8%, vs <0.5% for m/l): their batches take only 2-4 ms, so
launch/sync jitter (TRT warns it uses the default stream) is a bigger share. Small bumps at sizes
that aren't multiples of 8 (e.g. s at b=10, 18) are TensorRT's per-engine kernel choice, real
behaviour. A NumPy 2.2.6 vs onnxruntime ABI error shows in the logs: it comes from the
version-logging import (torchvision -> torch.onnx -> onnxruntime), not the timed path; onnxruntime is
unused. The server venv has numpy 2.2.6, not 1.26.4.

Plan meanwhile: no Jetson yet, so hardware work beyond this sweep waits; next is designing the
score-based policy.

---

## [DECISION] Scheduler policy: cost model v1, built from first principles

Built the first version of my own policy in one sitting, from scratch rather than adapting a
paper's rule.

**Notation**: M queues Q_1..Q_M; N_i(t) = jobs in queue i at decision time t; batch sizes
B_1..B_k; T_ij = latency of model i at batch size B_j (the measured profile).

**Options**: at every decision there are M·k + 1 options: serve (queue i, batch size j), or wait.
Padding is allowed, so a batch can run with B_j > N_i(t).

**What drives the cost**, given the goal of minimising turnaround and power:
1. Every job not run hurts, so small batches are bad.
2. Longer execution hurts (roughly, power), so big batches are bad.

**Structure**: give every option a cost and pick the cheapest.

    Cost(i, j) = β · (cost of jobs not run) + (1 - β) · (energy cost of running batch B_j)

β is a user-chosen knob trading latency against energy.

**Job cost**: only jobs not run now cost anything, and the cost grows the longer a job has
waited. With J_p = time the p-th job has spent in its queue so far:

    Cost_jobs(i, j) = Σ J_p over all jobs p not chosen by option (i, j)

For now J_p is the age right now; I ignore how much longer the unchosen jobs will keep waiting
after this decision. Linear in J_p for now (exponential is an option).

**Energy cost**: assume constant power, so energy is just T_ij.

**Padding needs no special rule**: a bigger batch removes one more job from the not-run cost only
while a real job fills the slot. For B_j > N_i(t) the energy cost keeps rising while the job cost
stops falling, so padding prices itself out.

**Policy v1**:

    Cost(i, j) = β · Σ_{p unchosen} J_p + (1 - β) · T_ij
    pick argmin over all (i, j); no wait option yet

**Parked, not decided**:
- Cost of the wait option (not defined yet).
- Time horizon: unchosen jobs keep aging during T_ij, not counted yet.
- Linear vs exponential growth of the job cost with J_p.
- Real power P_ij per (model, batch size) instead of constant power.
- Scale mismatch: the job term grows with queue length while T_ij doesn't, so the same β will
  behave differently at light and heavy load.

Status: theory only, not implemented in `scheduler-sim` yet. Planned as the policy to present at
meet 6.

---

## [RESULT] Simulator rerun on the dense profile

Rebuilt `scheduler-sim/profiles/yolov8_a100_trt_fp16.csv` from `results/batch_sweep_DENSE` (21
columns, b = 1..32 step 2, 40, 48, 56, 64) and reran the three sweeps at loads 0.1-1.0 step 0.1.

Poisson (`yolo_load_sweep`), seed-averaged:

| load | metric | fcfs_no_batch | fcfs_batch | longest_queue | timeout_batch |
|---|---|---|---|---|---|
| 0.1 | mean / p95 ms | 2.35 / 4.17 | 2.33 / 4.07 | 2.32 / 4.00 | 15.04 / 18.96 |
| 0.3 | mean / p95 ms | 3.85 / 8.92 | 3.15 / 6.28 | 3.09 / 6.52 | 14.00 / 21.17 |
| 0.5 | mean / p95 ms | 922 (unstable) | 4.49 / 9.02 | 4.35 / 10.30 | 13.94 / 22.01 |
| 1.0 | mean / p95 ms | unstable | 9.77 / 18.69 | 9.36 / 22.69 | 15.20 / 25.99 |
| 0.3 | K/I, busy s | 1.0, 37.0 | 0.907, 34.5 | 0.917, 34.7 | 0.415, 21.2 |

- Same picture as on the 8-point profile, numbers moved by a few percent.
- **New**: `longest_queue` has the best mean turnaround but a worse p95 than `fcfs_batch` from
  load ~0.3 up (22.7 vs 18.7 ms at 1.0). Serving the longest queue starves short queues, which
  shows up in the tail. Mean-only comparisons hide this.
- `timeout_batch` still cuts GPU busy time ~38% at load 0.3 for ~11 ms more mean turnaround.
- Sticky: same ordering, everything slightly slower. Periodic (synced starts): batching
  policies flat at ~6.1 ms up to load 0.7; `longest_queue` better mean and p95 at high load.

---

## [DEAD-END] TensorRT engines give wrong outputs; simulator switched to the eager FP32 profile

Proofreading the batch-sweep methodology turned up a real problem. At compile time the sweep runs
each TRT engine on one batch next to the eager model and stores the comparison
(`check_vs_eager` in each compile record), but never fails on it. Across all 168 compiled engines
(OLD and DENSE runs) the outputs do not match: after NMS, eager finds 14,093 detections and the
trt_fp32 engines 2,249 (trt_fp16: 2,265 vs 14,574), with max output differences up to ~830
(pixel scale). It happens in FP32 too, so it is an export bug, not precision.

Most likely only the box decoding is broken: the Detect head's anchors/strides look lost in
`torch.export` (the run logs warn "get_attr Node with no underlying reference"). The mean
difference over the whole (B, 84, 8400) output is only ~0.7, which fits the 4 box rows being off
by ~15 px with the 80 class-score rows near exact. If so, the convolutions are correct and the TRT
timings are close to right, but that is not verified yet.

**Decision**: until the engines pass the check, the simulator uses the eager FP32 profile
(`scheduler-sim/profiles/yolov8_a100_eager_fp32.csv`, built from `batch_sweep_DENSE`). Eager is
correct but launch-bound (~5 ms floor for small batches), so its latencies are an upper bound on a
deployed engine. The trt_fp16 profile is kept for later.

**Simulator rerun on eager** (capacity: ~0.17 jobs/ms at batch 1, ~0.69 at batch 64, so loads are
now 0.05-0.7). Poisson, seed-averaged:
- `fcfs_no_batch` saturates past ~0.17 jobs/ms; every policy saturates near 0.69.
- `longest_queue` has the best mean; its p95 is the worst of the batching policies at loads
  0.25-0.5 (39.3 vs 30.8 ms for `fcfs_batch` at 0.3).
- Timeout vs longest queue: 0.05: 6.9 -> 19.9 ms for 18% less GPU time; 0.1: 8.2 -> 19.6 ms, 26%
  less; 0.2: 11.8 -> 19.9 ms, 28% less; 0.5: 28.0 -> 30.6 ms, 3% less. Same conclusion as on TRT:
  the fixed wait is mispriced at light load and pays off in a middle band.

**Other methodology notes** (no change needed now): T(b) is forward-only (NMS, preprocessing,
host-to-GPU copy excluded), so simulated latencies are lower bounds on end-to-end; one fixed-shape
engine per batch size is fine on the A100 but unlikely on a Jetson; turnaround counts only jobs
finished within the 60 s horizon (understates overload, only matters past saturation); busy time as
an energy proxy assumes constant power, but the A100 throttles at large batches; the simulator
rejects batches larger than the queue, which policy v1's padding will need relaxed.

**Next**: a quick server check (compare score rows vs box rows of one engine against eager), then
fix the export (keep anchors/strides, or compile only backbone + head and decode in PyTorch),
re-time a few configs, and switch the profile back.
