# Week 5

## Meeting notes & tasks

**Tasks**:
1. Literature search: find whether the same or a similar scheduling problem has already been
   solved. If the same work exists, adapt it to our setting rather than rebuild it.
2. Build a discrete event simulator for the scheduler. The scheduling problem is decoupled from
   PERTINENCE itself, so it gets designed and validated in simulation first.
3. Once the scheduler works in simulation, prove it experimentally with PERTINENCE in the loop.
4. Mail the meeting notes to Prof. Gayathri and ask for a group chat for occasional quick
   opinions between meetings.

Target setting: scheduling on compute-constrained edge devices (Jetsons), not large-scale
datacenter serving.

Later extensions (not this cycle):
- Real-time constraints: per-job deadlines.
- Priority-based operation: some video streams matter more than others.

Next meeting (meet 6) is in two weeks, no meeting in between.

---

## [MEETING] Scheduler direction accepted, simulator first

Presented the formal problem definition and the YOLO correctness work (Week 4). The direction was
received well.

The literature review was presented as still to do. No prior-art findings were shown in this
meeting.

**Key decision**: the scheduling problem stands on its own. The scheduler only sees queues, batch
sizes and runtimes T_i(b), not how frames got routed. So it gets built and evaluated in a discrete
event simulator first, and only then validated experimentally with PERTINENCE doing the routing.

**Scope**: optimise for edge devices (Jetsons). This separates the work from most of the existing
scheduling and serving literature, which targets large-scale, datacenter-sized optimisation. The
constraint here is limited compute on one device.

**Extensions** raised for later: deadlines (real-time constraints) and stream priorities.

## Log

## [DECISION] Compute metric: report both compression ratio and busy time, pick the primary experimentally

The Week 4 journal defined compute as accelerator busy time, while the Week 4 deck used
compression ratio. Settling it before the simulator, since it's the number every policy gets
scored on.

**The two candidates**:
- *Compression ratio* K / I: batches run per job processed (1 / mean batch size). K / I = 1
  means no batching.
- *Busy time* C = sum of T_i(b) over every batch run: total time the accelerator spends computing.

**When they agree**: if a model's runtime is affine, T(b) = α + βb, total busy time is
α·K + β·I. I is fixed, so for a single model, minimizing busy time and minimizing K are the same
thing.

**When they don't**:
- Convex curves: once per-image cost rises with batch size, K rewards batches that are slower. On
  the Week 2 resnet50 numbers, one batch of 32 takes 149.6 ms while four batches of 8 take
  109.0 ms, yet K prefers the batch of 32.
- A heterogeneous pool: K counts a nano batch and a large batch as one unit each, so a lower K
  across models doesn't mean less compute.

**Decision**: the simulator reports both. Only one will be the primary objective, and which one
gets decided experimentally: run the candidate policies on the measured YOLOv8 T_i(b) curves and
check whether ranking policies by K / I and by busy time gives the same order. If they agree over
the batch sizes we allow, compression ratio is the simpler, hardware-independent choice. If they
diverge, the divergence itself is the result to look at.

## [SETUP] scheduler-sim built: DES engine, 4 v1 policies, 4 routers, tests-first, example sweeps run

Built the discrete-event simulator this week's meeting called for, under a new top-level
`scheduler-sim/` folder — same self-contained convention as `yolo-analysis/`: own
`requirements.txt`, own `README.md`, no imports from the rest of the repo. Everything is
object-oriented and swappable through a small name-based registry (`scheduler_sim/registry.py`),
so a config picks a router or policy by string without the engine caring which one it got.

**Built, in the "tests first" order the spec asked for**:
- `RuntimeProfile` — the T_i(b) matrix, loaded from CSV, `pad` (default, matches a fixed-shape
  TensorRT engine rounding a partial batch up) or `interpolate` partial-batch handling, optional
  per-switch cost.
- `Job`, `Workload`/`Stream` with `PeriodicArrival` (+ jitter) and `PoissonArrival`, both drawing
  from an explicitly seeded `numpy.random.Generator` — nothing touches python's global `random`.
- `JobRouter` + 4 implementations: `UniformRandomRouter`, `WeightedRandomRouter`, `TraceRouter`
  (replays a `queue_id`-column CSV — dummy example at `scheduler-sim/data/trace_router_example.csv`),
  `StickyRouter` (per-stream Markov chain, mimics real video staying on one model across
  consecutive frames). Plus `routers/recall_labeling.py`: derives router weights/traces straight
  from `yolo-analysis/results/<split>/coco_class_recall_benchmark.csv` (cheapest model with
  recall >= 0.80, the threshold this project settled on in Week4, recomputed from the raw CSV
  every call rather than trusting the `*_minimum_match.csv` files that get overwritten).
- `QueueSet` (N FIFO queues, one per model) behind a pluggable `DropPolicy` (v1: drop the
  arriving job when a queue is full).
- `SchedulingPolicy` + 4 v1 policies: `FCFSNoBatch` (reference baseline, batch size 1),
  `FCFSBatch` (oldest job's queue, largest batch it can fill), `LongestQueue`, `TimeoutBatch`
  (Triton-style: run on filling max batch size or a per-queue timeout tau, whichever first).
  Every returned action is validated before the engine acts on it — an invalid one (empty queue,
  oversized batch, busy accelerator) raises loudly instead of getting coerced.
- `Accelerator` (single server, tracks busy intervals + current model) and the `Simulator` engine
  itself: a `heapq` event queue over `ARRIVAL`/`BATCH_DONE`/`TIMER` (+ an internal
  `QUEUE_SAMPLE` for the queue-length-over-time metric), fixed type-priority tie-breaking at
  equal simulated time, fully deterministic given its seed.
- `MetricsCollector`: turnaround (mean/p50/p95/p99/max, per queue), both compute candidates from
  the compression-ratio-vs-busy-time decision above, throughput/utilization/idle time, per-model
  batch-size histograms, queue-length-over-time + max queue length, a stability flag (linear
  trend on sampled total queue occupancy), deadline miss rate (unused by any v1 policy, but the
  field is live — `Job.deadline`/`Job.priority` are in the schema already so the deadline/
  priority extensions this week's meeting notes flagged for later don't need a data-model
  rewrite).

**Tests first, actually first**: wrote the pytest suite (29 tests) against the interfaces before
finishing the policy implementations, per spec. Covers deterministic periodic arrivals matching
service time exactly, Poisson mean wait matching the M/D/1 formula within 20% (rho=0.5,
D=5ms -> expected Wq=2.5ms), conservation (every arrived job completes/drops/queues/is
in-flight, no job served twice, busy time never exceeds sim time beyond one in-flight batch's
slack), same-seed determinism, `RuntimeProfile` pad/interpolate/switch-cost logic, invalid-action
rejection, and router label derivation on a tiny hand-made recall CSV. All 29 pass.

**Example profiles + sweeps, actually run**: `resnet_example` (real resnet18/resnet50 fp32
matrix, straight from `archive/quantization_experiments/results/exp2_batch_size_sweep.csv`) and
`synthetic_4model` (clearly-labeled SYNTHETIC YOLOv8 n/s/m/l stand-in — n/s affine, m/l convex,
matching the real resnet18-vs-resnet50 shape split — until real batched YOLO curves exist).
Ran both through `experiments/run_experiment.py`'s hardcoded policy x load x seed sweep
(4 policies, 5-6 load levels, 3 seeds each, 60ms warmup excluded). Headline numbers:

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

The routing weights for the YOLO sweep came from the real val2017 recall benchmark via
`recall_labeling.py` at the 0.80 threshold: `[0.528 nano, 0.149 small, 0.083 medium, 0.240
large]` — the recall>=0.80 skew toward nano this project settled on in Week4 shows up directly
as load pressure on the cheapest queue. On both profiles, `fcfs_no_batch` goes unstable well
before the batching policies do (expected — no batching means no headroom against the
convex-cost queues), and `longest_queue`/`fcfs_batch` track each other closely while
`timeout_batch` trades some turnaround for materially lower compression ratio at the same load,
since its tau forces batches to run before they're full. Busy time and compression ratio didn't
always rank the four policies identically at a given load in this run (see each experiment's
`ranking_agreement.png`) — a first data point for the Week5 compute-metric decision above, not
yet enough runs across the full YOLO measured curves (still not measured) to call it settled.

Full config, per-policy per-load-level CSVs, and all plots (`turnaround_vs_busy_time.png`,
`*_vs_load.png` for turnaround/utilization/compression ratio, `ranking_agreement.png`) are in
`scheduler-sim/results/{yolo_synthetic_load_sweep,resnet_load_sweep}/`.

**Not done yet**: the `synthetic_4model` profile is exactly that — synthetic — so none of the
YOLO-side numbers above are a real hardware result, only a shakedown of the simulator itself.
Next real step per this week's meeting notes: measure actual YOLOv8 n/s/m/l batched-inference
curves (extends the existing `yolo-analysis/` benchmark to batched `T_i(b)` rather than
single-image latency) and rerun the sweep against real numbers before drawing any conclusion
from this simulator that isn't "the engine behaves correctly." Also still open from the meeting
notes: the literature review (not started), and validating the eventual winning policy
experimentally with PERTINENCE actually doing the routing.
