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

---

## Idea (open, not yet decided): size-aware parallel scheduling via MPS

While winding down theory work, a side conversation surfaced a possible way to break the
"one batch at a time" assumption the formal problem and `scheduler-sim` both currently make.

**The idea.** CUDA MPS lets multiple processes' kernels actually run concurrently on the GPU,
not just get queued one after another. Small-batch ML inference is typically
memory-bandwidth-bound rather than compute-bound: the compute cores sit partly idle waiting for
weights/activations to stream in, which is exactly why batching helps in the first place (it
amortizes one memory load over more compute). If two small enough models both fit comfortably
inside the GPU's memory bandwidth budget, running their batches concurrently via MPS should fill
that idle compute with little to no slowdown, rather than serializing them.

If this holds, the accelerator doesn't strictly have to run one queue's batch at a time — a
second, small-enough batch from a different queue could run alongside it. The scheduler's job
would then include a **size-aware parallel scheduling** decision: not just (queue, batch size),
but whether two candidate batches are cheap enough (bandwidth-wise) to safely co-run, or must be
serialized.

**Why it's not free.** Checked against the current bibliography: nobody has cleanly
characterized this. BCEdge learns an interference predictor (RL-based, not a formula) for
concurrent execution; SEEB-GPU sidesteps the question entirely by physically partitioning GPU
compute units per model (TPC masking) rather than truly sharing them. So the actual
slowdown-vs-concurrency curve for small edge models is an open, unmeasured question, not
something we can just plug in.

**What's needed before this goes anywhere:** understand how real runtime behaves under MPS —
does running two of our models concurrently actually cost close to nothing when neither is
memory-bandwidth-bound, or does contention (kernel launch overhead, SM occupancy fragmentation)
eat the theoretical benefit? That has to be measured on the actual target hardware before it's
anything more than a hypothesis. Also a practical blocker for now: MPS on Jetson is only
supported since JetPack 6.1 / CUDA 12.5, and has reported issues inside containers/Kubernetes —
needs checking against whatever JetPack version is actually in use.

Not pursuing this experimentally yet — flagged here so it isn't lost, and to keep the formal
problem's "single accelerator, one batch at a time, non-preemptive" assumption honest: it's a
simplifying assumption, not a proven hardware constraint.

---

## Idea (open, not yet decided): what "building a policy" actually means for this scheduler

Not a decision yet, just writing down where my head's at on how the actual scheduling policy
gets built, before I go implement anything.

**The shape of it, as I currently understand it**: take the current state (queue lengths, wait
times, T_i(b) for each model, whatever else is relevant), pass it through some function, get a
score back for each option I could take right now. Options being: serve queue i at some batch
size, for each queue, or wait. Whichever option scores highest, do that. This is apparently
called a greedy/myopic index policy (cμ-rule, CAW index are the classic examples) — I'm not
predicting the whole future, just scoring "how good does each choice look right now" and taking
the best one, every time the accelerator frees up.

**The catch**: "wait" can't just be scored as zero or ignored. Whether waiting is actually good
depends on whether more jobs are about to show up to fill a bigger batch — and that's not
something I know for sure, it's probabilistic. My arrivals overall are periodic (a frame shows up
every Δ), but which queue a given frame lands in depends on the dispatcher reading the frame
content, so from any one queue's point of view it's not periodic at all, it's a periodic tick that
gets randomly thinned by routing. So I can't just say "a job arrives every Δ for this queue," I
need to estimate, per queue, how likely a job is to land there.

**Current plan for the "how likely"**: don't assume a textbook distribution (Poisson etc, which
is what the OR papers assume and which doesn't actually match how my arrivals work) — instead
track a sliding window of the last X arrivals and estimate the per-queue landing probability
empirically from that, then use that estimate to calculate what waiting is actually worth before
comparing it against just serving now. Window size X is itself something to tune later (too short
= noisy, too long = can't react to routing shifts, e.g. a scene change dumping way more frames on
one model) — not solving that now, just flagging it.

Net: policy = state -> score function (including a stochastic wait-value term estimated from
recent history) -> pick the max. That's the target shape for the actual scheduling policy once
literature search and real T_i(b) measurements are done.

---

## [SETUP] scheduler-sim rewritten as plain student-style OOP, same results within noise

The first version of `scheduler-sim` worked but I couldn't read it top to bottom: registries,
ABCs, frozen view dataclasses, decorators, import-for-side-effect registration, a JSONL tracer and
hooks nothing used. I rebuilt it in place as three plain files I can actually follow:

- `sim.py`: `Job`, `Queue`, `Profile` (the T_i(b) table), the two base classes `Workload` and
  `Scheduler`, the `Simulator` (one heap, one while loop) and `compute_metrics`.
- `workloads.py`: `UniformWorkload`, `WeightedWorkload`, `StickyWorkload`, `PeriodicWorkload`.
- `schedulers.py`: the same 4 policies as before (`fcfs_no_batch`, `fcfs_batch`,
  `longest_queue`, `timeout_batch` with tau=15ms), same tie-breaks.

The design is now built around the two knobs I actually want to experiment with. A Workload
decides when a job arrives and which queue it goes to (a subclass usually only writes
`choose_queue`), and a Scheduler decides which queue and batch size run next (only `decide`). The
same `Queue` objects are handed to both, so there are no copied view objects in between. The
Simulator still checks every scheduler decision and raises on an invalid one. Metrics keep the
same keys and the same `results.csv` columns, and the `stable` heuristic is unchanged (it still
wrongly says unstable on some very light loads, left as a TODO).

**Removed**: the registries, ABCs, policy hooks, JSONL event tracer, deadline/priority fields,
the trace router and `recall_labeling.py`, interpolate mode, model switch cost, and the live
read of the recall CSV. The YOLO routing weights are now hardcoded from one run of the old code
at recall >= 0.80 on val2017: `[0.5278, 0.149, 0.0834, 0.2398]` for n/s/m/l (2639 / 745 / 417 /
1199 of 5000 images).

**Regression check against the old implementation**: RNG usage changed (one generator now drives
both arrivals and routing), so numbers can't match bit for bit. On `yolo_synthetic_load_sweep`
every seed-averaged turnaround, utilization and compression ratio is within 7% of the old value
and the policy ranking is identical at every load. On `resnet_load_sweep` loads 0.05 to 0.20 are
within 6% with identical rankings. At 0.25 and 0.30 a few cells moved 10 to 55% and the 3-seed
ranking flipped, but those loads sit at 97 to 100% utilization, where the old code's own
seed-to-seed spread was already about 40%. I reran both old and new code on 20 fresh seeds for
those cells: they agree within 1 to 2 standard errors everywhere and rank the policies the same
way, so the flips were seed noise, not a behaviour change.

**New experiment** `yolo_sticky_load_sweep` (not part of the regression check): same YOLO setup
but `StickyWorkload` with stay_probability=0.9, so each stream keeps hitting the same queue like
real video. `longest_queue` still wins at every load and everything gets 5 to 15% slower. The
one ranking change is at load 0.25, where `fcfs_no_batch` drops to last (21.2ms against 9.9ms
without stickiness): bursts hurt the policy that can't batch the most.
