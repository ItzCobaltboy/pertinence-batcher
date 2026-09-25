# scheduler-sim

A self-contained discrete-event simulator for the scheduling problem defined in
`Journel/Week4.md`'s "[DECISION] Formal problem definition" entry: N models, one FIFO queue
per model, jobs arriving from K streams and routed to a queue, a single accelerator that runs
one batch from one queue at a time, non-preemptively, and a scheduling policy that decides
which queue and batch size to run whenever the accelerator is free.

Built per `Journel/Week5.md`'s meeting decision: the scheduling problem is decoupled from
PERTINENCE (the scheduler only sees queues, batch sizes and runtimes, not how frames got
routed), so it's designed and validated here in simulation first.

**Standalone package.** `scheduler-sim/` does not import anything from the rest of the
`pertinence-batcher` monorepo. It has its own `requirements.txt` and reads external data (the
resnet batch-size sweep, the YOLO/COCO recall benchmark) only as CSV files passed in through
config, never as python imports. It can be copied out of this repo and still work.

## Install

```
pip install -r scheduler-sim/requirements.txt
```

## Quickstart

```
cd scheduler-sim
python -m pytest tests/ -q          # 29 tests, correctness first
python profiles/build_profiles.py   # (re)generates profiles/*.csv from repo data
python experiments/run_experiment.py
```

The experiment runner writes `results/<experiment>/results.csv` (one row per run) plus plots
to `results/<experiment>/*.png`.

## Architecture

Every component is swappable through a small name-based registry (`scheduler_sim/registry.py`)
so a config can pick an implementation by string, without touching the engine.

```
scheduler_sim/
  job.py              Job dataclass (arrival/start/finish times, optional deadline/priority)
  runtime_profile.py  RuntimeProfile: the T_i(b) matrix, loaded from CSV, pad/interpolate modes
  workload.py         Workload/Stream + ArrivalProcess (periodic, poisson)
  router.py           JobRouter ABC + registry
  routers/            uniform_random, weighted_random, trace, sticky, recall_labeling (helper)
  queueing.py          QueueSet (N FIFO queues) + DropPolicy ABC (v1: drop_arriving)
  policy.py            SchedulingPolicy ABC + Action types (RunBatch/Wait) + action validation
  policies/             fcfs_no_batch, fcfs_batch, longest_queue, timeout_batch
  accelerator.py        single-server accelerator, tracks busy intervals + current model
  engine.py             the DES engine: heapq event queue (ARRIVAL/BATCH_DONE/TIMER/QUEUE_SAMPLE)
  metrics.py            MetricsCollector: turnaround, compute, throughput, per-model, stability
  logging_setup.py      logging levels + optional JSONL event trace (off by default)
tests/                  pytest suite (29 tests) -- run before any policy/router work relies on it
profiles/                build_profiles.py + the two shipped RuntimeProfile CSVs
experiments/             run_experiment.py -- hardcoded sweep configs, tidy CSV + 3 plot types
data/                    tiny example inputs (e.g. trace_router_example.csv)
results/                 experiment outputs (gitignored except for a .gitkeep)
```

### The DES engine

`scheduler_sim/engine.py`'s `Simulator` runs a `heapq`-based event queue with four event kinds:
`ARRIVAL`, `BATCH_DONE`, `TIMER`, and an internal `QUEUE_SAMPLE` (for the queue-length-over-time
metric). Ties at the same simulated time are broken by a fixed type priority
(`BATCH_DONE < ARRIVAL < TIMER < QUEUE_SAMPLE`) then insertion order, so a run is fully
deterministic given its seed. All randomness (arrival jitter, Poisson draws, router randomness)
goes through an explicitly-owned `numpy.random.Generator` seeded from `SimConfig.seed` --
nothing touches python's global `random` module.

Whenever the accelerator goes idle, the engine builds a read-only `PolicyView` (current time,
per-queue length + oldest arrival time, the `RuntimeProfile`, whether the accelerator is free)
and calls the policy's `decide(view)`. The returned `Action` (`RunBatch(queue_id, batch_size)`
or `Wait(until_time)`) is validated (`policy.validate_action`) before being acted on --
an invalid action (empty queue, unsupported batch size, batch larger than the queue, etc.)
raises `InvalidAction` immediately rather than being silently coerced.

### RuntimeProfile and partial batches

`RuntimeProfile.from_csv` loads a `model,<batch sizes...>` matrix; an empty cell means that
model doesn't support that batch size. `runtime(model, b)` for an unsupported `b` falls back to
`partial_batch_mode`:

- `"pad"` (default): run `b` jobs at the cost of the next larger supported size, matching a
  fixed-shape TensorRT engine that pads a partial batch up.
- `"interpolate"`: linearly interpolate between the two bracketing supported sizes (flat
  extrapolation past the largest measured point).

An optional `switch_cost_ms` is added whenever the accelerator's model changes between
consecutive batches (default 0).

## Adding a new router or policy

Both are one file plus a `@registry.register("name")` decorator -- nothing else needs to
change. Example router:

```python
# scheduler_sim/routers/round_robin.py
from ..job import Job
from ..router import JobRouter, router_registry

@router_registry.register("round_robin")
class RoundRobinRouter(JobRouter):
    def __init__(self, num_queues: int):
        self.num_queues = num_queues
        self._next = 0

    def route(self, job: Job, now: float) -> int:
        q = self._next
        self._next = (self._next + 1) % self.num_queues
        return q
```

Then import it once (e.g. add it to `scheduler_sim/routers/__init__.py`'s import list) so the
decorator runs and registers it, and pick it anywhere by name:
`router_registry.create("round_robin", num_queues=4)`.

A new `SchedulingPolicy` follows the same shape: subclass `SchedulingPolicy`, implement
`decide(view) -> RunBatch | Wait`, register it in `scheduler_sim/policies/__init__.py`, and use
`policy_registry.create("your_policy_name", **kwargs)`.

## Running an experiment

`experiments/run_experiment.py` hardcodes its sweep definitions at the top of the file (per the
project's convention: RuntimeProfile CSVs are always file-loaded/config-driven, but the sweep
*definitions* -- which loads, which policies, which seeds -- live in python, not YAML). It
sweeps policy x offered load x seed for two example configs:

- **`yolo_synthetic_load_sweep`**: the `synthetic_4model` profile, with routing weights derived
  live from `yolo-analysis/results/val2017/coco_class_recall_benchmark.csv` via
  `routers/recall_labeling.py` (cheapest model with recall >= 0.80, falling back to weights
  `[0.25]*4` if that CSV isn't present).
- **`resnet_load_sweep`**: the `resnet_example` profile (real resnet18/resnet50 fp32 batch-size
  sweep), 50/50 routing (no real routing signal exists for this pair).

```
python experiments/run_experiment.py
```

writes, per experiment, to `results/<experiment>/`:
- `results.csv` -- one row per (policy, load, seed) run: turnaround (mean/p50/p95/p99/max),
  busy time, compression ratio, throughput, utilization, max queue length, stability, drops.
- `turnaround_vs_busy_time.png` -- mean turnaround vs. busy time, per policy.
- `turnaround_vs_load.png`, `utilization_vs_load.png`, `compression_ratio_vs_load.png` -- each
  metric vs. load, per policy, with seed-to-seed error bars.
- `ranking_agreement.png` -- at each load level, whether ranking policies by busy time and by
  compression ratio gives the same order (the Week5 open question on which compute metric
  should be primary).

To run your own sweep, edit `POLICY_SPECS`, `SEEDS`, and the `YOLO_EXPERIMENT`/
`RESNET_EXPERIMENT` dicts at the top of `experiments/run_experiment.py`, or add a new spec dict
and pass it to `run_experiment()`.

## Example profiles

Regenerate both from `profiles/build_profiles.py` (reads repo data, writes into `profiles/`):

- **`resnet_example.csv`**: real 2-model matrix (resnet18, resnet50), fp32, batch sizes
  1/4/8/16/32, read straight from
  `archive/quantization_experiments/results/exp2_batch_size_sweep.csv`'s `latency_ms_total`
  column.
- **`synthetic_4model.csv`**: a clearly-labeled **SYNTHETIC** 4-model matrix standing in for
  YOLOv8 n/s/m/l until real batched-inference curves are measured -- small models (n/s) kept
  roughly affine, larger ones (m/l) made convex, matching the affine-vs-convex split the real
  resnet18/resnet50 curves show in `Journel/Week4.md`. The CSV's leading `#` comment lines say
  so explicitly; `RuntimeProfile.from_csv` skips `#`-prefixed lines.

## Deriving router probabilities/traces from real YOLO/COCO data

`scheduler_sim/routers/recall_labeling.py` recomputes, from the long-format
`yolo-analysis/results/<split>/coco_class_recall_benchmark.csv` (columns: `image_id, model,
recall, ...`), a per-image label = the cheapest model (default order yolov8n/s/m/l) with
`recall >= threshold` (default 0.80, the correctness definition from `Journel/Week4.md`'s
"[DECISION] Correctness definition for the YOLO pool" entry); if no model clears the threshold
for an image, it falls back to the largest model. It reads the raw per-(image, model) CSV and
takes `threshold` as a parameter on every call -- it deliberately does **not** read the repo's
`*_minimum_match.csv` files, which get overwritten by whichever threshold was last run there.

```python
from scheduler_sim.routers.recall_labeling import (
    label_images_from_recall_csv, label_counts_to_weights, labels_to_trace,
)

labels = label_images_from_recall_csv("yolo-analysis/results/val2017/coco_class_recall_benchmark.csv",
                                       threshold=0.80)
weights = label_counts_to_weights(labels)   # -> feed WeightedRandomRouter/StickyRouter
trace = labels_to_trace(labels)             # -> feed TraceRouter
```

`TraceRouter` itself expects a single-column `queue_id` CSV (one row per arrival, in the global
order jobs are routed in) -- see `data/trace_router_example.csv` for a runnable dummy trace and
`scheduler_sim/routers/trace.py`'s docstring for the exact schema.

## Tests

```
python -m pytest tests/ -q
```

29 tests, all passing at the time of writing, covering (per the "tests first" ordering this
package was built in):

- deterministic periodic arrivals, one queue, batch size 1, no overload -> turnaround equals
  service time exactly (`tests/test_periodic_arrivals.py`)
- Poisson arrivals, one queue, fixed service time, batch size 1 -> mean wait matches the M/D/1
  formula within sampling error (`tests/test_poisson_md1.py`)
- conservation: every arrived job completes, is dropped, is still queued, or is in-flight at the
  end; busy time never exceeds sim time (beyond one in-flight batch's slack); no job served
  twice (`tests/test_conservation.py`)
- same seed gives identical results, different seeds diverge (`tests/test_determinism.py`)
- `RuntimeProfile` padding/interpolation logic and switch cost (`tests/test_runtime_profile.py`)
- invalid-action rejection -- empty queue, oversized batch, busy accelerator, negative `Wait`,
  wrong return type, and a full engine run with a deliberately broken policy
  (`tests/test_invalid_actions.py`)
- router label derivation on a tiny hand-made recall CSV, including the threshold and
  weights/trace helpers (`tests/test_recall_labeling.py`)
- router behavior (weighted/trace/sticky) in isolation (`tests/test_routers.py`)

## Out of scope (v1)

No custom/novel scheduling policy, no PERTINENCE integration, no GPU or real model timing, no
web UI -- see the spec this was built from (`Journel/Week5.md`'s `[SETUP]` entry for this
package). `Job.deadline`/`Job.priority` and `MetricsCollector`'s deadline-miss-rate exist in the
schema already, unused by any v1 policy, so a deadline- or priority-aware extension (flagged as
a later cycle in `Journel/Week5.md`'s meeting notes) doesn't need a data-model rewrite.
