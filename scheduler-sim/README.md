# scheduler-sim

A small discrete-event simulator for the batching problem in `Journel/Week4.md` ("[DECISION] Formal
problem definition"). N models each have one FIFO queue. Jobs (video frames) arrive from K streams
and get routed to a queue. A single accelerator runs one batch at a time, from one queue only, and
never gets interrupted. Whenever it is free, a scheduler picks a queue and a batch size, or waits.
Running model i on b jobs takes T_i(b) ms, read from a CSV in `profiles/`. Self-contained: it
imports nothing from the rest of the repo.

## Files

    sim.py             Job, Queue, Profile, the Workload and Scheduler base classes, Simulator,
                       raw data save/load, compute_metrics
    workloads.py       Workload subclasses: Uniform, Weighted, Sticky, Periodic
    schedulers.py      Scheduler subclasses: FCFSNoBatch, FCFSBatch, LongestQueue, TimeoutBatch
    run_experiment.py  the load sweeps, results CSVs and plots
    profiles/          T_i(b) tables (resnet_example.csv is measured, synthetic_4model.csv is SYNTHETIC)
    tests/test_sim.py  pytest tests
    results/           written by run_experiment.py, one folder per experiment

## How to run

    pip install -r requirements.txt
    python -m pytest tests/ -q      # about 5 seconds
    python run_experiment.py        # about 1 minute, writes results/<experiment>/

Each experiment folder gets `results.csv` (one row per policy, load and seed), 5 plots
(`turnaround_vs_busy_time.png`, `turnaround_vs_load.png`, `utilization_vs_load.png`,
`compression_ratio_vs_load.png`, `ranking_agreement.png`) and a `raw/` folder with one `.npz`
file per run.

## Raw data and adding a metric

Every run saves its raw data to `results/<experiment>/raw/<policy>_load<L>_seed<S>.npz`, and
all metrics are computed from those files by `compute_metrics` in `sim.py`. To add or change a
metric without re-running any simulation:

1. Edit `compute_metrics` (and add the column to `CSV_COLUMNS` in `run_experiment.py`).
2. Set `SIMULATE = False` at the top of `run_experiment.py` and run it (about 10 seconds). It
   loads the saved files and rewrites every `results.csv` and plot.

`raw/` folders are git-ignored (about 50 MB in total). Runs are deterministic from their seed,
so `SIMULATE = True` regenerates them exactly.

Each file holds numpy arrays, loaded with `load_raw_data(path)` from `sim.py`:

| array | contents |
|---|---|
| `job_id`, `job_stream_id`, `job_queue_id` | one entry per job ever created |
| `job_status` | where the job ended up: 0 completed, 1 in service at the horizon, 2 still queued, 3 dropped |
| `job_arrival_time`, `job_start_time`, `job_finish_time` | ms; NaN when it never happened |
| `job_batch_size` | size of the batch the job ran in, 0 if it never started |
| `batch_start`, `batch_end`, `batch_size`, `batch_queue_id` | one entry per batch, in start order |
| `sample_time`, `sample_lengths` | queue-length snapshots; `sample_lengths[i][q]` is queue q at `sample_time[i]` |
| `model_names` | model of each queue, index = queue_id |
| `horizon_ms`, `sample_interval_ms` | run settings |

Quick look at one run from a Python prompt in `scheduler-sim/`:

```python
from sim import load_raw_data
raw = load_raw_data("results/resnet_load_sweep/raw/fcfs_batch_load0.20_seed1.npz")
done = raw["job_status"] == 0
turnaround = raw["job_finish_time"][done] - raw["job_arrival_time"][done]
print(turnaround.mean(), raw["batch_size"].mean())
```

## How it flows

                 pushes jobs                 reads queues,             runs one batch,
    Workload  ---------------->  Queues  <---------------  Scheduler  -------------> Accelerator
    (when + which queue)       (shared objects)          (which queue, how many)   (inside Simulator)

The same Queue objects are handed to both the Workload and the Scheduler when they are built. The
workload pushes jobs in, the scheduler looks at them, and the Simulator pops the chosen batch and
times it with the Profile. The Simulator jumps from event to event (arrival, batch done, wake-up,
queue sample) using a heap, so idle time costs nothing to simulate.

## The two knobs

**Knob 1, the Workload:** when jobs arrive and which queue gets them. The base class already does
Poisson arrivals and all the job bookkeeping, so a new workload usually only writes
`choose_queue`. Override `next_arrival_time` too if you want different arrival timing (see
`PeriodicWorkload`).

```python
from sim import Workload

class RoundRobinWorkload(Workload):
    """Poisson arrivals, jobs go to queues 0, 1, 2, ... in turn."""

    def __init__(self, queues, rng, num_streams, load_jobs_per_ms):
        Workload.__init__(self, queues, rng, num_streams, load_jobs_per_ms)
        self.next_queue = 0

    def choose_queue(self, stream_id):
        queue_id = self.next_queue
        self.next_queue = (self.next_queue + 1) % len(self.queues)
        return queue_id
```

**Knob 2, the Scheduler:** which queue and batch size to run next. Only `decide(now)` is needed.
Return `("run", queue_id, batch_size)`, `("wait", until_time)` or `None`. The Simulator checks the
answer and raises an error on anything invalid (empty queue, batch too big).

```python
from sim import Scheduler

class ShortestQueueScheduler(Scheduler):
    """Serve the shortest non-empty queue, as big a batch as it can fill."""

    def decide(self, now):
        best_id = None
        for queue_id in self.nonempty_queue_ids():
            if best_id is None or self.queues[queue_id].length() < self.queues[best_id].length():
                best_id = queue_id
        if best_id is None:
            return None
        return ("run", best_id, self.full_batch_size(best_id))
```

To use either one, add it to `POLICIES` or set an experiment's `workload_class` and
`workload_kwargs` at the top of `run_experiment.py`.

## Metrics

`compute_metrics` in `sim.py` summarises a run's raw data from t=0 to `HORIZON_MS`, using every
completed job and every batch started. Jobs still queued or mid-batch when time runs out count as not
completed.

| metric | how it is computed |
|---|---|
| `turnaround_*_ms` | finish time minus arrival time per completed job: mean, p50, p95, p99, max, and mean per queue |
| `wait_mean_ms` | start time minus arrival time, averaged over completed jobs |
| `busy_time_ms` | sum of batch durations |
| `compression_ratio` | batches divided by completed jobs (1.0 means no batching) |
| `throughput_jobs_per_ms` | completed jobs divided by the horizon |
| `utilization` | busy time divided by the horizon |
| `idle_time_ms` | horizon minus busy time |
| `max_queue_length` | largest single queue length seen in the queue-length samples (taken every `SAMPLE_INTERVAL_MS`) |
| `stable` | a straight line fitted to total queued jobs over time; False if it climbs by more than half the average total |
| `per_model` | per model: jobs served, number of batches, batch-size histogram, mean wait |

## Things to know

- Partial batches are padded: a batch of 5 on a model with engines for 4 and 8 pays the cost of 8.
- Every stream sends its first job at t=0, and the queues start empty, so the first moments look
  slightly better than steady state. That bias shrinks as `HORIZON_MS` grows.
- Overloaded runs have no steady state: their turnaround grows with the run length and only says
  "this policy cannot keep up".
- A batch still running at the horizon is counted at its full duration, so utilization can read
  slightly above 1.0 by at most one batch divided by the horizon.
- The `stable` flag is a rough heuristic. It can misfire at very light loads, where small wiggles
  look like growth, and near saturation, where the fill-up from the empty start looks like growth
  (see the TODO in `sim.py`).
- `profiles/synthetic_4model.csv` is made up. Do not draw conclusions from it about real YOLO
  timings.
