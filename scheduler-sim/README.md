# scheduler-sim

A small discrete-event simulator for the batching problem in `Journel/Week4.md` ("[DECISION] Formal
problem definition"). N models each have one FIFO queue. Jobs (video frames) arrive from K streams
and get routed to a queue. A single accelerator runs one batch at a time, from one queue only, and
never gets interrupted. Whenever it is free, a scheduler picks a queue and a batch size, or waits.
Running model i on b jobs takes T_i(b) ms, read from a CSV in `profiles/`. Self-contained: it
imports nothing from the rest of the repo.

## Files

    sim.py             Job, Queue, Profile, the Workload and Scheduler base classes, Simulator, compute_metrics
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

Each experiment folder gets `results.csv` (one row per policy, load and seed) and 5 plots:
`turnaround_vs_busy_time.png`, `turnaround_vs_load.png`, `utilization_vs_load.png`,
`compression_ratio_vs_load.png` and `ranking_agreement.png`.

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

## Things to know

- Partial batches are padded: a batch of 5 on a model with engines for 4 and 8 pays the cost of 8.
- There is no warm-up: metrics cover the whole run from t=0. The empty start makes the first
  moments look slightly too good, and that bias shrinks as `HORIZON_MS` grows. Jobs still queued
  or mid-batch when time runs out are not counted as completed.
- The `stable` flag is a rough heuristic and wrongly says False on some very light loads (see
  the TODO in `sim.py`).
- `profiles/synthetic_4model.csv` is made up. Do not draw conclusions from it about real YOLO
  timings.
