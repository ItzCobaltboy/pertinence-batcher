"""Core of the batching-scheduler simulator.

The problem: N models, one FIFO queue per model. Jobs (video frames) arrive
from K streams and a Workload decides which queue each job goes to. A single
accelerator runs one batch at a time, never interrupted. A batch holds jobs
from one queue only. Whenever the accelerator is free, a Scheduler picks a
queue and a batch size (or says "wait"). Running model i on a batch of b jobs
takes T_i(b) milliseconds, read from a Profile.

This file holds the shared pieces: Job, Queue, Profile, the two base classes
(Workload and Scheduler), the Simulator itself and compute_metrics.
"""
import csv
import heapq

import numpy as np

# Event kinds. The number is the tie-break priority when two events happen at
# the exact same time: a finishing batch is handled first (so the accelerator
# is free again), then arrivals, then wake-ups, then queue-length sampling.
# This fixed order is what keeps runs deterministic.
BATCH_DONE = 0
ARRIVAL = 1
WAKE = 2
SAMPLE = 3

DEFAULT_SAMPLE_INTERVAL_MS = 100.0

# Where each job ended up, as stored in the saved raw data (job_status column).
COMPLETED = 0
IN_SERVICE = 1
QUEUED = 2
DROPPED = 3

# Stability check settings, see is_stable() below.
MIN_SAMPLES_FOR_STABILITY = 4
MAX_RELATIVE_GROWTH = 0.5


class Job:
    """One video frame waiting to be processed by one model."""

    def __init__(self, job_id, stream_id, arrival_time, queue_id):
        self.job_id = job_id
        self.stream_id = stream_id
        self.arrival_time = arrival_time
        self.queue_id = queue_id
        # Filled in by the Simulator when the job is put into a batch / finishes.
        self.start_time = None
        self.finish_time = None
        self.batch_size = None


class Queue:
    """A FIFO queue of jobs bound to one model.

    The same Queue objects are shared by the Workload (which pushes jobs in)
    and the Scheduler (which looks at them to decide what to run next).
    """

    def __init__(self, queue_id, model_name, max_size=None):
        self.queue_id = queue_id
        self.model_name = model_name
        self.max_size = max_size  # None means unlimited
        self.jobs = []  # oldest job first

    def push(self, job):
        """Add a job at the back. Returns False (job dropped) if the queue is full."""
        if self.max_size is not None and len(self.jobs) >= self.max_size:
            return False
        self.jobs.append(job)
        return True

    def pop_batch(self, n):
        """Remove and return the n oldest jobs."""
        if n > len(self.jobs):
            raise ValueError(f"asked for {n} jobs but queue {self.queue_id} has {len(self.jobs)}")
        batch = self.jobs[:n]
        self.jobs = self.jobs[n:]
        return batch

    def length(self):
        return len(self.jobs)

    def oldest_arrival(self):
        """Arrival time of the job at the front, or None if the queue is empty."""
        if len(self.jobs) == 0:
            return None
        return self.jobs[0].arrival_time


class Profile:
    """The T_i(b) table: how many ms model i needs for a batch of size b.

    CSV format: lines starting with '#' are comments, the header is
    `model,1,4,8,16,32`, and an empty cell means that model has no engine for
    that batch size.
    """

    def __init__(self, csv_path):
        self.models = []  # model names in file order (queue i is bound to models[i])
        self.table = {}   # model -> {batch_size: runtime_ms}

        lines = []
        with open(csv_path, newline="") as csv_file:
            for line in csv_file:
                if line.strip() != "" and not line.lstrip().startswith("#"):
                    lines.append(line)
        rows = list(csv.reader(lines))
        batch_sizes = [int(cell) for cell in rows[0][1:]]  # header: model,1,4,8,...

        for row in rows[1:]:
            model = row[0].strip()
            runtimes = {}
            for batch_size, cell in zip(batch_sizes, row[1:]):
                if cell.strip() != "":  # empty cell: no engine for this batch size
                    runtimes[batch_size] = float(cell)
            if len(runtimes) == 0:
                raise ValueError(f"model {model} has no supported batch sizes")
            self.models.append(model)
            self.table[model] = runtimes

    def max_batch(self, model):
        """Largest batch size this model supports."""
        return max(self.table[model].keys())

    def runtime(self, model, batch_size):
        """Runtime in ms for a batch of batch_size jobs.

        If that exact size has no engine we "pad": run it on the smallest
        supported size that is at least batch_size, and pay that cost. This
        matches fixed-shape TensorRT engines.
        """
        runtimes = self.table[model]
        if batch_size in runtimes:
            return runtimes[batch_size]
        padded_size = None
        for size in sorted(runtimes.keys()):
            if size >= batch_size:
                padded_size = size
                break
        if padded_size is None:
            raise ValueError(f"batch size {batch_size} is above the max for model {model}")
        return runtimes[padded_size]


class Workload:
    """Base class for "how work arrives". A workload decides two things:

    1. WHEN the next job of a stream shows up  -> next_arrival_time()
    2. WHICH queue a new job goes to           -> choose_queue()

    A subclass MUST implement choose_queue. It MAY override next_arrival_time;
    the default is Poisson arrivals (every stream gets an equal share of
    load_jobs_per_ms). Everything else (creating jobs, pushing them into
    queues) is done here so subclasses stay tiny.
    """

    def __init__(self, queues, rng, num_streams, load_jobs_per_ms=None):
        self.queues = queues
        self.rng = rng  # numpy Generator owned by this workload, never global state
        self.num_streams = num_streams
        self.load_jobs_per_ms = load_jobs_per_ms
        self.jobs_created = 0
        self.next_job_id = 0

    def start_events(self):
        """First arrival of every stream: all streams start at time 0."""
        events = []
        for stream_id in range(self.num_streams):
            events.append((0.0, stream_id))
        return events

    def handle_arrival(self, now, stream_id):
        """Create a job, route it to a queue, push it. Returns (job, accepted)."""
        queue_id = self.choose_queue(stream_id)
        job = Job(self.next_job_id, stream_id, now, queue_id)
        self.next_job_id += 1
        self.jobs_created += 1
        accepted = self.queues[queue_id].push(job)
        return job, accepted

    def next_arrival_time(self, now, stream_id):
        """Default: Poisson, i.e. exponential gaps between a stream's jobs."""
        if self.load_jobs_per_ms is None or self.load_jobs_per_ms <= 0:
            raise ValueError("default Poisson arrivals need a positive load_jobs_per_ms")
        rate_per_stream = self.load_jobs_per_ms / self.num_streams
        return now + self.rng.exponential(1.0 / rate_per_stream)

    def choose_queue(self, stream_id):
        raise NotImplementedError("subclass must implement this")


class Scheduler:
    """Base class for "what to run next". Only called when the accelerator is free.

    decide(now) must return one of:
        ("run", queue_id, batch_size)
        ("wait", until_time)
        None   (all queues are empty, nothing to do)
    The helpers below are shared by the concrete schedulers.
    """

    def __init__(self, queues, profile):
        self.queues = queues
        self.profile = profile

    def decide(self, now):
        raise NotImplementedError("subclass must implement this")

    def nonempty_queue_ids(self):
        ids = []
        for queue in self.queues:
            if queue.length() > 0:
                ids.append(queue.queue_id)
        return ids

    def oldest_queue_id(self, queue_ids):
        """Among queue_ids, the queue whose front job arrived first.
        On a tie the lower queue_id wins (we only replace on strictly older)."""
        best_id = None
        for queue_id in queue_ids:
            arrival = self.queues[queue_id].oldest_arrival()
            if best_id is None or arrival < self.queues[best_id].oldest_arrival():
                best_id = queue_id
        return best_id

    def max_batch_for(self, queue_id):
        return self.profile.max_batch(self.queues[queue_id].model_name)

    def full_batch_size(self, queue_id):
        """Run as many jobs as the queue has, capped at the model's max batch size."""
        return min(self.queues[queue_id].length(), self.max_batch_for(queue_id))


class Simulator:
    """Discrete-event simulation: jump from event to event instead of ticking a clock."""

    def __init__(self, queues, workload, scheduler, profile, horizon_ms,
                 sample_interval_ms=DEFAULT_SAMPLE_INTERVAL_MS):
        self.queues = queues
        self.workload = workload
        self.scheduler = scheduler
        self.profile = profile
        self.horizon_ms = horizon_ms
        self.sample_interval_ms = sample_interval_ms

        self.events = []  # heap of (time, priority, sequence_number, kind, data)
        self.sequence_number = 0  # breaks remaining ties by insertion order
        self.now = 0.0
        self.accelerator_busy = False
        self.completed_jobs = []
        self.dropped_jobs = []
        self.in_service = []  # jobs in the batch currently running
        self.batches = []  # (start, end, model, batch_size, queue_id)
        self.queue_samples = []  # (time, [length of each queue])

    def push_event(self, time, kind, data=None):
        # kind doubles as the priority, see the constants at the top of the file
        heapq.heappush(self.events, (time, kind, self.sequence_number, kind, data))
        self.sequence_number += 1

    def run(self):
        """Run until horizon_ms and return the metrics dict."""
        for start_time, stream_id in self.workload.start_events():
            self.push_event(start_time, ARRIVAL, stream_id)
        self.push_event(0.0, SAMPLE)

        while len(self.events) > 0:
            time, priority, sequence_number, kind, data = heapq.heappop(self.events)
            if time > self.horizon_ms:
                self.now = self.horizon_ms
                break
            self.now = time

            if kind == ARRIVAL:
                self.handle_arrival(data)
            elif kind == BATCH_DONE:
                self.handle_batch_done(data)
            elif kind == WAKE:
                pass  # only exists to trigger the dispatch step below
            elif kind == SAMPLE:
                self.handle_sample()

            if kind != SAMPLE and not self.accelerator_busy:
                self.try_to_dispatch()

        self.raw_data = collect_raw_data(self)
        return compute_metrics(self.raw_data)

    def handle_arrival(self, stream_id):
        job, accepted = self.workload.handle_arrival(self.now, stream_id)
        if not accepted:
            self.dropped_jobs.append(job)
        next_time = self.workload.next_arrival_time(self.now, stream_id)
        if next_time is not None:
            self.push_event(next_time, ARRIVAL, stream_id)

    def handle_batch_done(self, batch_jobs):
        self.accelerator_busy = False
        for job in batch_jobs:
            job.finish_time = self.now
            self.completed_jobs.append(job)
            self.in_service.remove(job)

    def handle_sample(self):
        lengths = []
        for queue in self.queues:
            lengths.append(queue.length())
        self.queue_samples.append((self.now, lengths))
        self.push_event(self.now + self.sample_interval_ms, SAMPLE)

    def try_to_dispatch(self):
        """Ask the scheduler what to do, check the answer, and start a batch."""
        action = self.scheduler.decide(self.now)
        if action is None:
            return
        if action[0] == "wait":
            if action[1] > self.now:
                self.push_event(action[1], WAKE)
            return
        if action[0] != "run":
            raise ValueError(f"unknown scheduler action: {action}")

        queue_id = action[1]
        batch_size = action[2]
        queue = self.queues[queue_id]
        model = queue.model_name
        # Fail loudly on a bad decision, never silently fix it.
        if queue.length() == 0:
            raise ValueError(f"scheduler asked to run empty queue {queue_id}")
        if batch_size < 1 or batch_size > queue.length():
            raise ValueError(f"batch size {batch_size} not between 1 and queue length {queue.length()}")
        if batch_size > self.profile.max_batch(model):
            raise ValueError(f"batch size {batch_size} is above the max for {model}")

        duration = self.profile.runtime(model, batch_size)
        batch_jobs = queue.pop_batch(batch_size)
        for job in batch_jobs:
            job.start_time = self.now
            job.batch_size = batch_size
            self.in_service.append(job)
        end_time = self.now + duration
        self.batches.append((self.now, end_time, model, batch_size, queue_id))
        self.accelerator_busy = True
        self.push_event(end_time, BATCH_DONE, batch_jobs)


def collect_raw_data(sim):
    """Everything that happened in a finished run, as plain numpy arrays.

    Saved to disk by run_experiment.py, so any metric can be computed later
    from the file without running the simulation again. Every job ever created
    is included, with a status saying where it ended up. Times that never
    happened (e.g. finish_time of a queued job) are NaN.
    """
    jobs_and_status = []
    for job in sim.completed_jobs:
        jobs_and_status.append((job, COMPLETED))
    for job in sim.in_service:
        jobs_and_status.append((job, IN_SERVICE))
    for queue in sim.queues:
        for job in queue.jobs:
            jobs_and_status.append((job, QUEUED))
    for job in sim.dropped_jobs:
        jobs_and_status.append((job, DROPPED))

    columns = {"job_id": [], "job_stream_id": [], "job_queue_id": [], "job_status": [],
               "job_arrival_time": [], "job_start_time": [], "job_finish_time": [],
               "job_batch_size": []}
    for job, status in jobs_and_status:
        columns["job_id"].append(job.job_id)
        columns["job_stream_id"].append(job.stream_id)
        columns["job_queue_id"].append(job.queue_id)
        columns["job_status"].append(status)
        columns["job_arrival_time"].append(job.arrival_time)
        columns["job_start_time"].append(nan_if_none(job.start_time))
        columns["job_finish_time"].append(nan_if_none(job.finish_time))
        columns["job_batch_size"].append(0 if job.batch_size is None else job.batch_size)

    raw = {}
    for name in columns:
        raw[name] = np.array(columns[name])
    raw["job_status"] = raw["job_status"].astype(int)

    # one entry per batch run on the accelerator, in the order they started
    raw["batch_start"] = np.array([batch[0] for batch in sim.batches], dtype=float)
    raw["batch_end"] = np.array([batch[1] for batch in sim.batches], dtype=float)
    raw["batch_size"] = np.array([batch[3] for batch in sim.batches], dtype=int)
    raw["batch_queue_id"] = np.array([batch[4] for batch in sim.batches], dtype=int)

    # queue-length snapshots: sample_lengths[i][q] = length of queue q at sample_time[i]
    raw["sample_time"] = np.array([sample[0] for sample in sim.queue_samples], dtype=float)
    raw["sample_lengths"] = np.array([sample[1] for sample in sim.queue_samples], dtype=int)

    raw["model_names"] = np.array([queue.model_name for queue in sim.queues])  # index = queue_id
    raw["horizon_ms"] = np.array(sim.horizon_ms, dtype=float)
    raw["sample_interval_ms"] = np.array(sim.sample_interval_ms, dtype=float)
    return raw


def nan_if_none(value):
    if value is None:
        return float("nan")
    return value


def save_raw_data(raw, path):
    """Write the raw data of one run to a compressed .npz file."""
    np.savez_compressed(path, **raw)


def load_raw_data(path):
    """Read a file written by save_raw_data back into a dict of numpy arrays."""
    raw = {}
    data = np.load(path)
    for name in data.files:
        raw[name] = data[name]
    data.close()
    return raw


def mean_or_none(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def percentile_or_none(values, p):
    if len(values) == 0:
        return None
    return float(np.percentile(values, p))


def is_stable(sample_times, sample_lengths):
    """Rough check that the queues are not growing without bound.

    Fit a straight line to (time, total jobs queued) over the whole run. If
    the line would add more than half of the average occupancy over the run,
    call it unstable.
    TODO: this is rough and needs tightening. At very light loads the mean
    occupancy is tiny, so small wiggles can look like growth. Near saturation
    (about 97% utilization) the fill-up from the empty start can look like
    growth too.
    """
    if len(sample_times) < MIN_SAMPLES_FOR_STABILITY:
        return True
    totals = []
    for lengths in sample_lengths:
        totals.append(int(np.sum(lengths)))
    slope, intercept = np.polyfit(sample_times, np.array(totals, dtype=float), 1)
    mean_total = float(np.mean(totals))
    if mean_total <= 0:
        mean_total = 1.0
    window = max(float(sample_times[-1] - sample_times[0]), 1e-9)
    return bool(slope * window / mean_total < MAX_RELATIVE_GROWTH)


def compute_metrics(raw):
    """Summarise one run from its raw data (see collect_raw_data), over the
    whole window from t=0 to the horizon. The empty start makes the first
    moments look slightly too good, and that bias shrinks as the run gets
    longer. Jobs still queued or mid-batch at the horizon are not completed
    and are left out."""
    measured_time = max(float(raw["horizon_ms"]), 1e-9)
    model_names = raw["model_names"]

    turnarounds = []
    waits = []
    turnarounds_per_queue = {}  # queue_id -> list of turnaround times
    waits_per_queue = {}        # queue_id -> list of wait times
    num_dropped = 0
    for i in range(len(raw["job_id"])):
        if raw["job_status"][i] == DROPPED:
            num_dropped += 1
        if raw["job_status"][i] != COMPLETED:
            continue
        queue_id = int(raw["job_queue_id"][i])
        arrival = float(raw["job_arrival_time"][i])
        turnarounds.append(float(raw["job_finish_time"][i]) - arrival)
        waits.append(float(raw["job_start_time"][i]) - arrival)
        turnarounds_per_queue.setdefault(queue_id, []).append(turnarounds[-1])
        waits_per_queue.setdefault(queue_id, []).append(waits[-1])

    # Only models that actually ran a batch get an entry.
    busy_time = 0.0
    per_model = {}
    for i in range(len(raw["batch_start"])):
        busy_time += float(raw["batch_end"][i]) - float(raw["batch_start"][i])
        queue_id = int(raw["batch_queue_id"][i])
        batch_size = int(raw["batch_size"][i])
        model = str(model_names[queue_id])
        if model not in per_model:
            per_model[model] = {"jobs_served": 0, "num_batches": 0, "batch_size_histogram": {},
                                "mean_wait_ms": mean_or_none(waits_per_queue.get(queue_id, []))}
        stats = per_model[model]
        stats["jobs_served"] += batch_size
        stats["num_batches"] += 1
        histogram = stats["batch_size_histogram"]
        histogram[batch_size] = histogram.get(batch_size, 0) + 1

    per_queue_mean = {}
    for queue_id in turnarounds_per_queue:
        per_queue_mean[queue_id] = mean_or_none(turnarounds_per_queue[queue_id])
    max_queue_length = 0
    if raw["sample_lengths"].size > 0:
        max_queue_length = int(np.max(raw["sample_lengths"]))
    num_completed = len(turnarounds)
    num_batches = len(raw["batch_start"])
    compression_ratio = None
    if num_completed > 0:
        compression_ratio = num_batches / num_completed

    return {
        "num_jobs_completed": num_completed,
        "num_jobs_dropped": num_dropped,
        "num_batches": num_batches,
        "turnaround_mean_ms": mean_or_none(turnarounds),
        "turnaround_p50_ms": percentile_or_none(turnarounds, 50),
        "turnaround_p95_ms": percentile_or_none(turnarounds, 95),
        "turnaround_p99_ms": percentile_or_none(turnarounds, 99),
        "turnaround_max_ms": percentile_or_none(turnarounds, 100),
        "turnaround_per_queue_mean_ms": per_queue_mean,
        "wait_mean_ms": mean_or_none(waits),
        "busy_time_ms": busy_time,
        "compression_ratio": compression_ratio,
        "throughput_jobs_per_ms": num_completed / measured_time,
        "utilization": busy_time / measured_time,
        "idle_time_ms": max(measured_time - busy_time, 0.0),
        "max_queue_length": max_queue_length,
        "stable": is_stable(raw["sample_time"], raw["sample_lengths"]),
        "per_model": per_model,
    }
