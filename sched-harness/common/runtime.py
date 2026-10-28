# The live part of a run, split over two processes so the scheduler is not slowed by (and does not slow) the
# job maker:
#   - ArrivalProcess (own process, no torch, pinned to its own core): sleeps to each planned arrival time and
#     sends the job through a pipe, stamped with time.monotonic().
#   - Runtime (main process): reads the pipe, keeps the four queues, asks the policy what to do and runs one
#     batch at a time on the GPU.
# time.monotonic() is one clock for the whole machine, so job stamps, batch stamps and the tegrastats Sampler
# share one time axis.
import collections
import multiprocessing
import os
import signal
import time

import config
from common import models
from common.policies import Serve, State, Wait

DONE = -1       # job_id of the last message from the arrival process: ("done", time)


class Job:
    def __init__(self, job_id, queue, pool_index, t_target, in_window):
        self.job_id = job_id
        self.queue = queue
        self.pool_index = pool_index
        self.t_target = t_target
        self.in_window = in_window
        self.t_enq = None
        self.lateness_ms = None
        self.t_start = None
        self.t_end = None
        self.batch_id = None
        self.batch_size = None
        self.dropped = 0


def pin_cores():
    """Give the arrival process the last online core and keep the main process off it. Returns (main cores,
    arrival core), or (None, None) when pinning is off or there are fewer than 3 cores."""
    cores = sorted(os.sched_getaffinity(0))
    if not config.PIN_CORES or len(cores) < 3:
        return None, None
    return cores[:-1], cores[-1]


def arrival_main(schedule, warmup_s, conn, t0_value, go_event, core):
    """Runs in the arrival process. Waits for the start signal, then sends every job at its planned time."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl+C is handled by the main process, which stops us
    if core is not None:
        os.sched_setaffinity(0, {core})
    go_event.wait()
    t0 = t0_value.value
    for job_id, (t_rel, q, pool_index) in enumerate(schedule):
        target = t0 + t_rel
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        t_enq = time.monotonic()
        conn.send((job_id, q, pool_index, target, t_enq, 1 if t_rel >= warmup_s else 0))
    conn.send((DONE, time.monotonic()))
    conn.close()


class ArrivalProcess:
    """Start it BEFORE any CUDA work (fork after CUDA initialisation is unsafe); the child never touches CUDA."""

    def __init__(self, schedule, warmup_s, core):
        ctx = multiprocessing.get_context("fork")
        self.conn, send_conn = ctx.Pipe(duplex=False)
        self.t0_value = ctx.Value("d", 0.0)
        self.go_event = ctx.Event()
        self.process = ctx.Process(target=arrival_main, args=(schedule, warmup_s, send_conn, self.t0_value,
                                                              self.go_event, core))
        self.process.daemon = True
        self.send_conn = send_conn

    def start(self):
        self.process.start()
        self.send_conn.close()          # the parent only reads

    def go(self, t0):
        """Release the arrival process: job k is created at t0 + schedule[k][0]."""
        self.t0_value.value = t0
        self.go_event.set()

    def stop(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=3)


class Runtime:
    def __init__(self, policy, nets, buffers, pool, service_ms, arrivals):
        self.policy = policy
        self.nets = nets
        self.buffers = buffers
        self.pool = pool
        self.service_ms = service_ms        # function (queue, b) -> ms
        self.arrivals = arrivals            # ArrivalProcess

        self.queues = [collections.deque() for _ in config.QUEUES]
        self.jobs = []
        self.batches = []                   # dicts
        self.waits = []                     # (t_start, t_end, reason)
        self.arrivals_done = False
        self.t_arrivals_done = None
        self.error = None
        self.overflow = 0
        self.drain_timeout = False
        self.decide_calls = 0
        self.decide_total_ms = 0.0

    # ---------- the pipe from the arrival process ----------

    def ingest(self):
        """Move every message waiting in the pipe into the queues. After the end marker the pipe is closed
        (it would read as ready forever), so we stop reading."""
        while not self.arrivals_done and self.arrivals.conn.poll(0):
            msg = self.arrivals.conn.recv()
            if msg[0] == DONE:
                self.arrivals_done = True
                self.t_arrivals_done = msg[1]
                continue
            job_id, q, pool_index, target, t_enq, in_window = msg
            job = Job(job_id, q, pool_index, target, in_window)
            job.t_enq = t_enq
            job.lateness_ms = (t_enq - target) * 1000.0
            if len(self.queues[q]) >= config.QUEUE_CAP:
                job.dropped = 1
                self.overflow += 1
            else:
                self.queues[q].append(job)
            self.jobs.append(job)

    def idle_wait(self, timeout, reason):
        """Sleep until a message arrives or the timeout passes (None = until a message arrives). Logged as idle."""
        t_start = time.monotonic()
        if self.arrivals_done:
            time.sleep(timeout)         # nothing more can arrive; the closed pipe would return at once
        elif timeout is None:
            while not self.arrivals.conn.poll(0.5):
                if not self.arrivals.process.is_alive() and not self.arrivals.conn.poll(0):
                    raise RuntimeError("the arrival process died before sending its end marker")
        else:
            self.arrivals.conn.poll(timeout)
        self.waits.append((t_start, time.monotonic(), reason))

    # ---------- the executor loop (main process, main thread) ----------

    def run(self):
        while True:
            self.ingest()
            if all(len(q) == 0 for q in self.queues):
                if self.arrivals_done:
                    return
                self.idle_wait(None, "empty")
                continue
            now = time.monotonic()
            if self.arrivals_done and now - self.t_arrivals_done > config.DRAIN_TIMEOUT_S:
                self.drain_timeout = True
                return
            state = State(now, [[j.t_enq for j in q] for q in self.queues], self.service_ms,
                          config.B_MAX, self.arrivals_done)
            t = time.perf_counter()
            decision = self.policy.decide(state)
            decide_ms = (time.perf_counter() - t) * 1000.0
            self.decide_calls += 1
            self.decide_total_ms += decide_ms
            drops = getattr(decision, "drops", None)
            if drops:
                for dq, count in drops.items():
                    for _ in range(min(count, len(self.queues[dq]))):
                        self.queues[dq].popleft().dropped = 2        # cancelled by the policy
                if all(len(q) == 0 for q in self.queues):
                    continue
            if isinstance(decision, Wait):
                if decision.until < now:
                    raise RuntimeError("policy %s returned Wait in the past" % self.policy.name)
                self.idle_wait(decision.until - now, "policy")
                continue
            if not isinstance(decision, Serve):
                raise RuntimeError("policy %s returned %r, expected Serve or Wait" % (self.policy.name, decision))
            q, b = decision.queue, decision.b
            if not 0 <= q < len(self.queues) or not 1 <= b <= min(len(self.queues[q]), config.B_MAX):
                raise RuntimeError("invalid decision from %s: queue %r, b %r, queue length %d"
                                   % (self.policy.name, q, b, len(self.queues[q]) if 0 <= q < len(self.queues) else -1))
            queue_len_before = len(self.queues[q])
            jobs = [self.queues[q].popleft() for _ in range(b)]
            self.run_batch(q, jobs, decide_ms, queue_len_before)

    def run_batch(self, q, jobs, decide_ms, queue_len_before):
        letter = config.QUEUES[q]
        arrays = [self.pool[j.pool_index] for j in jobs]
        t_start = time.monotonic()
        h2d_ms, forward_ms, nms_ms, n_boxes = models.run_batch(self.nets[letter], self.buffers[letter], arrays)
        t_end = time.monotonic()
        batch_id = len(self.batches)
        for j in jobs:
            j.t_start = t_start
            j.t_end = t_end
            j.batch_id = batch_id
            j.batch_size = len(jobs)
        self.batches.append({"batch_id": batch_id, "queue": q, "b": len(jobs), "t_start": t_start, "t_end": t_end,
                             "h2d_ms": h2d_ms, "forward_ms": forward_ms, "nms_ms": nms_ms, "n_boxes": n_boxes,
                             "decide_ms": decide_ms, "queue_len_before": queue_len_before})


def wait_cool(sampler):
    """Pause while thermal > 80 C until it is below 65 C (as in the sweep). Returns the start temperature."""
    sample = sampler.latest()
    temp = sample[1]["temps"].get("thermal") if sample else None
    if temp is None or temp <= config.THERMAL_PAUSE_ABOVE:
        return temp
    print("thermal %.1f C > %d: pausing until below %d C" % (temp, config.THERMAL_PAUSE_ABOVE, config.THERMAL_RESUME_BELOW))
    while sampler.latest()[1]["temps"].get("thermal", 0) > config.THERMAL_RESUME_BELOW:
        time.sleep(5)
    return temp


def board_temp(sampler):
    sample = sampler.latest()
    return sample[1]["temps"].get("thermal") if sample else None
