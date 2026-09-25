"""The discrete-event simulator engine: a heapq-based event queue with
ARRIVAL, BATCH_DONE and TIMER (+ an internal QUEUE_SAMPLE) events,
deterministic tie-breaking, and an explicitly seeded RNG (never global
random state).
"""
from __future__ import annotations

import heapq
import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .accelerator import Accelerator
from .job import Job, next_job_id
from .logging_setup import EventTraceWriter, get_logger
from .metrics import MetricsCollector
from .policy import PolicyView, QueueView, RunBatch, SchedulingPolicy, Wait, validate_action
from .queueing import QueueSet
from .router import JobRouter
from .runtime_profile import RuntimeProfile
from .workload import Workload

ARRIVAL = "ARRIVAL"
BATCH_DONE = "BATCH_DONE"
TIMER = "TIMER"
QUEUE_SAMPLE = "QUEUE_SAMPLE"

# Fixed priority so that, at equal simulated time, event processing
# order is deterministic regardless of insertion order: batch
# completions before arrivals before timers before sampling.
_TYPE_PRIORITY = {BATCH_DONE: 0, ARRIVAL: 1, TIMER: 2, QUEUE_SAMPLE: 3}


@dataclass(order=True)
class _Event:
    time: float
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


@dataclass
class SimConfig:
    queue_to_model: List[str]  # queue_id -> model name, one queue per model
    runtime_profile: RuntimeProfile
    workload: Workload
    router: JobRouter
    policy: SchedulingPolicy
    seed: int
    horizon_ms: Optional[float] = None
    max_jobs: Optional[int] = None
    warmup_ms: float = 0.0
    max_queue_size: Optional[int] = None
    queue_sample_interval_ms: float = 50.0
    event_trace_path: Optional[str] = None
    log_level: str = "WARNING"


@dataclass
class SimResult:
    metrics_summary: dict
    completed_jobs: List[Job]
    dropped_jobs: List[Job]
    end_time: float
    in_flight_jobs: List[Job] = field(default_factory=list)
    queued_jobs: List[Job] = field(default_factory=list)


class Simulator:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.queue_set = QueueSet(
            num_queues=len(config.queue_to_model), max_queue_size=config.max_queue_size
        )
        self.accelerator = Accelerator()
        self.metrics = MetricsCollector(
            num_queues=len(config.queue_to_model),
            warmup_time=config.warmup_ms,
            queue_length_sample_interval=config.queue_sample_interval_ms,
        )
        self.trace = EventTraceWriter(config.event_trace_path)
        self.logger: logging.Logger = get_logger()
        self.logger.setLevel(getattr(logging, config.log_level.upper()))

        self._heap: List[_Event] = []
        self._seq = itertools.count()
        self._now = 0.0
        self._arrivals_generated = 0
        self._jobs_by_id: Dict[int, Job] = {}
        self._in_service: List[Job] = []  # jobs popped from a queue, batch not yet finished

    # -- event scheduling helpers --

    def _push(self, time: float, kind: str, **payload) -> None:
        heapq.heappush(
            self._heap, _Event(time, _TYPE_PRIORITY[kind], next(self._seq), kind, payload)
        )

    def _trace_write(self, kind: str, **fields) -> None:
        if self.trace.enabled:
            self.trace.write({"time": self._now, "kind": kind, **fields})

    # -- run --

    def run(self) -> SimResult:
        cfg = self.cfg
        if cfg.horizon_ms is None and cfg.max_jobs is None:
            raise ValueError("SimConfig needs at least one of horizon_ms / max_jobs")

        for stream in cfg.workload.streams:
            self._push(stream.start_offset_ms, ARRIVAL, stream_id=stream.stream_id)
        if cfg.queue_sample_interval_ms > 0:
            self._push(0.0, QUEUE_SAMPLE)

        while self._heap:
            event = heapq.heappop(self._heap)
            if cfg.horizon_ms is not None and event.time > cfg.horizon_ms:
                self._now = cfg.horizon_ms
                break
            self._now = event.time

            if event.kind == ARRIVAL:
                self._handle_arrival(event.payload["stream_id"])
            elif event.kind == BATCH_DONE:
                self._handle_batch_done(event.payload)
            elif event.kind == TIMER:
                pass  # just a wakeup, fallthrough to dispatch below
            elif event.kind == QUEUE_SAMPLE:
                self._handle_queue_sample()

            if event.kind != QUEUE_SAMPLE:
                self._maybe_dispatch()

        end_time = self._now if cfg.horizon_ms is None else cfg.horizon_ms
        summary = self.metrics.compute(sim_end_time=end_time)
        self.trace.close()
        self._log_summary(summary)
        queued_jobs = [
            j for q in range(len(cfg.queue_to_model)) for j in self.queue_set.peek(q)
        ]
        return SimResult(
            metrics_summary=summary,
            completed_jobs=self.metrics.completed_jobs,
            dropped_jobs=self.metrics.dropped_jobs,
            end_time=end_time,
            in_flight_jobs=list(self._in_service),
            queued_jobs=queued_jobs,
        )

    # -- event handlers --

    def _handle_arrival(self, stream_id: int) -> None:
        cfg = self.cfg
        if cfg.max_jobs is not None and self._arrivals_generated >= cfg.max_jobs:
            return
        job = Job(id=next_job_id(), stream_id=stream_id, arrival_time=self._now, queue_id=-1)
        queue_id = cfg.router.route(job, self._now)
        job.queue_id = queue_id
        self._arrivals_generated += 1
        self._jobs_by_id[job.id] = job

        admitted = self.queue_set.enqueue(job)
        self._trace_write("arrival", job_id=job.id, stream_id=stream_id, queue_id=queue_id, admitted=admitted)
        if not admitted:
            self.metrics.record_drop(job)
        else:
            cfg.policy.on_arrival(job, self._now)

        # schedule this stream's next arrival
        stream = cfg.workload.streams[stream_id]
        if cfg.max_jobs is None or self._arrivals_generated < cfg.max_jobs:
            interarrival = stream.arrival_process.next_interarrival(self.rng)
            self._push(self._now + interarrival, ARRIVAL, stream_id=stream_id)

    def _handle_batch_done(self, payload: dict) -> None:
        self.accelerator.finish_batch()
        jobs: List[Job] = payload["jobs"]
        for j in jobs:
            j.finish_time = self._now
            self.metrics.record_completion(j)
            self._in_service.remove(j)
        self._trace_write(
            "batch_done", queue_id=payload["queue_id"], model=payload["model"],
            batch_size=len(jobs), job_ids=[j.id for j in jobs],
        )
        self.cfg.policy.on_batch_complete(payload["queue_id"], jobs, self._now)

    def _handle_queue_sample(self) -> None:
        self.metrics.record_queue_lengths(self._now, self.queue_set.lengths())
        interval = self.cfg.queue_sample_interval_ms
        if interval > 0 and (self.cfg.horizon_ms is None or self._now + interval <= self.cfg.horizon_ms):
            self._push(self._now + interval, QUEUE_SAMPLE)

    # -- dispatch --

    def _build_view(self) -> PolicyView:
        cfg = self.cfg
        queues = {
            q: QueueView(
                length=self.queue_set.length(q),
                oldest_arrival_time=self.queue_set.oldest_arrival(q),
            )
            for q in range(len(cfg.queue_to_model))
        }
        return PolicyView(
            now=self._now,
            queues=queues,
            queue_to_model={q: m for q, m in enumerate(cfg.queue_to_model)},
            runtime_profile=cfg.runtime_profile,
            accelerator_free=not self.accelerator.busy,
            current_model=self.accelerator.current_model,
        )

    def _maybe_dispatch(self) -> None:
        if self.accelerator.busy:
            return
        view = self._build_view()
        action = self.cfg.policy.decide(view)
        validate_action(action, view)

        if isinstance(action, Wait):
            if action.until_time > self._now:
                self._push(action.until_time, TIMER)
            return

        assert isinstance(action, RunBatch)
        queue_id, batch_size = action.queue_id, action.batch_size
        model = self.cfg.queue_to_model[queue_id]
        duration = self.cfg.runtime_profile.cost(self.accelerator.current_model, model, batch_size)
        batch_jobs = self.queue_set.pop_batch(queue_id, batch_size)
        for j in batch_jobs:
            j.start_time = self._now
            j.batch_size = batch_size
        self._in_service.extend(batch_jobs)
        end = self.accelerator.start_batch(model, batch_size, self._now, duration)
        self.metrics.record_batch(self._now, end, model, batch_size, queue_id)
        self._trace_write(
            "run_batch", queue_id=queue_id, model=model, batch_size=batch_size,
            duration_ms=duration, job_ids=[j.id for j in batch_jobs],
        )
        self._push(end, BATCH_DONE, jobs=batch_jobs, queue_id=queue_id, model=model)

    def _log_summary(self, summary: dict) -> None:
        self.logger.info(
            "run done: %d completed, %d dropped, %d batches, mean turnaround=%.3fms, "
            "utilization=%.3f, compression_ratio=%s, stable=%s",
            summary["num_jobs_completed"],
            summary["num_jobs_dropped"],
            summary["num_batches"],
            summary["turnaround_mean_ms"] or float("nan"),
            summary["utilization"] or float("nan"),
            summary["compression_ratio"],
            summary["stable"],
        )
