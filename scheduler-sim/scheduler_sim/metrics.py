"""MetricsCollector: observer attached to engine events. Computes the
summary statistics described in the spec: turnaround, compute (busy
time + compression ratio), throughput/utilization/idle time, per-model
batch histograms, queue-length-over-time, a stability flag, and
deadline miss rate when deadlines are set.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .job import Job


@dataclass
class MetricsCollector:
    num_queues: int
    warmup_time: float = 0.0
    queue_length_sample_interval: float = 50.0

    completed_jobs: List[Job] = field(default_factory=list)
    dropped_jobs: List[Job] = field(default_factory=list)
    # (start, end, model, batch_size, queue_id)
    batches: List[Tuple[float, float, str, int, int]] = field(default_factory=list)
    # (time, [length per queue])
    queue_length_samples: List[Tuple[float, List[int]]] = field(default_factory=list)

    def record_completion(self, job: Job) -> None:
        self.completed_jobs.append(job)

    def record_drop(self, job: Job) -> None:
        self.dropped_jobs.append(job)

    def record_batch(
        self, start: float, end: float, model: str, batch_size: int, queue_id: int
    ) -> None:
        self.batches.append((start, end, model, batch_size, queue_id))

    def record_queue_lengths(self, now: float, lengths: List[int]) -> None:
        self.queue_length_samples.append((now, list(lengths)))

    def _post_warmup(self, jobs: List[Job]) -> List[Job]:
        return [j for j in jobs if j.arrival_time >= self.warmup_time]

    def _post_warmup_batches(self):
        return [b for b in self.batches if b[0] >= self.warmup_time]

    @staticmethod
    def _percentile(values: List[float], p: float) -> Optional[float]:
        if not values:
            return None
        return float(np.percentile(values, p))

    def compute(self, sim_end_time: float) -> Dict:
        jobs = self._post_warmup(self.completed_jobs)
        dropped = self._post_warmup(self.dropped_jobs)
        turnarounds = [j.turnaround for j in jobs if j.turnaround is not None]
        waits = [j.wait_time for j in jobs if j.wait_time is not None]

        per_queue_turnaround: Dict[int, List[float]] = defaultdict(list)
        for j in jobs:
            if j.turnaround is not None:
                per_queue_turnaround[j.queue_id].append(j.turnaround)

        batches = self._post_warmup_batches()
        busy_time = sum(end - start for start, end, *_ in batches)
        effective_horizon = max(sim_end_time - self.warmup_time, 1e-9)
        num_jobs = len(jobs)
        num_batches = len(batches)
        compression_ratio = (num_batches / num_jobs) if num_jobs else None

        per_model_batches: Dict[str, List[int]] = defaultdict(list)
        per_model_jobs_served: Dict[str, int] = defaultdict(int)
        per_model_wait: Dict[str, List[float]] = defaultdict(list)
        queue_to_model: Dict[int, str] = {}
        for start, end, model, b, qid in batches:
            per_model_batches[model].append(b)
            per_model_jobs_served[model] += b
            queue_to_model[qid] = model
        for j in jobs:
            model = queue_to_model.get(j.queue_id)
            if model is not None and j.wait_time is not None:
                per_model_wait[model].append(j.wait_time)

        max_queue_len = 0
        if self.queue_length_samples:
            max_queue_len = max(max(lens) for _, lens in self.queue_length_samples)

        stability = self._assess_stability()

        deadline_jobs = [j for j in jobs if j.deadline is not None]
        deadline_miss_rate = None
        if deadline_jobs:
            misses = sum(1 for j in deadline_jobs if j.deadline_missed())
            deadline_miss_rate = misses / len(deadline_jobs)

        summary = {
            "num_jobs_completed": num_jobs,
            "num_jobs_dropped": len(dropped),
            "num_batches": num_batches,
            "turnaround_mean_ms": float(np.mean(turnarounds)) if turnarounds else None,
            "turnaround_p50_ms": self._percentile(turnarounds, 50),
            "turnaround_p95_ms": self._percentile(turnarounds, 95),
            "turnaround_p99_ms": self._percentile(turnarounds, 99),
            "turnaround_max_ms": max(turnarounds) if turnarounds else None,
            "turnaround_per_queue_mean_ms": {
                q: float(np.mean(v)) for q, v in per_queue_turnaround.items()
            },
            "wait_mean_ms": float(np.mean(waits)) if waits else None,
            "busy_time_ms": busy_time,
            "compression_ratio": compression_ratio,
            "throughput_jobs_per_ms": num_jobs / effective_horizon if num_jobs else 0.0,
            "utilization": busy_time / effective_horizon if effective_horizon > 0 else None,
            "idle_time_ms": max(effective_horizon - busy_time, 0.0),
            "max_queue_length": max_queue_len,
            "stable": stability,
            "deadline_miss_rate": deadline_miss_rate,
            "per_model": {
                model: {
                    "jobs_served": per_model_jobs_served[model],
                    "num_batches": len(per_model_batches[model]),
                    "batch_size_histogram": {
                        int(k): int(v)
                        for k, v in zip(*np.unique(per_model_batches[model], return_counts=True))
                    }
                    if per_model_batches[model]
                    else {},
                    "mean_wait_ms": float(np.mean(per_model_wait[model]))
                    if per_model_wait[model]
                    else None,
                }
                for model in per_model_batches
            },
        }
        return summary

    def _assess_stability(self) -> bool:
        """Heuristic: compare queue-length-over-time in the first vs.
        second half of the (post-warmup) sampled window; a run is
        flagged unstable if total queued length grows by more than 50%
        and keeps a positive slope -- a simple linear-trend check
        rather than a formal drift criterion."""
        samples = [
            (t, sum(lens)) for t, lens in self.queue_length_samples if t >= self.warmup_time
        ]
        if len(samples) < 4:
            return True
        times = np.array([t for t, _ in samples], dtype=float)
        totals = np.array([tot for _, tot in samples], dtype=float)
        # linear fit slope of total queue occupancy vs time
        slope, _ = np.polyfit(times, totals, 1)
        mean_total = totals.mean() if totals.mean() > 0 else 1.0
        # normalized slope: fraction of mean occupancy gained per ms,
        # scaled over the observed window
        window = max(times[-1] - times[0], 1e-9)
        relative_growth = slope * window / mean_total
        return bool(relative_growth < 0.5)
