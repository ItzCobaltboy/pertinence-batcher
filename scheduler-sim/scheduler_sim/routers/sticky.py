from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..job import Job
from ..router import JobRouter, router_registry


@router_registry.register("sticky")
class StickyRouter(JobRouter):
    """Per-stream Markov chain: a stream's consecutive frames stay in
    the same queue with probability `p`; otherwise the queue is
    redrawn from the base per-queue probabilities. Mimics real video,
    where consecutive frames tend to route to the same model.
    """

    def __init__(self, base_weights: Sequence[float], p: float, rng: np.random.Generator):
        base_weights = np.asarray(base_weights, dtype=float)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        total = base_weights.sum()
        if total <= 0:
            raise ValueError("base_weights must sum to > 0")
        self.base_probabilities = base_weights / total
        self.num_queues = len(base_weights)
        self.p = p
        self.rng = rng
        self._last_queue: Dict[int, int] = {}

    def route(self, job: Job, now: float) -> int:
        last = self._last_queue.get(job.stream_id)
        if last is not None and self.rng.random() < self.p:
            q = last
        else:
            q = int(self.rng.choice(self.num_queues, p=self.base_probabilities))
        self._last_queue[job.stream_id] = q
        return q
