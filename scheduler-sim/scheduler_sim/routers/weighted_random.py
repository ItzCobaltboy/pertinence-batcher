from __future__ import annotations

from typing import Sequence

import numpy as np

from ..job import Job
from ..router import JobRouter, router_registry


@router_registry.register("weighted_random")
class WeightedRandomRouter(JobRouter):
    """Fixed per-queue probabilities, drawn independently per job."""

    def __init__(self, weights: Sequence[float], rng: np.random.Generator):
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("weights must be a non-empty 1D sequence")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        total = weights.sum()
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        self.probabilities = weights / total
        self.num_queues = len(weights)
        self.rng = rng

    def route(self, job: Job, now: float) -> int:
        return int(self.rng.choice(self.num_queues, p=self.probabilities))
