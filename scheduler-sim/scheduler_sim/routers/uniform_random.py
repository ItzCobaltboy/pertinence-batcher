from __future__ import annotations

import numpy as np

from ..job import Job
from ..router import JobRouter, router_registry


@router_registry.register("uniform_random")
class UniformRandomRouter(JobRouter):
    """Every arriving job goes to a uniformly random queue, independent
    of its stream."""

    def __init__(self, num_queues: int, rng: np.random.Generator):
        self.num_queues = num_queues
        self.rng = rng

    def route(self, job: Job, now: float) -> int:
        return int(self.rng.integers(0, self.num_queues))
