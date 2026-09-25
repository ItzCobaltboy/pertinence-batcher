"""JobRouter: fixed interface for assigning an arriving job to a queue."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .job import Job
from .registry import Registry

router_registry = Registry("router")


class JobRouter(ABC):
    """route(job, now) -> queue id. Implementations may use `rng` (an
    explicitly-owned numpy.random.Generator, never the global `random`
    module) for any randomness, so routing is reproducible from seed."""

    @abstractmethod
    def route(self, job: Job, now: float) -> int:
        raise NotImplementedError
