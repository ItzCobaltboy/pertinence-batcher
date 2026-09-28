"""Workload: K streams, each producing arrivals under an ArrivalProcess.

All randomness (jitter, Poisson draws) goes through an explicitly
passed numpy.random.Generator -- never the global `random` module --
so a run is fully reproducible from its seed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .registry import Registry

arrival_process_registry = Registry("arrival process")


class ArrivalProcess(ABC):
    """Generates successive inter-arrival times (in ms) for one stream."""

    @abstractmethod
    def next_interarrival(self, rng: np.random.Generator) -> float:
        """Return the time (ms) until the next arrival, given the
        previous one just happened."""
        raise NotImplementedError


@arrival_process_registry.register("periodic")
@dataclass
class PeriodicArrival(ArrivalProcess):
    """Fixed period, optionally with uniform jitter of +/- jitter_ms."""

    period_ms: float
    jitter_ms: float = 0.0

    def next_interarrival(self, rng: np.random.Generator) -> float:
        if self.jitter_ms <= 0:
            return self.period_ms
        delta = rng.uniform(-self.jitter_ms, self.jitter_ms)
        return max(0.0, self.period_ms + delta)


@arrival_process_registry.register("poisson")
@dataclass
class PoissonArrival(ArrivalProcess):
    """Poisson process with a given mean rate (jobs/ms), i.e.
    exponential inter-arrival times."""

    rate_per_ms: float

    def next_interarrival(self, rng: np.random.Generator) -> float:
        return rng.exponential(1.0 / self.rate_per_ms)


@dataclass
class Stream:
    """One video stream feeding one (fixed) queue via the router."""

    stream_id: int
    arrival_process: ArrivalProcess
    start_offset_ms: float = 0.0


@dataclass
class Workload:
    streams: List[Stream]

    def initial_arrival_times(self) -> List[float]:
        return [s.start_offset_ms for s in self.streams]
