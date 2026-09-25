"""Job record."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Optional

_id_counter = count()


def next_job_id() -> int:
    """Monotonic id generator, used by Workload so job ids are stable
    and independent of any RNG draw order."""
    return next(_id_counter)


def reset_job_id_counter() -> None:
    """Reset the global id counter. Call this at the start of a run if
    you need job ids to be reproducible/deterministic across repeated
    simulations in the same process (e.g. in tests)."""
    global _id_counter
    _id_counter = count()


@dataclass
class Job:
    """One frame/job flowing through the simulator.

    `deadline` and `priority` are optional fields unused by the v1
    policies -- present so a later deadline- or priority-aware policy
    extension doesn't need a Job schema rewrite.
    """

    id: int
    stream_id: int
    arrival_time: float
    queue_id: int
    deadline: Optional[float] = None
    priority: Optional[float] = None

    # Filled in by the simulator as the job is served.
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    batch_size: Optional[int] = None
    dropped: bool = False

    @property
    def turnaround(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def wait_time(self) -> Optional[float]:
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    def deadline_missed(self) -> Optional[bool]:
        if self.deadline is None or self.finish_time is None:
            return None
        return self.finish_time > self.deadline
