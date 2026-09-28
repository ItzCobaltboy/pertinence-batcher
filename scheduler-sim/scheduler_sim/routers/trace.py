"""TraceRouter: replays a fixed sequence of queue labels from a CSV.

Expected CSV structure (header required): a single `queue_id` column,
one row per arrival, in the *global* arrival order jobs are routed in
(i.e. the order route() is called across all streams combined, not
per-stream). Example:

    queue_id
    0
    1
    0
    2

See scheduler-sim/data/trace_router_example.csv for a runnable dummy
trace, and TraceRouter.from_recall_csv() below for deriving a real
trace from the YOLO/COCO recall benchmark.
"""
from __future__ import annotations

import csv
from typing import List, Sequence

from ..job import Job
from ..router import JobRouter, router_registry


@router_registry.register("trace")
class TraceRouter(JobRouter):
    def __init__(self, trace: Sequence[int], loop: bool = True):
        if len(trace) == 0:
            raise ValueError("trace must be non-empty")
        self.trace: List[int] = list(trace)
        self.loop = loop
        self._i = 0

    @classmethod
    def from_csv(cls, path: str, loop: bool = True) -> "TraceRouter":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "queue_id" not in reader.fieldnames:
                raise ValueError(f"{path}: expected a 'queue_id' column")
            trace = [int(row["queue_id"]) for row in reader]
        return cls(trace, loop=loop)

    def route(self, job: Job, now: float) -> int:
        if self._i >= len(self.trace):
            if not self.loop:
                raise IndexError(
                    f"TraceRouter exhausted its {len(self.trace)}-entry trace "
                    "and loop=False"
                )
            self._i = 0
        q = self.trace[self._i]
        self._i += 1
        return q
