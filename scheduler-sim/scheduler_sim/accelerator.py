"""Accelerator: single server, tracks busy intervals and current model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Accelerator:
    current_model: Optional[str] = None
    busy: bool = False
    busy_intervals: List[Tuple[float, float, str, int]] = field(default_factory=list)
    # (start, end, model, batch_size) per batch run

    def start_batch(self, model: str, batch_size: int, now: float, duration_ms: float) -> float:
        if self.busy:
            raise RuntimeError("Accelerator.start_batch called while already busy")
        self.busy = True
        self.current_model = model
        end = now + duration_ms
        self.busy_intervals.append((now, end, model, batch_size))
        return end

    def finish_batch(self) -> None:
        if not self.busy:
            raise RuntimeError("Accelerator.finish_batch called while idle")
        self.busy = False

    @property
    def total_busy_time(self) -> float:
        return sum(end - start for start, end, _, _ in self.busy_intervals)

    @property
    def num_batches(self) -> int:
        return len(self.busy_intervals)
