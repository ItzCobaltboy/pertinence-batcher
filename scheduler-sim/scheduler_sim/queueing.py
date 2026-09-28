"""QueueSet: the N FIFO queues, plus a pluggable drop policy for
what happens when an arriving job would overflow a full queue.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Union

from .job import Job
from .registry import Registry

drop_policy_registry = Registry("drop policy")


class DropPolicy(ABC):
    """Decides what happens when an arriving job finds its queue full.
    v1 ships only DropArriving; the interface exists so other rules
    (e.g. drop-oldest / drop-tail-of-batch) can be added later without
    touching QueueSet."""

    @abstractmethod
    def on_full_queue_arrival(self, queue: Deque[Job], arriving: Job) -> Optional[Job]:
        """Return the Job that should be dropped (arriving or an
        existing job), or None to admit anyway (only sensible for a
        policy that also evicts space first)."""
        raise NotImplementedError


@drop_policy_registry.register("drop_arriving")
class DropArriving(DropPolicy):
    """v1 behaviour: the arriving job itself is dropped; the queue is
    left untouched."""

    def on_full_queue_arrival(self, queue: Deque[Job], arriving: Job) -> Optional[Job]:
        return arriving


class QueueSet:
    def __init__(
        self,
        num_queues: int,
        max_queue_size: Optional[Union[int, Dict[int, int]]] = None,
        drop_policy: Optional[DropPolicy] = None,
    ):
        self.num_queues = num_queues
        self._queues: List[Deque[Job]] = [deque() for _ in range(num_queues)]
        if max_queue_size is None or isinstance(max_queue_size, int):
            self._max_size = {q: max_queue_size for q in range(num_queues)}
        else:
            self._max_size = {q: max_queue_size.get(q) for q in range(num_queues)}
        self.drop_policy = drop_policy or DropArriving()
        self.dropped_jobs: List[Job] = []

    # -- mutation, only the engine should call these --

    def enqueue(self, job: Job) -> bool:
        """Returns True if admitted, False if dropped."""
        q = self._queues[job.queue_id]
        limit = self._max_size.get(job.queue_id)
        if limit is not None and len(q) >= limit:
            dropped = self.drop_policy.on_full_queue_arrival(q, job)
            if dropped is not None:
                dropped.dropped = True
                self.dropped_jobs.append(dropped)
                if dropped is not job:
                    # a policy that evicts an existing job instead
                    q.remove(dropped)
                    q.append(job)
                return dropped is not job
            # policy admitted anyway (only valid if it made room itself)
        q.append(job)
        return True

    def pop_batch(self, queue_id: int, batch_size: int) -> List[Job]:
        q = self._queues[queue_id]
        if batch_size > len(q):
            raise ValueError(
                f"requested batch_size={batch_size} > queue {queue_id} length {len(q)}"
            )
        return [q.popleft() for _ in range(batch_size)]

    # -- read-only views --

    def length(self, queue_id: int) -> int:
        return len(self._queues[queue_id])

    def lengths(self) -> List[int]:
        return [len(q) for q in self._queues]

    def oldest_arrival(self, queue_id: int) -> Optional[float]:
        q = self._queues[queue_id]
        return q[0].arrival_time if q else None

    def peek(self, queue_id: int) -> List[Job]:
        """Read-only view of the jobs currently in a queue (oldest first)."""
        return list(self._queues[queue_id])

    def is_empty(self, queue_id: int) -> bool:
        return len(self._queues[queue_id]) == 0

    def total_queued(self) -> int:
        return sum(len(q) for q in self._queues)
