"""SchedulingPolicy: fixed interface, Action types, and the read-only
view a policy decides on."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .registry import Registry
from .runtime_profile import RuntimeProfile

policy_registry = Registry("scheduling policy")


class InvalidAction(Exception):
    """Raised when a policy returns an action that fails validation
    (empty queue, unsupported batch size, batch larger than the queue
    unless padding is enabled, etc). Fails loudly by design -- the
    engine never silently coerces an invalid action."""


@dataclass(frozen=True)
class RunBatch:
    queue_id: int
    batch_size: int


@dataclass(frozen=True)
class Wait:
    until_time: float


Action = object  # RunBatch | Wait, kept loose to avoid a typing-union import


@dataclass(frozen=True)
class QueueView:
    length: int
    oldest_arrival_time: Optional[float]


@dataclass(frozen=True)
class PolicyView:
    """Read-only snapshot handed to SchedulingPolicy.decide()."""

    now: float
    queues: Dict[int, QueueView]  # queue_id -> QueueView
    queue_to_model: Dict[int, str]  # which model each queue is bound to
    runtime_profile: RuntimeProfile
    accelerator_free: bool
    current_model: Optional[str]  # model the accelerator last ran, or None

    def nonempty_queues(self) -> List[int]:
        return [q for q, v in self.queues.items() if v.length > 0]


class SchedulingPolicy(ABC):
    """decide(view) -> RunBatch(queue_id, batch_size) or Wait(until_time).

    Optional stateful hooks: on_arrival(job, now) and
    on_batch_complete(queue_id, batch, now) -- the engine calls these
    regardless of whether the policy overrides them (default no-ops),
    so a stateful policy (e.g. adaptive timeouts) can hook in without
    changing the engine.
    """

    @abstractmethod
    def decide(self, view: PolicyView) -> object:
        raise NotImplementedError

    def on_arrival(self, job, now: float) -> None:
        pass

    def on_batch_complete(self, queue_id: int, batch: List, now: float) -> None:
        pass


def validate_action(action: object, view: PolicyView) -> None:
    """Fails loudly (raises InvalidAction) rather than silently
    coercing anything the policy got wrong."""
    if isinstance(action, Wait):
        if action.until_time < view.now:
            raise InvalidAction(
                f"Wait(until_time={action.until_time}) is before now={view.now}"
            )
        return
    if not isinstance(action, RunBatch):
        raise InvalidAction(f"decide() must return RunBatch or Wait, got {type(action)}")

    if not view.accelerator_free:
        raise InvalidAction("RunBatch requested while accelerator is busy")
    if action.queue_id not in view.queues:
        raise InvalidAction(f"unknown queue_id {action.queue_id}")
    qv = view.queues[action.queue_id]
    if qv.length == 0:
        raise InvalidAction(f"RunBatch on empty queue {action.queue_id}")
    if action.batch_size <= 0:
        raise InvalidAction(f"batch_size must be positive, got {action.batch_size}")

    model = view.queue_to_model[action.queue_id]
    profile = view.runtime_profile
    if profile.partial_batch_mode == "pad":
        if action.batch_size > profile.max_batch_size(model):
            raise InvalidAction(
                f"batch_size {action.batch_size} exceeds max supported size "
                f"{profile.max_batch_size(model)} for model '{model}'"
            )
    else:
        # interpolate mode still requires >=1 and <= queue length below;
        # any size is "supported" via interpolation.
        pass

    if action.batch_size > qv.length:
        raise InvalidAction(
            f"batch_size {action.batch_size} exceeds queue {action.queue_id} "
            f"length {qv.length} (padding runs the cost of a larger engine on "
            f"the jobs actually present, it never invents extra jobs)"
        )
