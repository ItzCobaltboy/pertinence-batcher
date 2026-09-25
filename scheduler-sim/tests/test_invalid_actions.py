"""Invalid-action rejection: the engine fails loudly rather than
silently coercing a bad action."""
import pytest

from scheduler_sim.policy import (
    InvalidAction,
    PolicyView,
    QueueView,
    RunBatch,
    SchedulingPolicy,
    Wait,
    validate_action,
)
from scheduler_sim.runtime_profile import RuntimeProfile

from .helpers import make_sim, multi_batch_profile


def _view(profile, accel_free=True):
    return PolicyView(
        now=0.0,
        queues={0: QueueView(length=3, oldest_arrival_time=0.0)},
        queue_to_model={0: "modelA"},
        runtime_profile=profile,
        accelerator_free=accel_free,
        current_model=None,
    )


def test_run_batch_on_empty_queue_rejected():
    profile = multi_batch_profile()
    view = PolicyView(
        now=0.0,
        queues={0: QueueView(length=0, oldest_arrival_time=None)},
        queue_to_model={0: "modelA"},
        runtime_profile=profile,
        accelerator_free=True,
        current_model=None,
    )
    with pytest.raises(InvalidAction):
        validate_action(RunBatch(queue_id=0, batch_size=1), view)


def test_run_batch_exceeding_queue_length_rejected():
    profile = multi_batch_profile()
    view = _view(profile)
    with pytest.raises(InvalidAction):
        validate_action(RunBatch(queue_id=0, batch_size=10), view)


def test_run_batch_exceeding_max_supported_size_in_pad_mode_rejected():
    profile = multi_batch_profile()  # max supported size = 8
    view = PolicyView(
        now=0.0,
        queues={0: QueueView(length=20, oldest_arrival_time=0.0)},
        queue_to_model={0: "modelA"},
        runtime_profile=profile,
        accelerator_free=True,
        current_model=None,
    )
    with pytest.raises(InvalidAction):
        validate_action(RunBatch(queue_id=0, batch_size=16), view)


def test_run_batch_while_accelerator_busy_rejected():
    profile = multi_batch_profile()
    view = _view(profile, accel_free=False)
    with pytest.raises(InvalidAction):
        validate_action(RunBatch(queue_id=0, batch_size=1), view)


def test_wait_before_now_rejected():
    profile = multi_batch_profile()
    view = _view(profile)
    with pytest.raises(InvalidAction):
        validate_action(Wait(until_time=-1.0), view)


def test_non_action_return_type_rejected():
    profile = multi_batch_profile()
    view = _view(profile)
    with pytest.raises(InvalidAction):
        validate_action("not an action", view)


class BrokenPolicy(SchedulingPolicy):
    """Always returns an invalid batch size to prove the engine
    propagates the failure instead of swallowing it."""

    def decide(self, view):
        return RunBatch(queue_id=0, batch_size=999)


def test_engine_raises_on_invalid_policy_action():
    sim = make_sim(policy=BrokenPolicy(), profile=multi_batch_profile(), horizon_ms=100.0)
    with pytest.raises(InvalidAction):
        sim.run()
