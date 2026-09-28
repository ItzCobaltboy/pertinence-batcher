import numpy as np
import pytest

from scheduler_sim.job import Job
from scheduler_sim.routers.sticky import StickyRouter
from scheduler_sim.routers.trace import TraceRouter
from scheduler_sim.routers.weighted_random import WeightedRandomRouter


def _job(stream_id=0):
    return Job(id=0, stream_id=stream_id, arrival_time=0.0, queue_id=-1)


def test_weighted_random_respects_zero_weight_queues():
    rng = np.random.default_rng(0)
    router = WeightedRandomRouter(weights=[1.0, 0.0], rng=rng)
    results = {router.route(_job(), 0.0) for _ in range(200)}
    assert results == {0}


def test_trace_router_replays_in_order_and_loops():
    router = TraceRouter([0, 1, 2], loop=True)
    got = [router.route(_job(), 0.0) for _ in range(7)]
    assert got == [0, 1, 2, 0, 1, 2, 0]


def test_trace_router_raises_when_exhausted_without_loop():
    router = TraceRouter([0, 1], loop=False)
    router.route(_job(), 0.0)
    router.route(_job(), 0.0)
    with pytest.raises(IndexError):
        router.route(_job(), 0.0)


def test_sticky_router_sticks_with_probability_one():
    rng = np.random.default_rng(1)
    router = StickyRouter(base_weights=[1.0, 1.0, 1.0], p=1.0, rng=rng)
    first = router.route(_job(stream_id=5), 0.0)
    for _ in range(20):
        assert router.route(_job(stream_id=5), 0.0) == first


def test_sticky_router_independent_streams():
    rng = np.random.default_rng(2)
    router = StickyRouter(base_weights=[1.0, 1.0], p=1.0, rng=rng)
    q_a = router.route(_job(stream_id="A"), 0.0)
    q_b = router.route(_job(stream_id="B"), 0.0)
    # each stream keeps its own last-queue state
    assert router.route(_job(stream_id="A"), 0.0) == q_a
    assert router.route(_job(stream_id="B"), 0.0) == q_b
