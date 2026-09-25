"""Conservation: every arrived job completes, is dropped, is still
queued, or is in-flight (mid-batch) at the end; busy time <= sim time;
no job is served twice."""
from scheduler_sim.policies.fcfs_batch import FCFSBatch
from scheduler_sim.policies.longest_queue import LongestQueue
from scheduler_sim.routers.weighted_random import WeightedRandomRouter

import numpy as np

from .helpers import make_sim, multi_batch_profile


def _check_conservation(sim, result):
    total_accounted = (
        len(result.completed_jobs)
        + len(result.dropped_jobs)
        + len(result.queued_jobs)
        + len(result.in_flight_jobs)
    )
    assert total_accounted == sim._arrivals_generated

    seen_ids = set()
    for j in result.completed_jobs:
        assert j.id not in seen_ids, f"job {j.id} completed twice"
        seen_ids.add(j.id)

    # A batch that starts right at the horizon can legitimately run
    # past it (the engine doesn't preempt), so allow slack up to the
    # longest single batch duration recorded.
    max_batch_duration = max((e - s for s, e, *_ in sim.metrics.batches), default=0.0)
    assert result.metrics_summary["busy_time_ms"] <= result.end_time + max_batch_duration + 1e-9


def test_conservation_fcfs_batch():
    sim = make_sim(policy=FCFSBatch(), profile=multi_batch_profile(), period_ms=2.0, horizon_ms=5000.0)
    result = sim.run()
    _check_conservation(sim, result)
    assert len(result.completed_jobs) > 0


def test_conservation_longest_queue_multi_queue():
    rng = np.random.default_rng(3)
    router = WeightedRandomRouter(weights=[1.0, 2.0, 3.0], rng=rng)
    sim = make_sim(
        policy=LongestQueue(),
        router=router,
        profile=multi_batch_profile(),
        num_queues=3,
        period_ms=1.5,
        horizon_ms=5000.0,
    )
    result = sim.run()
    _check_conservation(sim, result)
    assert len(result.completed_jobs) > 0


def test_conservation_with_drops():
    sim = make_sim(
        policy=FCFSBatch(),
        profile=multi_batch_profile(),
        period_ms=0.5,  # overload -> queue should back up and drop
        horizon_ms=2000.0,
        max_queue_size=5,
    )
    result = sim.run()
    _check_conservation(sim, result)
    assert len(result.dropped_jobs) > 0, "expected overload to cause drops"
