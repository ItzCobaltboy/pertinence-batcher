"""Deterministic periodic arrivals, one queue, batch size 1, no
overload: turnaround must equal service time exactly (no queueing)."""
from scheduler_sim.policies.fcfs_no_batch import FCFSNoBatch

from .helpers import make_sim, single_model_profile


def test_turnaround_equals_service_time_no_overload():
    profile = single_model_profile(service_time_ms=5.0, batch_size=1)
    sim = make_sim(
        policy=FCFSNoBatch(),
        profile=profile,
        period_ms=10.0,  # << service time -> never backs up
        horizon_ms=1000.0,
    )
    result = sim.run()
    assert result.completed_jobs, "no jobs completed"
    for job in result.completed_jobs:
        assert job.turnaround == 5.0
        assert job.wait_time == 0.0
    assert result.metrics_summary["turnaround_mean_ms"] == 5.0
