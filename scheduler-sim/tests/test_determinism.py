"""Same seed gives identical results."""
from scheduler_sim.job import reset_job_id_counter
from scheduler_sim.policies.timeout_batch import TimeoutBatch

from .helpers import make_sim, multi_batch_profile


def _run(seed):
    reset_job_id_counter()
    sim = make_sim(
        policy=TimeoutBatch(tau_ms=8.0),
        profile=multi_batch_profile(),
        period_ms=2.0,
        jitter_ms=1.5,
        horizon_ms=3000.0,
        seed=seed,
    )
    return sim.run()


def test_same_seed_identical_results():
    r1 = _run(42)
    r2 = _run(42)
    assert r1.metrics_summary == r2.metrics_summary
    assert [j.turnaround for j in r1.completed_jobs] == [j.turnaround for j in r2.completed_jobs]
    assert [j.arrival_time for j in r1.completed_jobs] == [
        j.arrival_time for j in r2.completed_jobs
    ]


def test_different_seed_differs():
    r1 = _run(1)
    r2 = _run(2)
    a1 = [j.arrival_time for j in r1.completed_jobs]
    a2 = [j.arrival_time for j in r2.completed_jobs]
    assert a1 != a2
