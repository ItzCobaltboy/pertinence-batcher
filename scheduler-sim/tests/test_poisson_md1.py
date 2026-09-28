"""Poisson arrivals, one queue, fixed (deterministic) service time,
batch size 1: mean wait should match the M/D/1 formula within
sampling error.

M/D/1 mean queueing wait: Wq = rho * D / (2 * (1 - rho)), rho = lambda * D.
"""
from scheduler_sim.policies.fcfs_no_batch import FCFSNoBatch

from .helpers import make_sim, single_model_profile


def test_mean_wait_matches_md1_formula():
    service_time_ms = 5.0
    rho = 0.5
    rate_per_ms = rho / service_time_ms

    profile = single_model_profile(service_time_ms=service_time_ms, batch_size=1)
    sim = make_sim(
        policy=FCFSNoBatch(),
        profile=profile,
        poisson_rate_per_ms=rate_per_ms,
        horizon_ms=200_000.0,
        warmup_ms=5_000.0,
        seed=7,
    )
    result = sim.run()

    expected_wq = rho * service_time_ms / (2 * (1 - rho))  # = 2.5 ms
    observed_wq = result.metrics_summary["wait_mean_ms"]
    assert observed_wq is not None
    # Sampling error tolerance: within 20% of the theoretical value.
    assert abs(observed_wq - expected_wq) / expected_wq < 0.2, (
        observed_wq,
        expected_wq,
    )
