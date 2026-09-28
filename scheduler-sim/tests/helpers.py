"""Shared test helpers: a single-model runtime profile and a small
sim-building helper so individual tests stay short."""
from __future__ import annotations

import tempfile
from typing import List, Optional

import numpy as np

from scheduler_sim.engine import SimConfig, Simulator
from scheduler_sim.policy import SchedulingPolicy
from scheduler_sim.router import JobRouter
from scheduler_sim.routers.uniform_random import UniformRandomRouter
from scheduler_sim.runtime_profile import RuntimeProfile
from scheduler_sim.workload import PeriodicArrival, PoissonArrival, Stream, Workload


def single_model_profile(service_time_ms: float = 5.0, batch_size: int = 1) -> RuntimeProfile:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(f"model,{batch_size}\nmodelA,{service_time_ms}\n")
    f.close()
    return RuntimeProfile.from_csv(f.name)


def multi_batch_profile() -> RuntimeProfile:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write("model,1,4,8\nmodelA,5.0,6.0,9.0\n")
    f.close()
    return RuntimeProfile.from_csv(f.name)


def make_sim(
    policy: SchedulingPolicy,
    router: Optional[JobRouter] = None,
    profile: Optional[RuntimeProfile] = None,
    num_queues: int = 1,
    period_ms: float = 10.0,
    jitter_ms: float = 0.0,
    poisson_rate_per_ms: Optional[float] = None,
    seed: int = 1,
    horizon_ms: float = 1000.0,
    max_jobs: Optional[int] = None,
    warmup_ms: float = 0.0,
    max_queue_size: Optional[int] = None,
) -> Simulator:
    profile = profile or single_model_profile()
    rng = np.random.default_rng(seed)
    router = router or UniformRandomRouter(num_queues=num_queues, rng=rng)
    if poisson_rate_per_ms is not None:
        arrival = PoissonArrival(rate_per_ms=poisson_rate_per_ms)
    else:
        arrival = PeriodicArrival(period_ms=period_ms, jitter_ms=jitter_ms)
    workload = Workload(streams=[Stream(0, arrival)])
    cfg = SimConfig(
        queue_to_model=["modelA"] * num_queues if num_queues > 1 else ["modelA"],
        runtime_profile=profile,
        workload=workload,
        router=router,
        policy=policy,
        seed=seed,
        horizon_ms=horizon_ms,
        max_jobs=max_jobs,
        warmup_ms=warmup_ms,
        max_queue_size=max_queue_size,
        queue_sample_interval_ms=10.0,
    )
    return Simulator(cfg)
