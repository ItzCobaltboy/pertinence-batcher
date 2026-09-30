"""Tests for the simulator. Run from scheduler-sim/ with: python -m pytest tests/ -q"""
import os
import sys

import numpy as np
import pytest

# sim.py, workloads.py and schedulers.py live one folder up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import Job, Profile, Queue, Scheduler, Simulator
from schedulers import (FCFSBatchScheduler, FCFSNoBatchScheduler, LongestQueueScheduler,
                        TimeoutBatchScheduler)
from workloads import PeriodicWorkload, StickyWorkload, UniformWorkload, WeightedWorkload

ONE_MODEL_CSV = "model,1\nm0,3.0\n"
TWO_MODEL_CSV = "# two made-up models\nmodel,1,4,8\nm0,1.0,2.0,3.0\nm1,2.0,,6.0\n"


def make_profile(tmp_path, text):
    path = tmp_path / "profile.csv"
    path.write_text(text)
    return Profile(str(path))


def make_queues(profile):
    queues = []
    for queue_id in range(len(profile.models)):
        queues.append(Queue(queue_id, profile.models[queue_id]))
    return queues


def add_job(queue, arrival_time):
    queue.push(Job(0, 0, arrival_time, queue.queue_id))


def run_weighted(tmp_path, seed, load=0.3):
    profile = make_profile(tmp_path, TWO_MODEL_CSV)
    queues = make_queues(profile)
    workload = WeightedWorkload(queues, np.random.default_rng(seed), 4, load, [0.7, 0.3])
    scheduler = FCFSBatchScheduler(queues, profile)
    sim = Simulator(queues, workload, scheduler, profile, horizon_ms=5000)
    return sim, workload, sim.run()


def route_many(workload, number_of_jobs, stream_id=0):
    """Push number_of_jobs jobs through handle_arrival and count them per queue."""
    counts = [0] * len(workload.queues)
    for i in range(number_of_jobs):
        job, accepted = workload.handle_arrival(float(i), stream_id)
        counts[job.queue_id] += 1
    return counts


# ---- whole-simulation tests ----

def test_periodic_turnaround_equals_service_time(tmp_path):
    profile = make_profile(tmp_path, ONE_MODEL_CSV)
    queues = make_queues(profile)
    workload = PeriodicWorkload(queues, np.random.default_rng(0), 1, period_ms=10.0)
    sim = Simulator(queues, workload, FCFSNoBatchScheduler(queues, profile), profile, 1000)
    metrics = sim.run()
    assert metrics["num_jobs_completed"] > 90
    for job in sim.completed_jobs:
        assert job.finish_time - job.arrival_time == 3.0


def test_poisson_mean_wait_matches_md1(tmp_path):
    service_ms = 1.0
    rho = 0.5
    profile = make_profile(tmp_path, "model,1\nm0," + str(service_ms) + "\n")
    queues = make_queues(profile)
    workload = UniformWorkload(queues, np.random.default_rng(7), 1, rho / service_ms)
    sim = Simulator(queues, workload, FCFSNoBatchScheduler(queues, profile), profile,
                    horizon_ms=400000, sample_interval_ms=10000)
    metrics = sim.run()
    expected_wait = rho * service_ms / (2 * (1 - rho))
    assert metrics["wait_mean_ms"] == pytest.approx(expected_wait, rel=0.05)


def test_conservation(tmp_path):
    sim, workload, metrics = run_weighted(tmp_path, seed=3, load=0.8)
    still_queued = 0
    for queue in sim.queues:
        still_queued += queue.length()
    total = len(sim.completed_jobs) + len(sim.dropped_jobs) + still_queued + len(sim.in_service)
    assert workload.jobs_created == total
    job_ids = []
    for job in sim.completed_jobs:
        job_ids.append(job.job_id)
    assert len(job_ids) == len(set(job_ids))  # nobody served twice
    longest_batch = 6.0
    assert metrics["busy_time_ms"] <= sim.horizon_ms + longest_batch


def test_same_seed_same_result_different_seed_differs(tmp_path):
    first = run_weighted(tmp_path, seed=11)[2]
    second = run_weighted(tmp_path, seed=11)[2]
    other = run_weighted(tmp_path, seed=12)[2]
    assert first == second
    assert first != other


def test_invalid_action_raises(tmp_path):
    class RunsEmptyQueue(Scheduler):
        def decide(self, now):
            return ("run", 1, 1)  # queue 1 never gets any jobs below

    class BatchTooBig(Scheduler):
        def decide(self, now):
            return ("run", 0, 4)

    for bad_class in [RunsEmptyQueue, BatchTooBig]:
        profile = make_profile(tmp_path, TWO_MODEL_CSV)
        queues = make_queues(profile)
        workload = PeriodicWorkload(queues, np.random.default_rng(0), 1, 10.0, queue_for_stream=[0])
        sim = Simulator(queues, workload, bad_class(queues, profile), profile, 100)
        with pytest.raises(ValueError):
            sim.run()


# ---- profile ----

def test_profile_parsing_and_padding(tmp_path):
    profile = make_profile(tmp_path, TWO_MODEL_CSV)
    assert profile.models == ["m0", "m1"]
    assert profile.table["m1"] == {1: 2.0, 8: 6.0}  # empty cell for batch 4 is skipped
    assert profile.runtime("m0", 4) == 2.0
    assert profile.runtime("m0", 3) == 2.0  # padded up to 4
    assert profile.runtime("m1", 2) == 6.0  # 4 unsupported, padded up to 8
    assert profile.max_batch("m1") == 8
    with pytest.raises(ValueError):
        profile.runtime("m0", 9)


# ---- schedulers, on hand-built queues ----

def test_fcfs_no_batch_picks_oldest_with_tie_to_lower_id(tmp_path):
    profile = make_profile(tmp_path, TWO_MODEL_CSV)
    queues = make_queues(profile)
    scheduler = FCFSNoBatchScheduler(queues, profile)
    assert scheduler.decide(0.0) is None
    add_job(queues[0], 5.0)
    add_job(queues[1], 2.0)
    add_job(queues[1], 3.0)
    assert scheduler.decide(10.0) == ("run", 1, 1)
    queues[0].jobs[0].arrival_time = 2.0  # now a tie on 2.0
    assert scheduler.decide(10.0) == ("run", 0, 1)


def test_fcfs_batch_runs_whole_queue_capped_at_max(tmp_path):
    profile = make_profile(tmp_path, TWO_MODEL_CSV)
    queues = make_queues(profile)
    for t in range(10):
        add_job(queues[1], float(t))
    add_job(queues[0], 50.0)
    assert FCFSBatchScheduler(queues, profile).decide(100.0) == ("run", 1, 8)


def test_longest_queue_with_tie_to_lower_id(tmp_path):
    profile = make_profile(tmp_path, TWO_MODEL_CSV)
    queues = make_queues(profile)
    scheduler = LongestQueueScheduler(queues, profile)
    add_job(queues[0], 9.0)
    add_job(queues[1], 1.0)
    add_job(queues[1], 2.0)
    assert scheduler.decide(10.0) == ("run", 1, 2)
    add_job(queues[0], 9.5)
    assert scheduler.decide(10.0) == ("run", 0, 2)


def test_timeout_batch_waits_then_runs(tmp_path):
    profile = make_profile(tmp_path, TWO_MODEL_CSV)
    queues = make_queues(profile)
    scheduler = TimeoutBatchScheduler(queues, profile, tau_ms=5.0)
    add_job(queues[0], 1.0)
    add_job(queues[1], 2.0)
    assert scheduler.decide(3.0) == ("wait", 6.0)  # nobody full, queue 0 times out first
    assert scheduler.decide(6.0) == ("run", 0, 1)
    for t in range(8):
        add_job(queues[1], 3.0 + t)  # queue 1 is now full (9 >= 8) before its timeout
    assert scheduler.decide(4.0) == ("run", 1, 8)
    with pytest.raises(ValueError):
        TimeoutBatchScheduler(queues, profile, tau_ms=-1.0)


# ---- workloads ----

def test_uniform_workload_spreads_evenly(tmp_path):
    queues = make_queues(make_profile(tmp_path, TWO_MODEL_CSV)) + [Queue(2, "m0"), Queue(3, "m0")]
    workload = UniformWorkload(queues, np.random.default_rng(1), 1, 1.0)
    counts = route_many(workload, 20000)
    for count in counts:
        assert count == pytest.approx(20000 / 4, rel=0.05)


def test_weighted_workload_matches_weights(tmp_path):
    queues = make_queues(make_profile(tmp_path, TWO_MODEL_CSV))
    workload = WeightedWorkload(queues, np.random.default_rng(2), 1, 1.0, [3, 1])
    counts = route_many(workload, 20000)
    assert abs(counts[0] / 20000 - 0.75) < 0.02


def test_weighted_workload_rejects_bad_weights(tmp_path):
    queues = make_queues(make_profile(tmp_path, TWO_MODEL_CSV))
    for bad_weights in [[1.0], [1.0, -1.0], [0.0, 0.0]]:
        with pytest.raises(ValueError):
            WeightedWorkload(queues, np.random.default_rng(0), 1, 1.0, bad_weights)


def test_sticky_workload_extremes(tmp_path):
    queues = make_queues(make_profile(tmp_path, TWO_MODEL_CSV))
    always_stay = StickyWorkload(queues, np.random.default_rng(3), 5, 1.0, [0.5, 0.5], 1.0)
    for stream_id in range(5):
        counts = route_many(always_stay, 200, stream_id)
        assert 0 in counts  # every job of this stream went to one single queue

    never_stay = StickyWorkload(queues, np.random.default_rng(4), 1, 1.0, [3, 1], 0.0)
    counts = route_many(never_stay, 20000)
    assert abs(counts[0] / 20000 - 0.75) < 0.02
