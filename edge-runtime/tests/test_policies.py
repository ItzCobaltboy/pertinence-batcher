import math

import pytest

from common import Job, Profile
from policies import (Clockwork, CostV1, Drop, EdgeServing, FcfsBatch, FcfsNoBatch, Lqf, Serve, State,
                      Symphony, Triton, Wait)

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
# ms; n cheap, l expensive. Same numbers for every statistic.
TABLE = {
    "yolov8n": {1: 10, 2: 12, 4: 16, 8: 24, 16: 40},
    "yolov8s": {1: 15, 2: 18, 4: 24, 8: 36, 16: 60},
    "yolov8m": {1: 25, 2: 30, 4: 40, 8: 60, 16: 100},
    "yolov8l": {1: 40, 2: 50, 4: 70, 8: 110, 16: 190},
}
PROFILE = Profile.from_table(TABLE)
_ids = [0]


def job(model, t_arrival, slo_ms=100.0, t_enqueued=None):
    _ids[0] += 1
    j = Job(_ids[0], "cam", model, t_arrival, slo_ms, _ids[0])
    j.t_enqueued = t_arrival if t_enqueued is None else t_enqueued
    return j


def state(now, queues, b_max=16, profile=PROFILE):
    full = {m: tuple(queues.get(m, ())) for m in MODELS}
    return State(now, full, MODELS, profile, b_max)


# ---------------------------------------------------------------- baselines
def test_fcfs_and_lqf():
    s = state(1.0, {"yolov8n": [job("yolov8n", 0.95), job("yolov8n", 0.96), job("yolov8n", 0.97)],
                    "yolov8l": [job("yolov8l", 0.90)]})
    assert FcfsNoBatch().decide(s) == Serve("yolov8l", 1)
    assert FcfsBatch().decide(s) == Serve("yolov8l", 1)
    assert Lqf().decide(s) == Serve("yolov8n", 3)
    tie = state(1.0, {"yolov8n": [job("yolov8n", 0.95)], "yolov8s": [job("yolov8s", 0.94)]})
    assert Lqf().decide(tie) == Serve("yolov8s", 1)
    big = state(1.0, {"yolov8m": [job("yolov8m", 0.5 + i * 0.01) for i in range(20)]}, b_max=16)
    assert FcfsBatch().decide(big) == Serve("yolov8m", 16)


def test_state_sizes():
    s = state(0.0, {})
    assert s.sizes(1) == [1]
    assert s.sizes(5) == [1, 2, 4, 5]
    assert s.sizes(40) == [1, 2, 4, 8, 16]
    assert abs(s.t("yolov8n", 3) - 0.014) < 1e-12          # interpolated, seconds


# ---------------------------------------------------------------- Triton
def test_triton_waits_until_delay_then_serves():
    p = Triton(max_queue_delay_ms=10)
    q = [job("yolov8n", 1.000), job("yolov8n", 1.004)]
    d = p.decide(state(1.005, {"yolov8n": q}))
    assert isinstance(d, Wait) and abs(d.until - 1.010) < 1e-12
    assert p.decide(state(1.010, {"yolov8n": q})) == Serve("yolov8n", 2)


def test_triton_delay_counts_from_enqueue_not_capture():
    p = Triton(max_queue_delay_ms=10)
    q = [job("yolov8n", 1.000, t_enqueued=1.008)]   # 8 ms of preprocessing before the queue
    d = p.decide(state(1.012, {"yolov8n": q}))
    assert isinstance(d, Wait) and abs(d.until - 1.018) < 1e-12


def test_triton_full_and_preferred_batches_go_immediately():
    p = Triton(max_queue_delay_ms=50, max_batch=8)
    full = [job("yolov8s", 1.0 + i * 1e-3) for i in range(9)]
    assert p.decide(state(1.010, {"yolov8s": full})) == Serve("yolov8s", 8)
    pref = Triton(max_queue_delay_ms=50, preferred_batch_sizes=[2, 4])
    five = [job("yolov8s", 1.0 + i * 1e-3) for i in range(5)]
    assert pref.decide(state(1.006, {"yolov8s": five})) == Serve("yolov8s", 4)
    one = [job("yolov8s", 1.0)]
    assert isinstance(pref.decide(state(1.001, {"yolov8s": one})), Wait)


def test_triton_zero_delay_and_oldest_ready_head():
    p = Triton(max_queue_delay_ms=0)
    s = state(1.0, {"yolov8n": [job("yolov8n", 0.99)], "yolov8l": [job("yolov8l", 0.98)]})
    assert p.decide(s) == Serve("yolov8l", 1)
    per_queue = Triton(max_queue_delay_ms=100, delays_ms={"yolov8n": 0})
    assert per_queue.decide(s) == Serve("yolov8n", 1)


# ---------------------------------------------------------------- EdgeServing
def f(w, tau, C=10.0):
    return min(math.exp(w / tau - 1.0), C)


def test_edgeserving_picks_stability_score_minimiser():
    now = 1.0
    n_job = job("yolov8n", now - 0.090)                       # 90 ms old, SLO 100
    l_jobs = [job("yolov8l", now - 0.010) for _ in range(3)]  # 10 ms old
    s = state(now, {"yolov8n": [n_job], "yolov8l": l_jobs})
    p = EdgeServing()
    # serve n (B=1, 10 ms): the three l jobs wait 20 ms
    S_n = 3 * f(0.020, 0.1)
    # serve l (B=3, T(3) = 60 ms interpolated): the n job waits 150 ms
    S_l = f(0.150, 0.1)
    assert abs(p.score(s, "yolov8n", 1) - S_n) < 1e-9
    assert abs(p.score(s, "yolov8l", 3) - S_l) < 1e-9
    assert S_n < S_l
    assert p.decide(s) == Serve("yolov8n", 1)
    assert Lqf().decide(s) == Serve("yolov8l", 3)             # where LQF differs


def test_edgeserving_clip_bmax_and_p95():
    p = EdgeServing(b_max=10)
    many = [job("yolov8m", 1.0 - 0.001 * i) for i in range(12)]
    assert p.decide(state(1.0, {"yolov8m": many})) == Serve("yolov8m", 10)
    assert p.urgency(1e6, 0.1) == 10.0
    assert abs(p.urgency(0.1 * (1 + math.log(10)), 0.1) - 10.0) < 1e-9
    # p95 used for L: give the profile a p95 different from p50
    data = Profile.from_table(TABLE).data
    for m in data["models"]:
        for b in data["models"][m]:
            data["models"][m][b]["total"] = dict(data["models"][m][b]["total"])
            data["models"][m][b]["total"]["p95_ms"] = 2 * data["models"][m][b]["total"]["p50_ms"]
    prof = Profile(data)
    s = state(1.0, {"yolov8n": [job("yolov8n", 0.99)], "yolov8l": [job("yolov8l", 0.95)]}, profile=prof)
    # serve n: l job waits 50 ms + 2 x 10 ms (p95) = 70 ms
    assert abs(EdgeServing().score(s, "yolov8n", 1) - f(0.070, 0.1)) < 1e-9
    assert abs(EdgeServing(stat="p50").score(s, "yolov8n", 1) - f(0.060, 0.1)) < 1e-9


# ---------------------------------------------------------------- Clockwork
def test_clockwork_picks_most_urgent_valid_strategy_and_grows_it():
    now = 1.0
    # n: 5 jobs, oldest 40 ms old (SLO 100 -> exec deadline 55 ms from now with buffer 5)
    n = [job("yolov8n", now - 0.040 + 0.002 * i) for i in range(5)]
    # l: 1 fresh job, lots of slack
    l = [job("yolov8l", now - 0.001)]
    p = Clockwork(online=False)
    d = p.decide(state(now, {"yolov8n": n, "yolov8l": l}))
    # most urgent: n's head; (n0, 4) is valid (16 ms <= 55 ms), grows to the largest
    # supported size the 5 jobs allow = 4
    assert d == Serve("yolov8n", 4, [j.job_id for j in n[:4]])


def test_clockwork_skips_head_that_blocks_a_bigger_batch():
    now = 1.0
    head = job("yolov8m", now - 0.060)                        # d = 35 ms from now (buffer 5)
    fresh = [job("yolov8m", now - 0.001) for _ in range(7)]
    p = Clockwork(online=False)
    s = state(now, {"yolov8m": [head] + fresh})
    # head's strategies: 7 more jobs allow b = 8 (60 ms) > 35 ms -> every head strategy fails;
    # next: a fresh job's (j, 4) is valid (40 ms <= 94 ms) and the head can't make it -> skipped
    assert p.decide(s) == Drop("yolov8m", [head.job_id])
    s2 = state(now, {"yolov8m": fresh})
    assert p.decide(s2) == Serve("yolov8m", 4, [j.job_id for j in fresh[:4]])


def test_clockwork_drops_jobs_that_cannot_finish_alone():
    now = 1.0
    late = job("yolov8l", now - 0.070)       # 30 ms to deadline, minus 5 ms buffer < 40 ms
    ok = job("yolov8l", now - 0.010)
    p = Clockwork(online=False)
    assert p.decide(state(now, {"yolov8l": [late, ok]})) == Drop("yolov8l", [late.job_id])
    nd = Clockwork(online=False, drop=False)
    assert nd.decide(state(now, {"yolov8l": [late, ok]})) == Serve("yolov8l", 1, [ok.job_id])
    assert nd.decide(state(now, {"yolov8l": [late]})) == Serve("yolov8l", 1, [late.job_id])


def test_clockwork_online_estimate_is_p99_of_window():
    p = Clockwork(window=10)
    s = state(1.0, {})
    assert abs(p.est(s, "yolov8n", 1) - 0.010) < 1e-12        # seeded from the profile
    for ms in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:   # window keeps the last 10
        p.on_batch_done("yolov8n", 1, ms)
    # window = 12..21 ms; p99 position 0.99 * 9 = 8.91 -> 20 + 0.91 * (21 - 20)
    assert abs(p.est(s, "yolov8n", 1) - 0.02091) < 1e-9


# ---------------------------------------------------------------- Symphony
def test_symphony_waits_until_frontrun_then_launches():
    p = Symphony()
    q = [job("yolov8n", 1.000), job("yolov8n", 1.010)]       # head deadline 1.100
    d = p.decide(state(1.011, {"yolov8n": q}))
    # |B| = 2, frontrun = d - l(3) = 1.100 - 0.014
    assert isinstance(d, Wait) and abs(d.until - 1.086) < 1e-9
    assert p.decide(state(1.086, {"yolov8n": q})) == Serve("yolov8n", 2, [j.job_id for j in q])


def test_symphony_batch_limited_by_deadline_and_bmax():
    p = Symphony()
    # head deadline 1.050, now 1.005: l(1) = 40 ms fits (1.045), l(2) = 50 ms does not (1.055)
    q = [job("yolov8l", 0.950), job("yolov8l", 1.000)]
    d = p.decide(state(1.005, {"yolov8l": q}))
    assert d == Serve("yolov8l", 1, [q[0].job_id])           # exec = max(now, d - l(2)) = now
    full = [job("yolov8n", 1.0 + 1e-4 * i) for i in range(4)]
    assert p.decide(state(1.001, {"yolov8n": full}, b_max=4)) == Serve("yolov8n", 4, [j.job_id for j in full])


def test_symphony_most_urgent_due_candidate_and_drops():
    p = Symphony()
    n = [job("yolov8n", 0.920)]            # d 1.020: due (frontrun 1.008), latest 1.010
    s = [job("yolov8s", 0.918)]            # d 1.018: due (frontrun 1.000), latest 1.003
    assert p.decide(state(1.002, {"yolov8n": n, "yolov8s": s})) == Serve("yolov8s", 1, [s[0].job_id])
    late = job("yolov8m", 0.900)           # d 1.000 < now + 25 ms
    assert p.decide(state(1.002, {"yolov8m": [late]})) == Drop("yolov8m", [late.job_id])
    nd = Symphony(drop=False)
    assert nd.decide(state(1.002, {"yolov8m": [late], "yolov8n": n})) == Serve("yolov8m", 1, [late.job_id])


# ---------------------------------------------------------------- cost_v1
def test_cost_v1_beta_extremes():
    now = 1.0
    s = state(now, {"yolov8n": [job("yolov8n", 0.99) for _ in range(3)], "yolov8l": [job("yolov8l", 0.90)]})
    # beta = 0: only compute counts -> the cheapest option, n at b = 1
    assert CostV1(beta=0.0).decide(s) == Serve("yolov8n", 1)
    # beta = 1: only waiting counts -> clear as much age as possible
    d = CostV1(beta=1.0).decide(s)
    assert d.queue == "yolov8n" and d.b == 3
    # lam > 0 weighs the old l job super-linearly
    assert CostV1(beta=1.0, lam=0.1).decide(s) == Serve("yolov8l", 1)


def test_symphony_margin_shifts_the_window():
    q = [job("yolov8n", 1.000), job("yolov8n", 1.010)]
    d = Symphony(margin_ms=5).decide(state(1.011, {"yolov8n": q}))
    assert isinstance(d, Wait) and abs(d.until - 1.081) < 1e-9
