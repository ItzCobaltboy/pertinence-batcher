# Plain asserts on policy decisions over hand-built queue states. No torch needed.
# Run: python tests/test_policies.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import policies
from common.policies import Serve, State, Wait


def flat(q, b):
    return 10.0 + b


def state(arrivals, now=10.0, draining=False, b_max=8):
    return State(now, arrivals, flat, b_max, draining)


def test_fcfs_nobatch_takes_oldest_head_one_job():
    d = policies.FcfsNoBatch().decide(state([[5.0, 6.0], [4.0], [], [7.0]]))
    assert isinstance(d, Serve) and d.queue == 1 and d.b == 1


def test_fcfs_batch_takes_whole_queue_up_to_bmax():
    d = policies.FcfsBatch().decide(state([[5.0, 6.0], [4.0], [], [7.0]]))
    assert d.queue == 1 and d.b == 1
    d = policies.FcfsBatch().decide(state([[1.0] * 12, [4.0], [], []]))
    assert d.queue == 0 and d.b == 8


def test_lqf_longest_then_oldest_head():
    d = policies.Lqf().decide(state([[5.0, 6.0], [4.0], [1.0, 2.0, 3.0], [7.0]]))
    assert d.queue == 2 and d.b == 3
    # tie on length 2: queue 1 has the older head
    d = policies.Lqf().decide(state([[5.0, 6.0], [4.0, 9.0], [], []]))
    assert d.queue == 1 and d.b == 2


def test_triton_waits_before_timer_and_fires_after():
    p = policies.Triton(delay_ms=20)
    arrivals = [[9.99], [9.995], [], []]
    d = p.decide(state(arrivals, now=10.0))            # oldest waited 10 ms of 20
    assert isinstance(d, Wait) and abs(d.until - 10.01) < 1e-9
    d = p.decide(state(arrivals, now=10.0101))         # 20.1 ms
    assert isinstance(d, Serve) and d.queue == 0 and d.b == 1


def test_triton_full_queue_fires_without_waiting():
    p = policies.Triton(delay_ms=50)
    d = p.decide(state([[9.99] * 8, [9.995], [], []], now=10.0, b_max=8))
    assert isinstance(d, Serve) and d.queue == 0 and d.b == 8


def test_triton_delay_zero_is_oldest_head_full_batch():
    d = policies.Triton(delay_ms=0).decide(state([[9.0, 9.5], [8.0], [], []]))
    assert d.queue == 1 and d.b == 1


def test_triton_serves_oldest_among_ready():
    p = policies.Triton(delay_ms=5)
    d = p.decide(state([[9.0], [8.0], [9.999], []], now=10.0))
    assert d.queue == 1


def test_fixed_batch_waits_below_b():
    p = policies.FixedBatchB(4)
    d = p.decide(state([[9.0, 9.1, 9.2], [], [], []]))
    assert isinstance(d, Wait) and d.until > 10.0
    d = p.decide(state([[9.0, 9.1, 9.2, 9.3, 9.4], [1.0], [], []]))
    assert d.queue == 0 and d.b == 4


def test_fixed_batch_flushes_when_draining():
    d = policies.FixedBatchB(4).decide(state([[9.0, 9.1], [8.0], [], []], draining=True))
    assert isinstance(d, Serve) and d.queue == 1 and d.b == 1


def test_labels():
    assert policies.label("lqf", {}) == "lqf"
    assert policies.label("triton", {"delay_ms": 5}) == "triton_delay_ms5"
    assert policies.run_name("skewed", 16, "triton", {"delay_ms": 20}, 0) == "skewed_lam16_triton_delay_ms20_s0"


def test_stub_policies_refuse():
    try:
        policies.make_policy("index_T", {}).decide(state([[1.0], [], [], []]))
    except NotImplementedError:
        return
    assert False, "stub policy should raise"


def es_state(arrivals, now=10.0, b_max=8):
    return State(now, arrivals, flat, b_max, False)      # flat: 10 + b ms


def test_edgeserving_largest_batch_that_meets_slo():
    # 10 + b ms <= 15.5 ms  ->  b = 5
    d = policies.EdgeServing(15.5).decide(es_state([[10.0] * 6, [], [], []]))
    assert isinstance(d, Serve) and d.queue == 0 and d.b == 5
    d = policies.EdgeServing(1000.0).decide(es_state([[10.0] * 6, [], [], []]))
    assert d.b == 6


def test_edgeserving_falls_back_to_one_when_nothing_fits_and_never_waits():
    d = policies.EdgeServing(5.0).decide(es_state([[9.0] * 4, [], [], []]))
    assert isinstance(d, Serve) and d.b == 1


def test_edgeserving_serves_the_queue_with_the_urgent_job():
    # queue 0 holds a job that waited 0.5 s, queue 1 a fresh one: serving queue 0 leaves the lower urgency sum
    d = policies.EdgeServing(1000.0).decide(es_state([[9.5], [9.99], [], []]))
    assert d.queue == 0 and d.b == 1


def test_edgeserving_urgency_is_clipped():
    p = policies.EdgeServing(100.0)
    assert abs(p.urgency(0.1) - 1.0) < 1e-9
    assert abs(p.urgency(100.0) - 10.0) < 1e-9


def test_edgeserving_registered():
    assert isinstance(policies.make_policy("edgeserving", {"slo_ms": 300}), policies.EdgeServing)
    assert policies.label("edgeserving", {"slo_ms": 300}) == "edgeserving_slo_ms300"


def cw_svc(q, b, stat=None):
    assert stat == "p99"        # Clockwork plans with the p99 time
    return 10.0 + b


def cw_state(arrivals, now=10.0, b_max=16):
    return State(now, arrivals, cw_svc, b_max, False)


def test_clockwork_greedy_largest_size_the_oldest_job_allows():
    # fresh jobs, slo 15.5 ms: 10 + b <= 15.5 -> b in {1, 2, 4}; 6 queued -> batch of 4
    d = policies.Clockwork(15.5).decide(cw_state([[10.0] * 6, [], [], []]))
    assert isinstance(d, Serve) and d.queue == 0 and d.b == 4 and not d.drops
    # lenient: 6 queued, sizes 1,2,4 fit the supply (8 and 16 need more jobs)
    d = policies.Clockwork(1000.0).decide(cw_state([[10.0] * 6, [], [], []]))
    assert d.b == 4


def test_clockwork_cancels_jobs_that_cannot_make_it_even_alone():
    # slo 20 ms, T(1) = 11 ms: the job from 9.98 (deadline 10.000) is dead, the other two live
    d = policies.Clockwork(20.0).decide(cw_state([[9.98, 9.995, 10.0], [], [], []]))
    assert isinstance(d, Serve) and d.drops == {0: 1} and d.b == 2


def test_clockwork_waits_when_everything_is_dead():
    d = policies.Clockwork(20.0).decide(cw_state([[9.9], [], [], []]))
    assert isinstance(d, Wait) and d.drops == {0: 1}


def test_clockwork_smallest_latest_start_wins():
    # queue 1 has the older head, so its latest start is earlier
    d = policies.Clockwork(1000.0).decide(cw_state([[9.9], [9.5], [], []]))
    assert d.queue == 1 and d.b == 1


def test_clockwork_registered():
    assert isinstance(policies.make_policy("clockwork", {"slo_ms": 300}), policies.Clockwork)
    assert policies.label("clockwork", {"slo_ms": 300}) == "clockwork_slo_ms300"


def sy_state(arrivals, now=10.0, b_max=16):
    return State(now, arrivals, cw_svc, b_max, False)      # T(b) = 10 + b ms, p99 only


def test_symphony_holds_a_batch_until_frontrun_then_serves():
    # slo 100 ms, one job at 10.0: d = 10.1, T(1) = 11 ms, next size 2 -> frontrun = 10.1 - 0.012 = 10.088
    d = policies.Symphony(100.0).decide(sy_state([[10.0], [], [], []]))
    assert isinstance(d, Wait) and abs(d.until - 10.088) < 1e-9 and not d.drops
    d = policies.Symphony(100.0).decide(sy_state([[10.0], [], [], []], now=10.0885))
    assert isinstance(d, Serve) and d.queue == 0 and d.b == 1


def test_symphony_batch_size_is_calibrated_size_that_fits():
    # 6 fresh jobs, lenient slo: b = 6 (calibrated), next = 8 -> still waiting
    d = policies.Symphony(100.0).decide(sy_state([[10.0] * 6, [], [], []]))
    assert isinstance(d, Wait) and abs(d.until - (10.1 - 0.018)) < 1e-9
    # 7 live jobs: 7 is not a calibrated size, batch stays 6
    d = policies.Symphony(100.0).decide(sy_state([[10.0] * 7, [], [], []], now=10.083))
    assert isinstance(d, Serve) and d.b == 6


def test_symphony_full_batch_goes_immediately():
    d = policies.Symphony(1000.0).decide(sy_state([[10.0] * 16, [], [], []]))
    assert isinstance(d, Serve) and d.b == 16


def test_symphony_deadline_limited_batch_goes_immediately():
    # 16 jobs but the oldest job's deadline (10.017) only fits T(b) = 10 + b <= 17 ms -> b = 6 (next 8 does not fit): ready now
    d = policies.Symphony(17.0).decide(sy_state([[10.0] * 16, [], [], []]))
    assert isinstance(d, Serve) and d.b == 6


def test_symphony_smallest_latest_among_ready_and_drops():
    # both queues are full batches (ready); queue 1 has the older head, so the smaller latest
    d = policies.Symphony(1000.0).decide(sy_state([[9.9] * 16, [9.5] * 16, [], []]))
    assert d.queue == 1
    # a dead head is dropped and reported
    d = policies.Symphony(20.0).decide(sy_state([[9.98, 10.0], [], [], []], now=10.0))
    assert d.drops == {0: 1}


def test_symphony_registered():
    assert isinstance(policies.make_policy("symphony", {"slo_ms": 300}), policies.Symphony)
    assert policies.label("symphony", {"slo_ms": 300}) == "symphony_slo_ms300"
    import config
    assert policies.Symphony.SIZES == tuple(config.CALIB_BATCHES)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print("%d tests passed" % len(tests))
