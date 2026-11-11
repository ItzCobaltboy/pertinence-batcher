# Unit tests for the DVFS policy (common/dvfs.py). Plain asserts, no board needed.  Run: python3 tests/test_dvfs.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common import dvfs
from common.policies import State, Serve

FREQS = [500, 1000, 1500]
SIZES = [1, 2, 4, 16]
# latency ms: base(b) * speed(f); energy J: minimum at 1000 MHz
SPEED = {500: 2.0, 1000: 1.0, 1500: 0.7}
ENERGY = {500: 1.3, 1000: 1.0, 1500: 1.5}


def make_table():
    rows = {}
    for m, base in (("a", 10.0), ("b", 20.0)):
        for b in SIZES:
            for f in FREQS:
                rows[(m, b, f)] = (base * (1 + 0.25 * (b - 1)) * SPEED[f], 0.01 * b * ENERGY[f] * base)
    return dvfs.Table(rows)


T = make_table()


def policy(alpha, slo_ms=1000.0, **kw):
    return dvfs.DvfsAlpha(alpha, slo_ms, T, ["a", "b"], switch_ms=88.0, **kw)


def state(arrivals, now=10.0, f0=1500, b_max=16):
    return State(now, arrivals, None, b_max, False, freq_now=f0)


def test_table_interpolates_between_sizes():
    lo, hi = T.lat("a", 4, 1000), T.lat("a", 16, 1000)
    mid = T.lat("a", 10, 1000)
    assert abs(mid - (lo + (hi - lo) * 0.5)) < 1e-9


def test_alpha_one_picks_least_energy_clock_that_is_feasible():
    d = policy(1.0).decide(state([[9.9], []]))
    assert isinstance(d, Serve) and d.queue == 0 and d.freq == 1000


def test_alpha_zero_picks_the_fastest_clock():
    d = policy(0.0).decide(state([[9.9], []], f0=500))
    assert d.freq == 1500


def test_prune_drops_clocks_that_miss_the_deadline():
    # slo 40 ms, job just arrived, f0 = 1500 so no switch on 1500. 500 MHz needs 20 + 88 ms, 1000 MHz needs 10 + 88 ms
    d = policy(1.0, slo_ms=40.0).decide(state([[10.0], []]))
    assert d.freq == 1500


def test_switch_cost_counts_only_when_the_clock_changes():
    # slo 50 ms: staying at 1000 (f0=1000) takes 10 ms and fits; leaving 1500 for 1000 adds 88 ms and does not
    d = policy(1.0, slo_ms=50.0).decide(state([[10.0], []], f0=1000))
    assert d.freq == 1000
    d = policy(1.0, slo_ms=50.0).decide(state([[10.0], []], f0=1500))
    assert d.freq == 1500


def test_batch_is_the_largest_that_has_a_feasible_clock():
    # b=16 at 1500 takes 10*(1+3.75)*0.7 = 33 ms; slo 20 ms rules it out; b=4 takes 10*1.75*0.7 = 12.25 ms fits
    d = policy(0.0, slo_ms=20.0).decide(state([[10.0] * 20, []]))
    assert d.b < 16 and d.b >= 1
    tau = 10.0 + T.lat("a", d.b, d.freq) / 1000.0
    assert tau <= 10.0 + 0.020 + 1e-9


def test_least_slack_queue_goes_first():
    # queue 1 has the older job, so less slack
    d = policy(1.0).decide(state([[9.9], [9.2]]))
    assert d.queue == 1


def test_check2_protects_the_other_queue():
    # queue 0 alone could use 500 MHz (alpha=1 would like 1000/500 for energy) but queue 1 is nearly late:
    # serving 0 slowly would push queue 1 past its deadline, so the pick must be queue 1 first or a fast clock.
    p = policy(1.0, slo_ms=300.0)
    d = p.decide(state([[10.0], [9.72]], now=10.0, f0=1500))
    assert d.queue == 1
    assert p.n_clean == 1 and p.n_risky == 0


def test_risky_fallback_when_check2_cannot_hold():
    # both queues are about to expire together: check 1 holds for the first, check 2 for the second cannot
    p = policy(1.0, slo_ms=30.0)
    d = p.decide(state([[9.985], [9.985]], now=10.0, f0=1500))
    assert isinstance(d, Serve) and p.n_risky == 1 and p.n_clean == 0


def test_miss_fallback_serves_least_slack_at_top_clock():
    p = policy(1.0, slo_ms=5.0)
    d = p.decide(state([[9.0], [9.5]], now=10.0, f0=500))
    assert p.n_miss == 1 and d.queue == 0 and d.freq == 1500 and d.b == 1


def test_hysteresis_keeps_the_current_clock_when_close():
    # alpha 0.5: scores for 1000 and 1500 are close; with f0=1500 and a wide tolerance the clock stays
    d = policy(0.5, keep_within=0.5).decide(state([[9.9], []], f0=1500))
    assert d.freq == 1500
    d = policy(0.5, keep_within=0.0).decide(state([[9.9], []], f0=1500))
    assert d.freq in FREQS


def test_alpha_sweep_is_monotone_in_clock():
    freqs = [policy(a / 10.0, keep_within=0.0).decide(state([[9.9], []], f0=1000)).freq for a in range(11)]
    assert freqs == sorted(freqs, reverse=True), freqs        # more weight on energy never picks a faster clock


def test_never_serves_more_than_queued_or_b_max():
    d = policy(0.5).decide(state([[9.9] * 3, []], b_max=2))
    assert d.b <= 2


def test_fixed_policy_uses_its_clock():
    d = dvfs.DvfsFixed(1000, 1000.0, T, ["a", "b"]).decide(state([[9.9], [9.0]]))
    assert d.freq == 1000 and d.queue == 1


def test_label_and_registry_names():
    from common import policies
    assert policies.label("dvfs_alpha", {"alpha": 0.3, "slo_ms": 600}) == "dvfs_alpha_alpha0.3_slo_ms600"
    assert policies.label("dvfs_alpha", {"alpha": 0.0, "slo_ms": 600}) == "dvfs_alpha_alpha0_slo_ms600"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print("PASS", n)
        except AssertionError as e:
            failed += 1
            print("FAIL", n, e)
    print("%d tests, %d failed" % (len(names), failed))
    sys.exit(1 if failed else 0)
