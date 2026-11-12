# Checks analyze_probe.py on SYNTHETIC dumps whose clock behaviour is known (step, ramp, stall, idle start).
# This tests the analysis code only; it says nothing about the board.  Run: python3 test_analyze_probe.py
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_probe as ap

rng = np.random.RandomState(0)


def freq_curve(kind, f_old, f_new, t_ms):
    """Clock in MHz at times t_ms (relative to the write start): a step at 1.7 ms, a 5 ms ramp from 1.7 ms, or a step with a 3 ms stall."""
    f = np.where(t_ms < 1.7, f_old, f_new).astype(float)
    if kind == "ramp":
        frac = np.clip((t_ms - 1.7) / 5.0, 0.0, 1.0)
        f = f_old + (f_new - f_old) * frac
    return f


def make_dump(prefix, kind, mode, reps=6):
    ev_rows, sm_rows = [], []
    base = 5_000_000_000
    for r in range(reps):
        up = (r % 2 == 0)
        f_old, f_new = (829.0, 1237.0) if up else (1237.0, 829.0)
        t0 = base + r * 1_000_000_000                       # write start in ns
        write_ms = 1.8
        ev_rows.append({"rep": r, "mode": mode, "from_hz": int(f_old * 1e6), "to_hz": int(f_new * 1e6), "t_write_start_ns": t0,
                        "t_write_mid_ns": t0 + int(0.9e6), "t_write_end_ns": t0 + int(write_ms * 1e6),
                        "t_launch_start_ns": t0 + int(2e6) if mode == "idle" else 0, "t_launch_end_ns": t0 + int(2.1e6) if mode == "idle" else 0})
        t_first = 2.2 if mode == "idle" else -30.0          # idle mode: the kernel starts after the write returned
        t = t_first + np.cumsum(rng.uniform(0.012, 0.030, size=14000))      # one new counter value every 12-30 us, in ms
        t = t[t < 250.0]
        if kind == "stall":
            t = t[~((t > 1.7) & (t < 4.7))]                 # no new counter values for 3 ms
        f = freq_curve("ramp" if kind == "ramp" else "step", f_old, f_new, t)
        if mode == "idle":
            f = f * np.clip(0.6 + 0.4 * (t - 2.2) / 15.0, 0.6, 1.0) if kind != "ramp" else f   # idle: starts slow, reaches full speed at +17 ms
        # counter = integral of f (MHz = cycles per us)
        dt_us = np.diff(np.concatenate([[t_first], t])) * 1000.0
        cycles = 1_000_000 + np.cumsum(f * dt_us)
        # the host stamps each value a little after the GPU wrote it: mostly 0-3 us, now and then 30 us (an uncached read, a descheduled reader)
        err_ms = rng.exponential(0.0015, size=len(t)) + np.where(rng.uniform(size=len(t)) < 0.01, 0.03, 0.0)
        for ti, ei, ci in zip(t, err_ms, cycles):
            sm_rows.append((r, t0 + int((ti + ei) * 1e6), int(ci)))
    pd.DataFrame(ev_rows).to_csv(prefix + "_events.csv", index=False)
    pd.DataFrame(sm_rows, columns=["rep", "t_ns", "cycles"]).to_csv(prefix + "_samples.csv", index=False)


def run(kind, mode):
    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "t")
        make_dump(prefix, kind, mode)
        res, _ = ap.analyse(prefix)
    return res


def test_step_busy():
    res = run("step", "busy")
    assert abs(res["write_ms"].median() - 1.8) < 1e-6
    assert (res["settle_ms"] > 1.7).all() and (res["settle_ms"] < 2.2).all(), res["settle_ms"].tolist()    # step at 1.7 ms, windows are 0.2 ms
    assert (res["after_write_ms"] < 0.5).all(), res["after_write_ms"].tolist()
    assert (res["max_gap_ms"] < 0.1).all()
    assert (res["min_ratio"] > 0.97).all(), res["min_ratio"].tolist()
    assert abs(res[res["direction"] == "up"]["f_new_mhz"].median() - 1237.0) < 5.0
    print("ok test_step_busy: settle", round(res["settle_ms"].median(), 2), "ms, expected 1.7-2.0")


def test_ramp_busy():
    res = run("ramp", "busy")
    assert (res["settle_ms"] > 6.0).all() and (res["settle_ms"] < 7.2).all(), res["settle_ms"].tolist()      # ramp ends at 6.7 ms
    assert (res["after_write_ms"] > 4.0).all()
    print("ok test_ramp_busy: settle", round(res["settle_ms"].median(), 2), "ms, expected about 6.3-6.5 (within 3 % of the final clock a little before the ramp ends)")


def test_stall_busy():
    res = run("stall", "busy")
    assert (res["max_gap_ms"] > 2.9).all() and (res["max_gap_ms"] < 3.2).all(), res["max_gap_ms"].tolist()
    print("ok test_stall_busy: longest gap", round(res["max_gap_ms"].median(), 2), "ms, expected 3.0")


def test_idle_start():
    res = run("step", "idle")
    assert (res["first_ratio"] < 0.75).all(), res["first_ratio"].tolist()          # first window is at about 60 % of the final clock
    assert (res["settle_ms"] > 12.0).all() and (res["settle_ms"] < 20.0).all(), res["settle_ms"].tolist()
    print("ok test_idle_start: first clock", round(res["first_ratio"].median(), 2), "of final, settled at", round(res["settle_ms"].median(), 1), "ms")


if __name__ == "__main__":
    test_step_busy()
    test_ramp_busy()
    test_stall_busy()
    test_idle_start()
    print("all passed")
