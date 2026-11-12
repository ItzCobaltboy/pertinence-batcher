#!/usr/bin/env python3
"""Turns the dump of clock_probe into numbers: how long the clock write blocks, and when the GPU really runs at the new clock.

Input: <prefix>_samples.csv (rep, t_ns, cycles) and <prefix>_events.csv (write start / mid / end, kernel launch), written by clock_probe.
The GPU clock is cycles / time in sliding windows (200 us wide, moved in 50 us steps). Times are in ms relative to the START of the write.

Per repeat:
  write_ms        how long the two sysfs writes took (end - start)
  settle_ms       from the write start until the clock stays within 3 % of its final value
                  (busy mode: end of the last window outside the band within the first 50 ms; idle mode: not before the first window after the kernel launch;
                  a running median over 11 windows removes single noisy windows)
  after_write_ms  settle_ms - write_ms (negative or 0: the clock was already at the new value when the write returned)
  min_ratio       slowest window after the write start, as a fraction of the slower of the two clocks (a stall shows up as a number far below 1)
  max_gap_ms      longest time between two consecutive new counter values in the first 50 ms after the write start (a pause of the GPU or the reader)
  first_ratio     idle mode: the clock in the first window after the launch, as a fraction of the final clock

Usage:  python3 analyze_probe.py run_busy_829_1237 [run_idle_829_1237 ...]
"""
import os
import sys

import numpy as np
import pandas as pd

W_MS = 0.2
STEP_MS = 0.05
TOL = 0.03
MED_N = 11            # running median over this many windows: single 200 us windows are noisy (timestamp jitter of a few us is 1-2 %, outliers 5-8 %)
SCAN_MS = 50.0        # only the first 50 ms after the write start decide the settle time (the end of the recording holds the kernel stop)


def clock_series(t_ms, cycles, lo, hi):
    """(window start in ms, clock in MHz) for sliding windows; windows with fewer than two samples or a counter that went backwards are dropped."""
    starts = np.arange(lo, hi, STEP_MS)
    i0 = np.searchsorted(t_ms, starts, side="left")
    i1 = np.searchsorted(t_ms, starts + W_MS, side="left") - 1
    keep = i1 > i0
    starts, i0, i1 = starts[keep], i0[keep], i1[keep]
    dt_ns = (t_ms[i1] - t_ms[i0]) * 1e6
    dc = cycles[i1].astype(float) - cycles[i0].astype(float)
    good = (dc > 0) & (dt_ns >= 0.5 * W_MS * 1e6)
    return starts[good], (dc[good] / dt_ns[good]) * 1000.0         # cycles per ns is GHz, times 1000 is MHz


def one_repeat(ev, t_ms, cycles):
    """ev is one row of the events table; t_ms and cycles are the samples of that repeat, times relative to the write start."""
    out = {"rep": int(ev["rep"]), "mode": ev["mode"], "from_mhz": ev["from_hz"] / 1e6, "to_mhz": ev["to_hz"] / 1e6,
           "direction": "up" if ev["to_hz"] > ev["from_hz"] else "down"}
    out["write_ms"] = (ev["t_write_end_ns"] - ev["t_write_start_ns"]) / 1e6
    out["first_write_ms"] = (ev["t_write_mid_ns"] - ev["t_write_start_ns"]) / 1e6
    if len(t_ms) < 50:
        return out
    starts, f = clock_series(t_ms, cycles, t_ms[0], t_ms[-1] - W_MS)
    if len(f) == 0:
        return out
    t_end = t_ms[-1]
    tail = (starts >= t_end - 60.0) & (starts <= t_end - 10.0)
    if tail.sum() < 20:
        return out
    f_new = float(np.median(f[tail]))
    out["f_new_mhz"] = f_new
    pre = (starts < -2.0) & (starts > -25.0)
    out["f_old_mhz"] = float(np.median(f[pre])) if pre.sum() >= 20 else np.nan
    f_raw = f
    f = pd.Series(f).rolling(MED_N, center=True, min_periods=1).median().to_numpy()
    after = starts >= 0.0
    scan = after & (starts <= SCAN_MS)
    off_band = np.abs(f - f_new) > TOL * f_new
    bad_after = starts[scan & off_band]
    t_first_obs = float(starts[after][0]) if after.any() else 0.0
    settle = (float(bad_after.max()) + W_MS) if len(bad_after) else t_first_obs
    out["settle_ms"] = max(settle, t_first_obs, 0.0)
    out["after_write_ms"] = out["settle_ms"] - out["write_ms"]
    lower = min(f_new, out["f_old_mhz"]) if not np.isnan(out["f_old_mhz"]) else f_new
    out["min_ratio"] = float(f[scan].min() / lower) if scan.any() else np.nan
    sel = (t_ms >= 0.0) & (t_ms <= 50.0)
    out["max_gap_ms"] = float(np.diff(t_ms[sel]).max()) if sel.sum() > 2 else np.nan
    if ev["mode"] == "idle" and after.any():
        out["first_ratio"] = float(np.median(f_raw[after][:5]) / f_new)
        out["first_obs_ms"] = t_first_obs
    return out


def analyse(prefix):
    samples = pd.read_csv(prefix + "_samples.csv")
    events = pd.read_csv(prefix + "_events.csv")
    rows, traces = [], {}
    for _, ev in events.iterrows():
        s = samples[samples["rep"] == ev["rep"]]
        t_ms = (s["t_ns"].to_numpy() - ev["t_write_start_ns"]) / 1e6
        c = s["cycles"].to_numpy()
        rows.append(one_repeat(ev, t_ms, c))
        traces[int(ev["rep"])] = (t_ms, c, ev)
    return pd.DataFrame(rows), traces


def summary(res):
    def q(x, p):
        return float(np.nanpercentile(x, p)) if len(x) else np.nan
    cols = [c for c in ("write_ms", "settle_ms", "after_write_ms", "min_ratio", "max_gap_ms", "first_ratio") if c in res]
    lines = []
    for (mode, d), g in res.groupby(["mode", "direction"]):
        lines.append("%s, clock %s (%d repeats, %.0f -> %.0f MHz)" % (mode, d, len(g), g["from_mhz"].iloc[0], g["to_mhz"].iloc[0]))
        for c in cols:
            x = g[c].dropna().to_numpy()
            lines.append("    %-15s median %9.3f   p95 %9.3f   max %9.3f   min %9.3f   (n=%d)" % (c, q(x, 50), q(x, 95), q(x, 100), q(x, 0), len(x)))
    return "\n".join(lines)


def plot(prefix, res, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, d in zip(axes, ("up", "down")):
        reps = list(res[res["direction"] == d]["rep"])[:4]
        for r in reps:
            t_ms, c, ev = traces[r]
            starts, f = clock_series(t_ms, c, t_ms[0], t_ms[-1] - W_MS) if len(t_ms) > 50 else ([], [])
            ax.plot(starts, f, lw=1, label="repeat %d" % r)
            ax.axvline((ev["t_write_end_ns"] - ev["t_write_start_ns"]) / 1e6, color="grey", lw=0.6, ls=":")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-5, 30)
        ax.set_xlabel("ms from the start of the write (dotted: write returned)")
        ax.set_ylabel("GPU clock from the cycle counter (MHz)")
        ax.set_title("clock %s" % d)
        ax.legend(frameon=False)
    fig.tight_layout()
    out = prefix + "_clock.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    for prefix in sys.argv[1:]:
        print("=" * 100)
        print(prefix)
        res, traces = analyse(prefix)
        res.to_csv(prefix + "_summary.csv", index=False)
        print(summary(res))
        plot(prefix, res, traces)


if __name__ == "__main__":
    main()
