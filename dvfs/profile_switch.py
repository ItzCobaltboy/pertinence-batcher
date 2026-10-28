# Switching cost of the GPU clock, GPU idle (no model loaded). For every ordered pair (from, to) of FREQS_HZ,
# SWITCH_REPEATS times, in shuffled order: pin `from`, wait SWITCH_WAIT_S, pin `to`, time how long cur_freq and
# tegrastats GR3D_FREQ take to show it, watch SWITCH_WAIT_S more. Averages per pair and for up vs down at the end.
# Run:  sudo python3 -m dvfs.profile_switch [--dry-run] [--ignore-busy]     (summary only: --summary-only DIR)
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import math
import os
import random
import time

import numpy as np

from dvfs import settings
from dvfs import tegrastats

FIELDS = ["rep_id", "order_index", "from_hz", "to_hz", "direction", "rep", "cur_before_hz", "t_write_start",
          "write_first", "write_first_ms", "write_second_ms", "write_total_ms", "t_cur_match", "cur_switch_ms",
          "cur_after_hz", "tegra_switch_ms", "dmesg_hits", "temp_max_c", "timestamp"]
TRACE_FIELDS = ["rep_id", "dt_s", "gr3d_mhz", "gr3d_load"]       # tegrastats samples around each switch, dt from the write start


def hot_c(parsed):
    return tegrastats.max_temp(parsed, settings.TEMP_IGNORE)


def plan():
    pairs = [(a, b) for a in settings.FREQS_HZ for b in settings.FREQS_HZ if a != b]
    reps = [(a, b, r) for (a, b) in pairs for r in range(settings.SWITCH_REPEATS)]
    random.Random(settings.SEED).shuffle(reps)
    return reps


def per_rep_s():
    return 2 * settings.SWITCH_WAIT_S + 0.5


def tegra_match(sampler, t_from, to_hz):
    """Monotonic time of the first tegrastats sample at or after t_from whose GR3D_FREQ shows the target, else None."""
    for t, parsed, _ in sampler.window(t_from, t_from + settings.SWITCH_WAIT_S + 1.0, pad=0):
        mhz = parsed.get("gr3d_mhz")
        if t >= t_from and mhz is not None and abs(mhz - to_hz / 1e6) <= settings.TEGRA_MATCH_TOL_MHZ:
            return t
    return None


def one_rep(s, order_index, frm, to, rep, rows, traces):
    gpu = s.gpu
    gpu.pin(frm)
    t, cur = gpu.wait_cur(frm, settings.SWITCH_POLL_TIMEOUT_S)
    if t is None:
        raise RuntimeError("cur_freq stayed at %d after pinning %d" % (cur, frm))
    time.sleep(settings.SWITCH_WAIT_S)
    cur_before = gpu.cur_freq()
    w = gpu.pin(to)
    t_match, cur_after = gpu.wait_cur(to, settings.SWITCH_POLL_TIMEOUT_S)
    time.sleep(max(0.0, settings.SWITCH_WAIT_S - (time.monotonic() - w["t_start"])))
    t_tegra = tegra_match(s.sampler, w["t_start"], to)
    rep_id = "%d_%d_%d" % (frm, to, rep)
    window = s.sampler.window(w["t_start"] - 0.5, w["t_start"] + settings.SWITCH_WAIT_S, pad=0)
    for t, parsed, _ in window:
        traces.append({"rep_id": rep_id, "dt_s": "%.4f" % (t - w["t_start"]), "gr3d_mhz": parsed.get("gr3d_mhz"),
                       "gr3d_load": parsed.get("gr3d_load")})
    temps = [hot_c(p) for _, p, _ in window]
    temps = [x for x in temps if x is not None]
    rows.append({"rep_id": rep_id, "order_index": order_index, "from_hz": frm, "to_hz": to,
                 "direction": "up" if to > frm else "down", "rep": rep, "cur_before_hz": cur_before,
                 "t_write_start": "%.6f" % w["t_start"], "write_first": w["order"][0],
                 "write_first_ms": "%.3f" % w["write_ms"][0], "write_second_ms": "%.3f" % w["write_ms"][1],
                 "write_total_ms": "%.3f" % ((w["t_end"] - w["t_start"]) * 1000.0),
                 "t_cur_match": "" if t_match is None else "%.6f" % t_match,
                 "cur_switch_ms": "" if t_match is None else "%.3f" % ((t_match - w["t_start"]) * 1000.0),
                 "cur_after_hz": cur_after,
                 "tegra_switch_ms": "" if t_tegra is None else "%.3f" % ((t_tegra - w["t_start"]) * 1000.0),
                 "dmesg_hits": len(s.sampler.dmesg_hits(w["t_start"] - 1.0, time.monotonic())),
                 "temp_max_c": "" if not temps else "%.1f" % max(temps), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})


def append_csv(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def stats(values):
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v)), "min": float(min(v)), "max": float(max(v))}


def num(x):
    return float(x) if x not in ("", None) else float("nan")


def summarize(out_dir):
    with open(os.path.join(out_dir, "switch_raw.csv")) as f:
        rows = list(csv.DictReader(f))
    metrics = ["write_total_ms", "cur_switch_ms", "tegra_switch_ms"]
    groups = {}
    for r in rows:
        groups.setdefault((int(r["from_hz"]), int(r["to_hz"])), []).append(r)
        groups.setdefault(r["direction"], []).append(r)
    out = []
    for key in sorted(groups, key=lambda k: (isinstance(k, str), k)):
        line = {"from_mhz": "" if isinstance(key, str) else key[0] / 1e6, "to_mhz": key if isinstance(key, str) else key[1] / 1e6}
        line["n_reps"] = len(groups[key])
        for m in metrics:
            st = stats([num(r[m]) for r in groups[key]])
            line.update({"%s_%s" % (m, k): st[k] for k in ("mean", "std", "min", "max")})
            line["%s_missing" % m] = len(groups[key]) - st["n"]
        out.append(line)
    with open(os.path.join(out_dir, "switch_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print("\n%-9s %-9s %4s | %-26s | %-26s | %-18s" % ("from MHz", "to MHz", "n", "cur_freq switch ms mean/max", "tegrastats switch ms mean/max", "write ms mean"))
    for line in out:
        print("%-9s %-9s %4d | %10.2f / %-12.2f | %10.2f / %-12.2f | %8.3f" % (
            line["from_mhz"], line["to_mhz"], line["n_reps"], line["cur_switch_ms_mean"], line["cur_switch_ms_max"],
            line["tegra_switch_ms_mean"], line["tegra_switch_ms_max"], line["write_total_ms_mean"]))
    missing = sum(num(r["cur_switch_ms"]) != num(r["cur_switch_ms"]) for r in rows)
    print("\nrepeats where cur_freq never reached the target within %.1f s: %d of %d" % (settings.SWITCH_POLL_TIMEOUT_S, missing, len(rows)))
    print("summary:", os.path.join(out_dir, "switch_summary.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the time estimate; touches nothing")
    ap.add_argument("--ignore-busy", action="store_true")
    ap.add_argument("--summary-only", metavar="DIR")
    args = ap.parse_args()
    if args.summary_only:
        return summarize(args.summary_only)
    reps = plan()
    print("%d clocks %s MHz -> %d ordered pairs x %d repeats = %d switches, about %.0f min"
          % (len(settings.FREQS_HZ), [f / 1e6 for f in settings.FREQS_HZ], len(reps) // settings.SWITCH_REPEATS,
             settings.SWITCH_REPEATS, len(reps), len(reps) * per_rep_s() / 60.0))
    if args.dry_run:
        return
    from dvfs.session import Session
    out_dir = os.path.join(settings.RESULTS_DIR, "switch_" + time.strftime("%Y%m%d_%H%M%S"))
    with Session(out_dir, ignore_busy=args.ignore_busy) as s:
        for i, (frm, to, rep) in enumerate(reps):
            rows, traces = [], []
            one_rep(s, i, frm, to, rep, rows, traces)
            append_csv(os.path.join(out_dir, "switch_raw.csv"), FIELDS, rows)
            append_csv(os.path.join(out_dir, "switch_tegra_trace.csv"), TRACE_FIELDS, traces)
            r = rows[0]
            print("[%d/%d] %g -> %g MHz: cur_freq %s ms, tegrastats %s ms, writes %s ms"
                  % (i + 1, len(reps), frm / 1e6, to / 1e6, r["cur_switch_ms"] or "never", r["tegra_switch_ms"] or "never", r["write_total_ms"]))
    summarize(out_dir)


if __name__ == "__main__":
    main()
