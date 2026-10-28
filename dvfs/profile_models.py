# Latency and power of every model at every pinned GPU clock and batch size. All four nets stay loaded.
# One run = (freq, model, batch): pin, warm up, then SUBRUNS subruns of exactly SUBRUN_BATCHES back-to-back batches
# (SUBRUN_GAP_S idle between them), then RUN_GAP_S of cooldown. Runs are in a seeded random order.
# Latency = h2d + forward + nms per batch (the profile.json definition). Power = tegrastats total_excl_sys5v
# integrated over exactly [first batch start, last batch end] of the subrun.
# Run:  sudo python3 -m dvfs.profile_models [--dry-run] [--resume DIR] [--ignore-busy]     (summary only: --summary-only DIR)
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import os
import random
import time

import numpy as np

from dvfs import settings, tegrastats
from dvfs.guard import SafetyAbort, hot_c

SUBRUN_FIELDS = ["order_index", "freq_hz", "model", "batch", "subrun", "n_batches", "t_start", "t_end", "duration_s",
                 "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms", "fwd_mean_ms",
                 "energy_per_batch_j", "energy_per_frame_j", "avg_power_w", "power_total_w", "power_gpu_w",
                 "gap_power_w", "n_power_samples", "cur_freq_end_hz", "gr3d_mhz_median", "t_start_c", "t_end_c",
                 "t_max_c", "timestamp"]
BATCH_FIELDS = ["freq_hz", "model", "batch", "subrun", "k", "t_start", "t_end", "h2d_ms", "forward_ms", "nms_ms",
                "total_ms", "n_boxes"]


def plan():
    items = [(f, m, b) for f in settings.FREQS_HZ for m in settings.QUEUES for b in settings.BATCHES]
    random.Random(settings.SEED).shuffle(items)
    return items


def run_s():
    gaps = (settings.SUBRUNS - 1) * settings.SUBRUN_GAP_S + settings.RUN_GAP_S
    return (settings.PIN_SETTLE_S + settings.WARMUP_S + settings.SUBRUNS * settings.SUBRUN_BATCHES * settings.EST_BATCH_S + gaps)


def batch_arrays(pool, k, b):
    """Different inputs on every call so the batch is not the same bytes each time (as in sched-harness 01_calibrate)."""
    return [pool[(k * b + i) % len(pool)] for i in range(b)]


def append_csv(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def median_mhz(sampler, t0, t1):
    vals = [p["gr3d_mhz"] for t, p, _ in sampler.window(t0, t1, pad=0) if t0 <= t <= t1 and p.get("gr3d_mhz") is not None]
    return float(np.median(vals)) if len(vals) >= 3 else None


def run_point(s, models, nets, buffers, pool, order_index, f, m, b, out_dir):
    guard, gpu, sampler = s.guard, s.gpu, s.sampler
    net, buf = nets[m], buffers[m]
    guard.set_target(None)
    gpu.pin(f)
    t, cur = gpu.wait_cur(f, settings.SWITCH_POLL_TIMEOUT_S)
    if t is None:
        raise RuntimeError("cur_freq stayed at %d after pinning %d" % (cur, f))
    guard.set_target(f)
    guard.sleep(settings.PIN_SETTLE_S)

    # warm-up at the new clock, discarded; tegrastats is also cross-checked against the pinned clock here
    t_w0 = time.monotonic()
    k = 0
    while k < settings.WARMUP_MIN_BATCHES or time.monotonic() - t_w0 < settings.WARMUP_S:
        guard.check()
        models.run_batch(net, buf, batch_arrays(pool, k, b))
        k += 1
    mhz = median_mhz(sampler, t_w0 + 0.5, time.monotonic())
    if settings.REQUIRE_TEGRASTATS_MATCH and (mhz is None or abs(mhz - f / 1e6) > settings.TEGRA_MATCH_TOL_MHZ):
        raise SafetyAbort("tegrastats GR3D_FREQ shows %s MHz under load while cur_freq and the request say %g MHz" % (mhz, f / 1e6))

    sub_rows = []
    for sub in range(settings.SUBRUNS):
        batches = []
        for k in range(settings.SUBRUN_BATCHES):
            guard.check()
            t0 = time.monotonic()
            h2d, fwd, nms, n_boxes = models.run_batch(net, buf, batch_arrays(pool, k, b))
            t1 = time.monotonic()
            batches.append((t0, t1, h2d, fwd, nms, n_boxes))
        t_start, t_end = batches[0][0], batches[-1][1]
        append_csv(os.path.join(out_dir, "batches.csv"), BATCH_FIELDS,
                   [{"freq_hz": f, "model": m, "batch": b, "subrun": sub, "k": i, "t_start": "%.6f" % x[0], "t_end": "%.6f" % x[1],
                     "h2d_ms": "%.4f" % x[2], "forward_ms": "%.4f" % x[3], "nms_ms": "%.4f" % x[4],
                     "total_ms": "%.4f" % (x[2] + x[3] + x[4]), "n_boxes": x[5]} for i, x in enumerate(batches)])
        guard.sleep(settings.POWER_TAIL_S)                 # a tegrastats sample past t_end for the interpolation

        total = np.array([x[2] + x[3] + x[4] for x in batches])
        samples = sampler.window(t_start, t_end, pad=0.5)
        times = [x[0] for x in samples]
        inside = [x for x in samples if t_start <= x[0] <= t_end]
        if len(samples) < 2:          # short windows (fast model, small batch) may hold no sample; the neighbours interpolate
            raise RuntimeError("fewer than 2 tegrastats samples around the %.2f s window" % (t_end - t_start))
        energy_j = tegrastats.energy_mj(times, [x[2][settings.PRIMARY_POWER] for x in samples], t_start, t_end) / 1000.0
        dur = t_end - t_start
        mhz_sub = median_mhz(sampler, t_start, t_end)
        if mhz_sub is not None and abs(mhz_sub - f / 1e6) > settings.TEGRA_MATCH_TOL_MHZ:
            raise SafetyAbort("tegrastats GR3D_FREQ median %.1f MHz during a subrun at %g MHz (clock dropped?)" % (mhz_sub, f / 1e6))
        temps = [hot_c(x[1]) for x in inside]
        temps = [x for x in temps if x is not None]
        before = [x for x in sampler.window(t_start - 1.0, t_start, pad=0) if x[0] <= t_start]
        t_before = hot_c(before[-1][1]) if before else None
        t_after = hot_c((inside or samples)[-1][1])
        row = {"order_index": order_index, "freq_hz": f, "model": m, "batch": b, "subrun": sub, "n_batches": len(batches),
               "t_start": "%.6f" % t_start, "t_end": "%.6f" % t_end, "duration_s": "%.4f" % dur,
               "lat_mean_ms": "%.4f" % total.mean(), "lat_p50_ms": "%.4f" % np.percentile(total, 50),
               "lat_p95_ms": "%.4f" % np.percentile(total, 95), "lat_p99_ms": "%.4f" % np.percentile(total, 99),
               "lat_max_ms": "%.4f" % total.max(), "fwd_mean_ms": "%.4f" % np.mean([x[3] for x in batches]),
               "energy_per_batch_j": "%.6f" % (energy_j / len(batches)), "energy_per_frame_j": "%.6f" % (energy_j / len(batches) / b),
               "avg_power_w": "%.4f" % (energy_j / dur),
               "power_total_w": "%.4f" % (tegrastats.mean_power(samples, "total", t_start, t_end) / 1000.0),
               "power_gpu_w": "%.4f" % (tegrastats.mean_power(samples, "gpu", t_start, t_end) / 1000.0),
               "n_power_samples": len(inside), "cur_freq_end_hz": gpu.cur_freq(),
               "gr3d_mhz_median": "" if mhz_sub is None else "%.1f" % mhz_sub,
               "t_start_c": "" if t_before is None else "%.1f" % t_before, "t_end_c": "" if t_after is None else "%.1f" % t_after,
               "t_max_c": "" if not temps else "%.1f" % max(temps), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        # idle gap with the clock still pinned; its power (first GAP_SKIP_S left out) is the idle power at this clock
        gap = settings.SUBRUN_GAP_S if sub < settings.SUBRUNS - 1 else settings.RUN_GAP_S
        guard.sleep(max(0.0, t_end + gap - time.monotonic()))
        g0, g1 = t_end + settings.GAP_SKIP_S, t_end + gap
        gap_samples = sampler.window(g0, g1, pad=0.5)
        row["gap_power_w"] = ("%.4f" % (tegrastats.mean_power(gap_samples, settings.PRIMARY_POWER, g0, g1) / 1000.0)
                              if len(gap_samples) >= 2 else "")
        sub_rows.append(row)
        append_csv(os.path.join(out_dir, "subruns.csv"), SUBRUN_FIELDS, [row])
        print("  sub %d: %d batches, lat %.1f ms, power %.2f W, %.3f J/frame, %.1f C max"
              % (sub, len(batches), total.mean(), energy_j / dur, energy_j / len(batches) / b, max(temps) if temps else float("nan")))
    guard.set_target(None)
    with sampler._lock:                      # keep RAM flat over hours; the raw gz log on disk has every sample
        cutoff = time.monotonic() - settings.KEEP_SAMPLES_S
        keep_from = next((i for i, x in enumerate(sampler.samples) if x[0] >= cutoff), len(sampler.samples))
        del sampler.samples[:keep_from]


def done_points(out_dir):
    path = os.path.join(out_dir, "subruns.csv")
    if not os.path.exists(path):
        return set()
    counts = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["freq_hz"]), r["model"], int(r["batch"]))
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n >= settings.SUBRUNS}


def num(x):
    return float(x) if x not in ("", None) else float("nan")


def summarize(out_dir):
    """One line per (freq, model, batch): latency weighted by batches, power weighted by duration."""
    with open(os.path.join(out_dir, "subruns.csv")) as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], int(r["batch"]), int(r["freq_hz"])), []).append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        nb = np.array([num(r["n_batches"]) for r in g])
        dur = np.array([num(r["duration_s"]) for r in g])
        lat = np.array([num(r["lat_mean_ms"]) for r in g])
        pw = np.array([num(r["avg_power_w"]) for r in g])
        gap = np.array([num(r["gap_power_w"]) for r in g])
        energy = pw * dur                                   # J per subrun
        out.append({"model": key[0], "batch": key[1], "freq_mhz": key[2] / 1e6, "n_subruns": len(g), "n_batches": int(nb.sum()),
                    "lat_mean_ms": float((lat * nb).sum() / nb.sum()), "lat_subrun_std_ms": float(lat.std()),
                    "lat_p50_ms": float(np.mean([num(r["lat_p50_ms"]) for r in g])),
                    "lat_p95_ms": float(np.mean([num(r["lat_p95_ms"]) for r in g])),
                    "lat_max_ms": float(max(num(r["lat_max_ms"]) for r in g)),
                    "avg_power_w": float(energy.sum() / dur.sum()), "power_subrun_std_w": float(pw.std()),
                    "energy_per_batch_j": float(energy.sum() / nb.sum()), "energy_per_frame_j": float(energy.sum() / nb.sum() / key[1]),
                    "idle_gap_power_w": float(np.nanmean(gap)) if not np.all(np.isnan(gap)) else float("nan"),
                    "t_max_c": float(np.nanmax([num(r["t_max_c"]) for r in g]))})
    with open(os.path.join(out_dir, "run_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print("\n%-5s %-5s %8s | %10s %10s | %8s %9s | %8s %6s" % ("model", "batch", "MHz", "lat mean", "lat p95", "power W", "J/frame", "idle W", "max C"))
    for r in out:
        print("%-5s %-5d %8.2f | %9.2f  %9.2f  | %8.3f %9.4f | %8.3f %6.1f" % (
            r["model"], r["batch"], r["freq_mhz"], r["lat_mean_ms"], r["lat_p95_ms"], r["avg_power_w"],
            r["energy_per_frame_j"], r["idle_gap_power_w"], r["t_max_c"]))
    print("\nsummary:", os.path.join(out_dir, "run_summary.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the time estimate; touches nothing")
    ap.add_argument("--resume", metavar="DIR", help="continue an earlier output folder, skipping finished runs")
    ap.add_argument("--ignore-busy", action="store_true")
    ap.add_argument("--summary-only", metavar="DIR")
    args = ap.parse_args()
    if args.summary_only:
        return summarize(args.summary_only)
    items = plan()
    print("%d clocks %s MHz x %d models x %d batch sizes %s = %d runs of %.0f s each: about %.1f h"
          % (len(settings.FREQS_HZ), [f / 1e6 for f in settings.FREQS_HZ], len(settings.QUEUES), len(settings.BATCHES),
             settings.BATCHES, len(items), run_s(), len(items) * run_s() / 3600.0))
    if args.dry_run:
        for i, (f, m, b) in enumerate(items[:12]):
            print("  %3d: %g MHz  yolov8%s  b=%d" % (i, f / 1e6, m, b))
        print("  ... (seed %d)" % settings.SEED)
        return
    from dvfs import inputs, models
    from dvfs.session import Session
    import torch
    out_dir = args.resume or os.path.join(settings.RESULTS_DIR, "models_" + time.strftime("%Y%m%d_%H%M%S"))
    done = done_points(out_dir)
    with Session(out_dir, ignore_busy=args.ignore_busy) as s:
        print("loading the four nets")
        pool, source = inputs.build_pool()
        nets = models.load_nets()
        buffers = {letter: inputs.BatchBuffer() for letter in settings.QUEUES}
        models.warm_up(nets, buffers, pool)
        torch.cuda.synchronize()
        todo = [(i, p) for i, p in enumerate(items) if p not in done]
        print("%d runs to do (%d already done)" % (len(todo), len(done)))
        for n, (i, (f, m, b)) in enumerate(todo):
            print("[%d/%d] order %d: %g MHz yolov8%s b=%d" % (n + 1, len(todo), i, f / 1e6, m, b))
            run_point(s, models, nets, buffers, pool, i, f, m, b, out_dir)
    summarize(out_dir)


if __name__ == "__main__":
    main()
