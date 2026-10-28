# Reads one run folder and computes latency, energy and the nine validation checks.
# numpy only (no torch, no pandas) so the smoke run on the board and 04_analyze.py share the same code.
import csv
import gzip
import json
import os

import numpy as np

import config
from common import profile as profile_lib
from common import tegrastats


# ---------- loading ----------

def read_table(path):
    """CSV -> {column: float array}; a column that is not numeric stays a list of strings."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        names = reader.fieldnames
    table = {}
    for name in names:
        values = [r[name] for r in rows]
        try:
            table[name] = np.array([float(v) if v != "" else float("nan") for v in values], dtype=float)
        except ValueError:
            table[name] = values
    return table


def load_run(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        info = json.load(f)
    run = {"dir": run_dir, "info": info}
    for name in ["requests", "batches", "waits", "power", "arrivals"]:
        run[name] = read_table(os.path.join(run_dir, name + ".csv"))
    return run


# ---------- latency ----------

def stats(x):
    if len(x) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(x)), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(np.max(x))}


def window_jobs(run):
    """Mask of served jobs that arrived in the measurement window, plus t_first and t_last (spec section 9)."""
    req = run["requests"]
    in_window = req["in_measurement_window"] == 1
    served = in_window & (req["dropped"] == 0) & ~np.isnan(req["t_batch_end"])
    t_first = float(req["t_arrival_target"][in_window].min())
    t_last = float(req["t_batch_end"][served].max())
    return served, t_first, t_last


def latencies_ms(run, mask):
    req = run["requests"]
    return (req["t_batch_end"][mask] - req["t_arrival_target"][mask]) * 1000.0


# ---------- energy ----------

def energy_j(power, key, t0, t1):
    return tegrastats.energy_mj(power["t_monotonic"], power[config.POWER_COLS[key]], t0, t1) / 1000.0


def busy_idle_j(power, key, batches, t0, t1):
    """Energy while a batch is on the GPU vs the rest of [t0, t1], each integrated separately."""
    order = np.argsort(batches["t_start"])
    cursor = t0
    busy = 0.0
    idle = 0.0
    for s, e in zip(batches["t_start"][order], batches["t_end"][order]):
        s = max(s, t0)
        e = min(e, t1)
        if e <= cursor:
            continue
        s = max(s, cursor)
        if s > cursor:
            idle += energy_j(power, key, cursor, s)
        busy += energy_j(power, key, s, e)
        cursor = e
    if t1 > cursor:
        idle += energy_j(power, key, cursor, t1)
    return busy, idle


def busy_fraction(batches, w0, w1):
    s = np.clip(batches["t_start"], w0, w1)
    e = np.clip(batches["t_end"], w0, w1)
    return float(np.sum(e - s) / (w1 - w0))


def power_from_raw_log(path, key):
    """Recompute the power curve straight from the raw tegrastats log (independent of power.csv)."""
    times = []
    values = []
    with gzip.open(path, "rt") as f:
        for line in f:
            t_text, text = line.rstrip("\n").split(" ", 1)
            p = tegrastats.powers(tegrastats.parse_line(text), config.RAILS)
            if p is not None:
                times.append(float(t_text))
                values.append(p[key])
    return times, values


# ---------- summary of one run ----------

def summarize(run, profile):
    info = run["info"]
    req = run["requests"]
    mask, t_first, t_last = window_jobs(run)
    w0 = info["t0"] + info["warmup_s"]
    w1 = w0 + info["duration_s"]
    lat = latencies_ms(run, mask)
    s = stats(lat)
    n = int(mask.sum())
    row = {"mix": info["mix"], "lam": round(info["lambda_total"], 1), "nominal_load": info["lambda_total"] / info["lambda_cap"], "policy": info["policy"], "params": info["policy_label"],
           "seed": info["seed"], "status": info["status"], "flags": ";".join(info.get("flags", [])),
           "n_jobs": n, "drops": int(np.count_nonzero(req["dropped"])),
           "lat_mean_ms": s["mean"], "lat_p50_ms": s["p50"], "lat_p95_ms": s["p95"], "lat_p99_ms": s["p99"],
           "lat_max_ms": s["max"],
           "queue_wait_mean_ms": float(np.mean((req["t_batch_start"][mask] - req["t_enqueued"][mask]) * 1000.0)),
           "service_mean_ms": float(np.mean((req["t_batch_end"][mask] - req["t_batch_start"][mask]) * 1000.0))}
    in_win = req["in_measurement_window"] == 1
    n_arr = int(in_win.sum())
    lost = int(np.sum(in_win & (req["dropped"] > 0)))           # cancelled or overflowed: never served, so a miss at every deadline
    row["policy_drops"] = int(np.sum(in_win & (req["dropped"] == 2)))
    for slo in config.SLO_MS_GRID:
        row["viol_pct_%d" % slo] = (float(np.sum(lat > slo)) + lost) / n_arr * 100.0
    for q, letter in enumerate(config.QUEUES):
        row["p99_%s_ms" % letter] = stats(lat[req["queue"][mask] == q])["p99"]
    span = t_last - t_first
    row.update({"t_first": t_first, "t_last": t_last, "span_s": span, "drain_s": t_last - w1})
    for key in config.POWER_COLS:
        row["energy_j_" + key] = energy_j(run["power"], key, t_first, t_last)
    primary = config.PRIMARY_POWER
    busy, idle = busy_idle_j(run["power"], primary, run["batches"], t_first, t_last)
    total = row["energy_j_" + primary]
    row.update({"primary_power": primary, "energy_j": total, "mean_power_w": total / span,
                "busy_j": busy, "idle_j": idle, "busy_share": busy / total if total else float("nan"),
                "dynamic_j": total - profile["p_idle_loaded_mw"][primary] * span / 1000.0,
                "energy_per_req_mj": total * 1000.0 / n,
                "throughput_rps": n / info["duration_s"],
                "gpu_busy_frac": busy_fraction(run["batches"], w0, w1),
                "decide_ms_mean": info.get("decide_ms_mean"),
                "batch_mean_size": float(np.mean(run["batches"]["b"]))})
    own = info["params"].get("slo_ms")              # the deadline the policy itself was run with (edgeserving, clockwork)
    good = int(np.sum(lat <= own)) if own is not None else None
    row["slo_ms"] = float(own) if own is not None else float("nan")
    row["goodput_pct"] = good / n_arr * 100.0 if own is not None else float("nan")
    row["energy_per_good_req_mj"] = (total * 1000.0 / good if good else float("nan")) if own is not None else float("nan")
    return row


# ---------- validation checks (spec section 9) ----------
# each returns (name, ok, detail)

def check_counts(run):
    req = run["requests"]
    expected = len(run["arrivals"]["job_id"])
    unfinished = int(np.sum(np.isnan(req["t_batch_end"]) & (req["dropped"] == 0)))
    ok = len(req["job_id"]) == expected and unfinished == 0
    return ("1 every job finished or dropped", ok,
            "%d logged of %d generated, %d unfinished, %d dropped" % (len(req["job_id"]), expected, unfinished, int(np.count_nonzero(req["dropped"]))))


def check_no_overlap(run):
    b = run["batches"]
    order = np.argsort(b["t_start"])
    starts = b["t_start"][order]
    ends = b["t_end"][order]
    overlap = starts[1:] - ends[:-1]
    worst = float(overlap.min()) if len(overlap) else 0.0
    return ("2 batches do not overlap", worst >= -1e-6, "smallest gap between batches %.3f ms" % (worst * 1000.0))


def check_service_times(run, profile):
    b = run["batches"]
    worst = (0.0, "")
    for q, letter in enumerate(config.QUEUES):
        for size in sorted(set(b["b"][b["queue"] == q].astype(int))):
            sel = (b["queue"] == q) & (b["b"] == size)
            if sel.sum() < 5:
                continue
            measured = float(np.mean((b["t_end"][sel] - b["t_start"][sel]) * 1000.0))
            expected = profile_lib.service_ms(profile, letter, size, config.PROFILE_STAT)
            err = abs(measured - expected) / expected
            if err >= worst[0]:
                worst = (err, "%s b=%d: %.1f ms in run vs %.1f ms calibrated" % (letter, size, measured, expected))
    return ("3 in-run service time vs calibration", worst[0] <= config.CHECK_SERVICE_TOL,
            "worst offender %.1f%% (%s)" % (worst[0] * 100.0, worst[1]))


def check_little(run):
    """L (time-average jobs in system over the window, from the log) against throughput x mean latency."""
    info = run["info"]
    req = run["requests"]
    w0 = info["t0"] + info["warmup_s"]
    w1 = w0 + info["duration_s"]
    arrive = req["t_arrival_target"]
    leave = req["t_batch_end"]
    done = ~np.isnan(leave)
    overlap = np.clip(leave[done], w0, w1) - np.clip(arrive[done], w0, w1)
    big_l = float(overlap.sum() / (w1 - w0))
    departed = done.copy()
    departed[done] = (leave[done] >= w0) & (leave[done] <= w1)
    throughput = departed.sum() / (w1 - w0)
    mean_latency = float(np.mean(leave[departed] - arrive[departed]))
    expected = throughput * mean_latency
    err = abs(big_l - expected) / expected if expected else float("nan")
    return ("4 Little's law", err <= config.CHECK_LITTLE_TOL,
            "L=%.3f vs throughput x latency=%.3f (%.1f%% apart)" % (big_l, expected, err * 100.0))


def check_arrival_rates(run):
    info = run["info"]
    req = run["requests"]
    window = req["in_measurement_window"] == 1
    worst = 0.0
    detail = []
    for q, rate in enumerate(info["rates"]):
        count = int(np.sum(window & (req["queue"] == q)))
        err = abs(count / info["duration_s"] - rate) / rate
        # Poisson counts scatter by 1/sqrt(N) on their own; allow 3 sigma beyond the 5% floor
        tol = max(config.CHECK_RATE_TOL, 3.0 / np.sqrt(max(count, 1)))
        worst = max(worst, err / tol)
        detail.append("%s %.1f%% (tol %.1f%%)" % (config.QUEUES[q], err * 100.0, tol * 100.0))
    return ("5 arrival rate per queue vs lambda_i", worst <= 1.0, ", ".join(detail))


def check_lateness(run):
    p99 = stats(run["requests"]["arrival_lateness_ms"])["p99"]
    return ("6 arrival lateness p99", p99 < config.LATENESS_P99_FLAG_MS, "p99 %.2f ms (limit %.1f)" % (p99, config.LATENESS_P99_FLAG_MS))


def check_power_trace(run):
    p = run["power"]
    t = p["t_monotonic"]
    interval = config.TEGRASTATS_INTERVAL_MS / 1000.0
    _, t_first, t_last = window_jobs(run)
    inside = t[(t >= t_first) & (t <= t_last)]
    if len(inside) < 2:
        return ("7 tegrastats cadence and coverage", False, "fewer than 2 power samples inside [t_first, t_last]")
    cadence = (len(inside) - 1) / (inside[-1] - inside[0])
    cadence_err = abs(cadence - 1.0 / interval) / (1.0 / interval)
    gaps = np.diff(t[(t >= t_first - 1.0) & (t <= t_last + 1.0)])
    max_gap = float(gaps.max())
    covered = t[0] <= t_first + config.CHECK_GAP_SAMPLES * interval and t[-1] >= t_last - config.CHECK_GAP_SAMPLES * interval
    ok = cadence_err <= config.CHECK_CADENCE_TOL and max_gap <= config.CHECK_GAP_SAMPLES * interval and covered
    return ("7 tegrastats cadence and coverage", ok,
            "%.2f Hz (asked %.2f), longest gap %.0f ms, covers span: %s" % (cadence, 1.0 / interval, max_gap * 1000.0, covered))


def check_energy_books(run):
    key = config.PRIMARY_POWER
    _, t_first, t_last = window_jobs(run)
    total = energy_j(run["power"], key, t_first, t_last)
    busy, idle = busy_idle_j(run["power"], key, run["batches"], t_first, t_last)
    books_err = abs(busy + idle - total) / total
    raw_path = os.path.join(run["dir"], "power_raw.log.gz")
    times, values = power_from_raw_log(raw_path, key)
    raw_total = tegrastats.energy_mj(times, values, t_first, t_last) / 1000.0
    raw_err = abs(raw_total - total) / total
    ok = books_err <= config.CHECK_ENERGY_TOL and raw_err <= config.CHECK_ENERGY_TOL
    return ("8 energy bookkeeping", ok,
            "busy+idle vs total %.2f%% apart; power.csv vs raw log %.2f%% apart (total %.1f J)" % (books_err * 100.0, raw_err * 100.0, total))


def run_checks(run, profile):
    """Checks 1 to 8 on one run. Check 9 compares runs, see check_same_jobs."""
    return [check_counts(run), check_no_overlap(run), check_service_times(run, profile), check_little(run),
            check_arrival_rates(run), check_lateness(run), check_power_trace(run), check_energy_books(run)]


def check_same_jobs(runs):
    """Check 9 for one (mix, lam, seed) group: identical arrival list and identical job set across policies."""
    hashes = set(r["info"]["arrivals_hash"] for r in runs)
    counts = set(int(np.sum(r["requests"]["in_measurement_window"] == 1)) for r in runs)    # arrivals, not served: policies may drop
    firsts = set(round(window_jobs(r)[1] - r["info"]["t0"], 3) for r in runs)
    ok = len(hashes) == 1 and len(counts) == 1 and len(firsts) == 1
    return ("9 same job set across policies", ok,
            "%d runs, %d distinct arrival hashes, job counts %s" % (len(runs), len(hashes), sorted(counts)))
