#!/usr/bin/env python3
"""One row per run of the DVFS main run -> analysis/dvfs_summary.csv, a seed-averaged table -> analysis/dvfs_by_policy.csv,
plus the validation checks.     Usage:  DEVICE_TAG=agx_xavier_jp511_main python3 05_dvfs_summary.py     (no board, no torch)
Per-run results are cached in analysis/cache/<run>.json (loading 500k-job logs is slow), so it can be stopped and rerun.
LIMIT=n processes at most n uncached runs per call. Only runs whose name starts with MIX (default light3) are read.
Run it only when the copy from the board is complete: a half-copied run folder gives wrong numbers."""
import glob
import json
import os
import time

import pandas as pd

import config
from common import metrics

MIX = os.environ.get("MIX", "light3")
LIMIT = int(os.environ.get("LIMIT", "1000"))
CACHE = os.path.join(config.ANALYSIS_DIR, "cache")
SHARD = os.environ.get("SHARD", "0/1")      # "i/n": this call takes every n-th uncached run starting at i (run n calls side by side)
BUDGET_S = float(os.environ.get("BUDGET_S", "1e9"))     # stop starting new runs after this many seconds
VERSION = 2          # bump when process() changes; older cache files are redone


def clock_stats(run_dir, info):
    """Busy-time share of every GPU clock and the time-weighted mean clock, over the measurement window."""
    b = pd.read_csv(os.path.join(run_dir, "batches.csv"))
    w0 = info["t0"] + info["warmup_s"]
    b = b[(b["t_start"] >= w0) & (b["t_start"] <= w0 + info["duration_s"])]
    dur = b["t_end"] - b["t_start"]
    share = dur.groupby(b["freq_mhz"]).sum() / dur.sum()
    return {str(int(f)): round(float(v), 4) for f, v in share.items()}, float((b["freq_mhz"] * dur).sum() / dur.sum())


def process(run_dir, profile):
    run = metrics.load_run(run_dir)
    info = run["info"]
    row = metrics.summarize(run, profile)
    shares, mean_mhz = clock_stats(run_dir, info)
    row["run"] = info["name"]
    row["policy_name"] = info["policy"]
    row["alpha"] = info["params"].get("alpha")
    row["fixed_mhz"] = info["params"].get("freq_mhz")
    row["arrivals_in_window"] = row["n_jobs"] + row["policy_drops"]
    row["drop_pct"] = row["policy_drops"] / row["arrivals_in_window"] * 100.0
    row["miss_pct"] = 100.0 - row["goodput_pct"]            # dropped or late, against the run's own deadline
    row["mean_clock_mhz"] = mean_mhz
    row["switches"] = info.get("clock_switches")
    row["switch_ms_mean"] = info.get("clock_switch_ms_mean")
    c = info.get("policy_counters") or {}
    row.update({"n_clean": c.get("clean"), "n_risky": c.get("risky"), "n_miss": c.get("miss"),
                "arrivals_hash": info.get("arrivals_hash"), "temp_end_c": info.get("temp_end_c")})
    row["energy_per_offered_mj"] = row["energy_j"] / row["arrivals_in_window"] * 1000.0
    checks = [[n, bool(ok), d] for n, ok, d in metrics.run_checks(run, profile)]
    return {"v": VERSION, "row": row, "checks": checks, "shares": shares}


def cached(path):
    if not os.path.exists(path):
        return False
    try:
        return json.load(open(path)).get("v") == VERSION
    except ValueError:
        return False            # half-written file


AGG = ["mean_power_w", "energy_per_good_req_mj", "drop_pct", "miss_pct", "lat_mean_ms", "lat_p99_ms", "mean_clock_mhz",
       "switches", "batch_mean_size", "throughput_rps"]


def merge(results):
    """Writes the per-run csv, the seed-averaged csv and the clock shares; prints the averaged table and the checks."""
    df = pd.DataFrame([r["row"] for r in results])
    df = df.sort_values(["slo_ms", "lam", "policy_name", "alpha", "fixed_mhz", "seed"]).reset_index(drop=True)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.ANALYSIS_DIR, "dvfs_summary.csv"), index=False)
    json.dump({r["row"]["run"]: r["shares"] for r in results}, open(os.path.join(config.ANALYSIS_DIR, "clock_shares.json"), "w"), indent=1)

    keys = ["slo_ms", "lam", "policy_name", "alpha", "fixed_mhz"]
    g = df.groupby(keys, dropna=False)
    by = g[AGG].mean()
    by.columns = [c + "_mean" for c in AGG]
    sd = g[AGG].std()
    for c in AGG:
        by[c + "_sd"] = sd[c]
    by["seeds"] = g["seed"].nunique()
    by = by.reset_index().sort_values(keys).reset_index(drop=True)
    by.to_csv(os.path.join(config.ANALYSIS_DIR, "dvfs_by_policy.csv"), index=False)

    show = by.copy()
    show["policy"] = [("alpha %g" % a) if p == "dvfs_alpha" else ("fixed %d" % f) for p, a, f in
                      zip(show["policy_name"], show["alpha"], show["fixed_mhz"].fillna(0))]
    cols = ["slo_ms", "lam", "policy", "seeds", "energy_per_good_req_mj_mean", "energy_per_good_req_mj_sd", "mean_power_w_mean",
            "drop_pct_mean", "miss_pct_mean", "lat_mean_ms_mean", "lat_p99_ms_mean", "mean_clock_mhz_mean", "switches_mean"]
    with pd.option_context("display.width", 260, "display.max_columns", 50, "display.max_rows", 500,
                           "display.float_format", "{:.2f}".format):
        print(show[cols].rename(columns=lambda c: c.replace("_mean", "").replace("energy_per_good_req_mj", "mJ_good")).to_string(index=False))
    print("\n%d runs, seeds present: %s" % (len(df), sorted(df["seed"].unique().tolist())))
    print("validation")
    failed = {}
    for r in results:
        for name, ok, detail in r["checks"]:
            if not ok:
                failed.setdefault(name, []).append((r["row"]["run"], detail))
    if not failed:
        print("  all per-run checks pass")
    for name, items in failed.items():
        print("  %s fails in %d of %d runs, e.g. %s | %s" % (name, len(items), len(results), items[0][0], items[0][1]))
    for (lam, seed), grp in df.groupby(["lam", "seed"]):
        n = grp["arrivals_hash"].nunique()
        print("  9 same jobs, load %g seed %d: %s (%d runs)" % (lam, seed, "ok" if n == 1 else "FAIL, %d different job lists" % n, len(grp)))
    return df, by


def main():
    profile = json.load(open(config.PROFILE_PATH))
    os.makedirs(CACHE, exist_ok=True)
    todo, done = [], 0
    for run_dir in sorted(glob.glob(os.path.join(config.RUNS_DIR, MIX + "_*"))):
        path = os.path.join(CACHE, os.path.basename(run_dir) + ".json")
        if cached(path):
            continue
        info_path = os.path.join(run_dir, "run.json")
        if not os.path.exists(info_path) or json.load(open(info_path)).get("status") != "done":
            print("SKIP (not done):", os.path.basename(run_dir))
            continue
        todo.append((run_dir, path))
    shard_i, shard_n = [int(x) for x in SHARD.split("/")]
    start = time.time()
    for run_dir, path in todo[shard_i::shard_n][:LIMIT]:
        if time.time() - start > BUDGET_S:
            break
        out = process(run_dir, profile)
        tmp = path + ".tmp"
        json.dump(out, open(tmp, "w"), default=float)
        os.replace(tmp, path)
        done += 1
        print("cached %d of %d: %s" % (done, len(todo), os.path.basename(run_dir)), flush=True)
    if len(todo) - done:
        print("%d runs were uncached at the start of this call, %d done here: run again" % (len(todo), done))
        return
    results = [r for r in (json.load(open(p)) for p in sorted(glob.glob(os.path.join(CACHE, MIX + "_*.json")))) if r.get("v") == VERSION]
    merge(results)


if __name__ == "__main__":
    main()
