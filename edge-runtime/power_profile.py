# Power profile E_i(b): a port of yolo-analysis-a100/power_analysis/ that also runs
# on a Jetson (tegrastats) and fits in a time budget.
#
#   python power_profile.py --budget-min 45            pick grid + phase lengths that fit, then run
#   python power_profile.py --modes 0,2                 repeat under several nvpmodel modes (needs sudo)
#   python power_profile.py --aggregate <out_dir>       redo tables/plots from saved data
#
# Same phase structure as power_analysis:
#   1. one power sampler for the whole sweep (one continuous trace)
#   2. baseline with nothing loaded: this process never imports torch, so no CUDA context
#   3. every (model, b) in random order, each in its own `--worker` subprocess:
#      load model + real batch, one warmup forward, idle with the model loaded, then
#      run phases of back-to-back forwards (at least N forwards AND at least S seconds),
#      sync at the end of each, gaps between them. The worker only records time.time() marks.
#   4. the trace is cut with those marks afterwards.
# Forward only (no NMS), eager FP32, like power_analysis.
#
# Per (model, b): mean board power and every rail, energy per batch, energy per frame
# (total, and dynamic = (P - P_idle_loaded) x T / b), GPU clock and temperature.
# Power mode: nvpmodel -q and jetson_clocks --show are recorded; --set-mode / --lock-clocks /
# --modes only act if `sudo -n` works, and the result is recorded either way.

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

import config
from common import load_json, run_cmd, save_json

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------- environment
def sudo_ok():
    code, _ = run_cmd(["sudo", "-n", "true"], timeout=5)
    return code == 0


def mode_info():
    info = {}
    code, out = run_cmd(["nvpmodel", "-q"])
    info["nvpmodel_q"] = out.strip() if code == 0 else None
    code, out = run_cmd(["sudo", "-n", "jetson_clocks", "--show"], timeout=10)
    info["jetson_clocks_show"] = out.strip() if code == 0 else "unavailable (%s)" % (out.strip()[:200],)
    return info


def set_mode(mode):
    if not sudo_ok():
        return {"requested_mode": mode, "ok": False, "why": "sudo -n not available"}
    code, out = run_cmd(["sudo", "-n", "nvpmodel", "-m", str(mode)], timeout=60)
    time.sleep(3)
    return {"requested_mode": mode, "ok": code == 0, "output": out.strip()[-500:]}


def lock_clocks():
    if not sudo_ok():
        return {"ok": False, "why": "sudo -n not available"}
    code, out = run_cmd(["sudo", "-n", "jetson_clocks"], timeout=30)
    return {"ok": code == 0, "output": out.strip()[-500:]}


def device_tag_no_torch():
    from common import device_tag, is_jetson
    if is_jetson():
        return device_tag()
    code, out = run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return device_tag(out.strip().splitlines()[0] if code == 0 and out.strip() else None)


# ---------------------------------------------------------------------- plan
def estimate_s(grid, models, idle_s, phases, phase_s, min_fwd, gap_s, t_ms, startup_s=15.0):
    total = config.POWER_BASELINE_S
    for m in models:
        for b in grid:
            run = max(phase_s, min_fwd * t_ms.get((m, b), 0.0) / 1000.0)
            total += startup_s + idle_s + phases * (run + gap_s)
    return total


def choose_plan(budget_min, models, sizes, t_ms):
    """First plan (most thorough first) whose estimate fits the budget."""
    small = [b for b in sizes if b in (1, 4, 16)] or sizes[:3]
    tiny = [b for b in sizes if b in (1, 8)] or sizes[:2]
    candidates = [
        (sizes, config.POWER_IDLE_S, config.POWER_RUN_PHASES, config.POWER_PHASE_MIN_S, config.POWER_PHASE_MIN_FORWARDS),
        (sizes, 10.0, 3, 6.0, 20),
        (small, 10.0, 3, 6.0, 20),
        (tiny, 8.0, 2, 5.0, 10),
    ]
    for grid, idle_s, phases, phase_s, min_fwd in candidates:
        est = estimate_s(grid, models, idle_s, phases, phase_s, min_fwd, config.POWER_GAP_S, t_ms)
        if budget_min is None or est <= budget_min * 60:
            return {"grid": grid, "idle_s": idle_s, "phases": phases, "phase_min_s": phase_s,
                    "phase_min_forwards": min_fwd, "estimate_s": est}
    grid, idle_s, phases, phase_s, min_fwd = candidates[-1]
    est = estimate_s(grid, models, idle_s, phases, phase_s, min_fwd, config.POWER_GAP_S, t_ms)
    return {"grid": grid, "idle_s": idle_s, "phases": phases, "phase_min_s": phase_s,
            "phase_min_forwards": min_fwd, "estimate_s": est, "over_budget": True}


# ---------------------------------------------------------------------- worker (subprocess)
def worker(job):
    import torch
    from inputs import TensorInputs
    from models import Runner
    marks = []

    def mark(label):
        marks.append([label, time.time()])

    t0 = time.time()
    runner = Runner([job["model"]], nms=False)
    batch = TensorInputs(job["frames_dir"], runner.device).sample_batch(job["batch"])
    runner.forward(job["model"], batch)
    load_s = time.time() - t0
    mark("idle_start")
    time.sleep(job["idle_s"])
    mark("idle_end")
    phases = []
    net = runner.nets[job["model"]]
    with torch.no_grad():
        for k in range(job["phases"]):
            mark("run%d_start" % k)
            start = time.time()
            n = 0
            while n < job["phase_min_forwards"] or time.time() - start < job["phase_min_s"]:
                net(batch)
                n += 1
                if n % 10 == 0:
                    runner.sync()        # keep the host from running far ahead of the GPU
            runner.sync()
            end = time.time()
            mark("run%d_end" % k)
            phases.append({"phase": k, "start": start, "end": end, "runtime_s": end - start, "forwards": n})
            time.sleep(job["gap_s"])
    save_json(job["out"], {"model": job["model"], "batch": job["batch"], "load_s": load_s,
                           "marks": marks, "phases": phases, "device": runner.info()})


# ---------------------------------------------------------------------- sweep
def sweep(out_dir, plan, models, backend, seed, frames_dir):
    import power
    os.makedirs(os.path.join(out_dir, "combos"), exist_ok=True)
    combos = [(m, b) for m in models for b in plan["grid"]]
    random.Random(seed).shuffle(combos)
    save_json(os.path.join(out_dir, "plan.json"), dict(plan, models=models, order=combos, mode=mode_info()))
    sampler = power.make_sampler(backend)
    trace = os.path.join(out_dir, "power.csv")
    sampler.start(trace)
    try:
        time.sleep(2)
        base = {"start": time.time()}
        time.sleep(config.POWER_BASELINE_S)
        base["end"] = time.time()
        save_json(os.path.join(out_dir, "baseline.json"), base)
        for i, (m, b) in enumerate(combos, 1):
            name = "%s_bs%d" % (m, b)
            path = os.path.join(out_dir, "combos", name + ".json")
            if os.path.exists(path):
                continue
            print("[%d/%d] %s b=%d" % (i, len(combos), m, b), flush=True)
            job = {"model": m, "batch": b, "out": path, "frames_dir": frames_dir, "idle_s": plan["idle_s"],
                   "phases": plan["phases"], "phase_min_s": plan["phase_min_s"],
                   "phase_min_forwards": plan["phase_min_forwards"], "gap_s": config.POWER_GAP_S}
            with open(os.path.join(out_dir, "combos", name + ".log"), "w") as log:
                try:
                    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(job)],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                    if r.returncode != 0:
                        print("  FAILED (exit %d), see the .log" % r.returncode)
                except subprocess.TimeoutExpired:
                    print("  TIMEOUT")
        time.sleep(sampler.tail_s)
    finally:
        sampler.stop()


# ---------------------------------------------------------------------- aggregate
def aggregate(out_dir):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import power

    pw = power.read(os.path.join(out_dir, "power.csv"))
    t, p = pw["t"].values, pw["power_w"].values
    base = load_json(os.path.join(out_dir, "baseline.json"))
    baseline_w = power.mean_power(t, p, base["start"], base["end"])
    rails = [c for c in pw.columns if c.startswith("rail_")]
    clock_col = next((c for c in ("gpu_freq_mhz", "sm_clock_mhz") if c in pw), None)

    def window_mean(col, a, b):
        inside = pw[(pw["t"] >= a) & (pw["t"] <= b)]
        return float(inside[col].mean()) if len(inside) and col in inside else float("nan")

    rows, phase_rows = [], []
    for path in sorted(glob.glob(os.path.join(out_dir, "combos", "*.json"))):
        c = load_json(path)
        marks = dict((k, v) for k, v in c["marks"])
        idle_w = power.mean_power(t, p, marks["idle_start"], marks["idle_end"])
        per = []
        for ph in c["phases"]:
            pw_mean = power.mean_power(t, p, ph["start"], ph["end"])
            r = {"model": c["model"], "batch": c["batch"], "phase": ph["phase"], "runtime_s": ph["runtime_s"],
                 "forwards": ph["forwards"], "power_w": pw_mean, "energy_j": pw_mean * ph["runtime_s"],
                 "samples": int(((t >= ph["start"]) & (t <= ph["end"])).sum())}
            for rail in rails:
                r[rail] = window_mean(rail, ph["start"], ph["end"])
            if clock_col:
                r["gpu_clock_mhz"] = window_mean(clock_col, ph["start"], ph["end"])
            if "temp_c" in pw:
                r["temp_c"] = window_mean("temp_c", ph["start"], ph["end"])
            per.append(r)
        phase_rows.extend(per)
        ph = pd.DataFrame(per)
        run_w = float(ph["power_w"].mean())
        ms_per_batch = float((ph["runtime_s"] / ph["forwards"]).mean()) * 1000.0
        e_batch = run_w * ms_per_batch / 1000.0
        row = {"model": c["model"], "batch": c["batch"], "load_s": c["load_s"], "baseline_w": baseline_w,
               "idle_loaded_w": idle_w, "run_w": run_w, "ms_per_batch": ms_per_batch,
               "energy_per_batch_j": e_batch, "energy_per_frame_j": e_batch / c["batch"],
               "dyn_energy_per_frame_j": (run_w - idle_w) * ms_per_batch / 1000.0 / c["batch"],
               "samples_per_phase": float(ph["samples"].mean())}
        for col in rails + ["gpu_clock_mhz", "temp_c"]:
            if col in ph:
                row[col] = float(ph[col].mean())
        rows.append(row)
    if not rows:
        print("no combos in", out_dir)
        return None
    summary = pd.DataFrame(rows).sort_values(["model", "batch"])
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.DataFrame(phase_rows).to_csv(os.path.join(out_dir, "phases.csv"), index=False)

    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for m, g in summary.groupby("model"):
        axes[0].plot(g["batch"], g["run_w"], marker="o", label=m)
        axes[1].plot(g["batch"], g["energy_per_frame_j"], marker="o", label=m)
        axes[2].plot(g["batch"], g["dyn_energy_per_frame_j"], marker="o", label=m)
    axes[0].axhline(baseline_w, color="black", linestyle="--", linewidth=1, label="baseline")
    for ax, title in zip(axes, ["board power while running, W", "energy per frame, J",
                                "dynamic energy per frame (P - P_idle_loaded), J"]):
        ax.set_xlabel("batch size")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "power_energy_vs_b.png"), dpi=110)
    plt.close(fig)
    fig = plt.figure(figsize=(16, 4))
    plt.plot(t - t[0], p, linewidth=0.5)
    plt.xlabel("seconds")
    plt.ylabel("board power, W")
    plt.title("whole sweep")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "trace.png"), dpi=100)
    plt.close(fig)
    print("%d combos -> %s/summary.csv (baseline %.2f W)" % (len(summary), out_dir, baseline_w))
    return summary


# ---------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description="Power profile E_i(b) (port of power_analysis).")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--aggregate", metavar="DIR")
    parser.add_argument("--budget-min", type=float)
    parser.add_argument("--models", default=",".join(config.MODELS))
    parser.add_argument("--batch-sizes", default=",".join(str(b) for b in config.BATCH_SIZES))
    parser.add_argument("--profile", help="profile.json to size phases (default results/<tag>/profile.json)")
    parser.add_argument("--power", default="auto")
    parser.add_argument("--frames-dir", default=config.FRAMES_DIR)
    parser.add_argument("--set-mode", type=int)
    parser.add_argument("--lock-clocks", action="store_true")
    parser.add_argument("--modes", help="comma-separated nvpmodel modes to repeat the sweep under")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="default results/<device_tag>/power_profile")
    parser.add_argument("--yes", action="store_true", help="don't wait 5 s after printing the estimate")
    args = parser.parse_args(argv)

    if args.worker:
        worker(json.loads(args.worker))
        return
    if args.aggregate:
        aggregate(args.aggregate)
        return

    tag = device_tag_no_torch()
    out = args.out or os.path.join(config.device_dir(tag), "power_profile")
    models = args.models.split(",")
    sizes = [int(b) for b in args.batch_sizes.split(",")]
    prof_path = args.profile or os.path.join(config.device_dir(tag), "profile.json")
    t_ms = {}
    if os.path.exists(prof_path):
        from common import Profile
        prof = Profile.load(prof_path, part="forward")
        sizes = [b for b in sizes if b in prof.batch_sizes] or sizes
        t_ms = {(m, b): prof.t(m, b) for m in models for b in sizes if m in prof.models}
    else:
        print("no %s: phase lengths assume fast forwards (run calibrate.py first for a real estimate)" % prof_path)

    modes = [int(x) for x in args.modes.split(",")] if args.modes else [None]
    budget_each = args.budget_min / len(modes) if args.budget_min else None
    plan = choose_plan(budget_each, models, sizes, t_ms)
    print("plan: b in %s, idle %.0f s, %d phases of >= %.0f s / %d forwards; estimate %.1f min x %d mode(s)%s" % (
        plan["grid"], plan["idle_s"], plan["phases"], plan["phase_min_s"], plan["phase_min_forwards"],
        plan["estimate_s"] / 60, len(modes), "  (OVER BUDGET)" if plan.get("over_budget") else ""))
    if not args.yes:
        time.sleep(5)

    record = {"device_tag": tag, "sudo": sudo_ok(), "before": mode_info()}
    if args.set_mode is not None:
        record["set_mode"] = set_mode(args.set_mode)
    if args.lock_clocks:
        record["lock_clocks"] = lock_clocks()
    for mode in modes:
        sub = out
        if mode is not None:
            record.setdefault("modes", {})[mode] = set_mode(mode)
            if not record["modes"][mode]["ok"]:
                print("mode %s: %s, skipped" % (mode, record["modes"][mode].get("why", "failed")))
                continue
            sub = os.path.join(out, "mode_%d" % mode)
        os.makedirs(sub, exist_ok=True)
        save_json(os.path.join(sub, "mode.json"), dict(record, during=mode_info()))
        sweep(sub, plan, models, args.power, args.seed, args.frames_dir)
        aggregate(sub)
    save_json(os.path.join(out, "power_profile_run.json"), dict(record, after=mode_info(), plan=plan))


if __name__ == "__main__":
    main()
