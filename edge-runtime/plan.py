# Experiment runner.
#
#   python plan.py example > my_plan.json          an example plan to edit
#   python plan.py run my_plan.json [--dry-run]    every run of the plan, resumable
#   python plan.py visit --budget-min 90           the whole Jetson visit (section 12 of the spec)
#   python plan.py copyout                         one tar.gz of results/<device_tag>
#
# A plan (JSON): policies (each with an optional parameter grid), traces (paths or
# globs), slos_ms, input ("tensor"/"frames" or a list), prep ("arrival"/"executor" or a
# list), duration_s, warmup_s, repeats, cooldown_s, budget_min, plus runtime options
# (nms, b_max, profile_stat, power). Runs are expanded repeat > trace > SLO > input >
# prep > policy, and that order is the priority: with a budget, the tail is dropped.
# Each run is its own subprocess (a crash or OOM loses one run); a run whose folder
# already has a complete run.json is skipped.

import argparse
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import tarfile
import time

import config
from common import Profile, load_json, save_json, suggest_slos

HERE = config.HERE
COMPLETE = ("ok", "collapsed", "drain_timeout")

EXAMPLE = {
    "device_tag": None,
    "profile": None,
    "policies": [
        {"name": "fcfs_nobatch"}, {"name": "fcfs_batch"}, {"name": "lqf"},
        {"name": "triton", "grid": {"max_queue_delay_ms": config.TRITON_DELAYS_MS}},
        {"name": "edgeserving"}, {"name": "clockwork"}, {"name": "symphony"},
    ],
    "traces": ["data/traces/periodic_load*.csv"],
    "slos_ms": config.SLOS_MS,
    "input": "tensor",
    "prep": "arrival",
    "duration_s": config.RUN_DURATION_S,
    "warmup_s": config.WARMUP_S,
    "repeats": config.REPEATS,
    "cooldown_s": config.COOLDOWN_S,
    "budget_min": None,
    "nms": True,
    "b_max": config.B_MAX,
    "profile_stat": config.PROFILE_STAT,
    "power": "auto",
}


def as_list(x):
    return x if isinstance(x, list) else [x]


def param_tag(params):
    if not params:
        return ""
    return "-" + "-".join("%s%s" % (re.sub(r"[^a-z]", "", k)[:6], v) for k, v in sorted(params.items()))


def expand(plan):
    policies = []
    for p in plan["policies"]:
        grid = p.get("grid") or {}
        base = p.get("params") or {}
        keys = sorted(grid)
        for values in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
            params = dict(base, **dict(zip(keys, values)))
            policies.append((p["name"], params))
    traces = []
    for pattern in plan["traces"]:
        path = pattern if os.path.isabs(pattern) else os.path.join(HERE, pattern)
        traces.extend(sorted(glob.glob(path)) or [path])
    runs = []
    for r, trace, slo, inp, prep in itertools.product(range(plan.get("repeats", 1)), traces, plan["slos_ms"],
                                                     as_list(plan.get("input", "tensor")),
                                                     as_list(plan.get("prep", "arrival"))):
        for name, params in policies:
            run_id = "%s%s__%s__slo%g__%s-%s__r%d" % (name, param_tag(params), os.path.splitext(os.path.basename(trace))[0],
                                                     slo, inp, prep, r)
            runs.append({"run_id": run_id, "policy": name, "params": params, "trace": trace, "slo_ms": slo,
                         "input": inp, "prep": prep, "repeat": r})
    return runs


def trace_seconds(path, duration_s):
    meta_path = path + ".meta.json"
    length = load_json(meta_path).get("duration_s", duration_s) if os.path.exists(meta_path) else duration_s
    return min(length, duration_s) if duration_s else length


def estimate_run_s(run, plan):
    return trace_seconds(run["trace"], plan.get("duration_s")) + 5.0 + config.RUN_OVERHEAD_S + plan.get("cooldown_s", 0)


def is_complete(run_dir):
    path = os.path.join(run_dir, "run.json")
    if not os.path.exists(path):
        return False
    try:
        return load_json(path).get("status") in COMPLETE
    except ValueError:
        return False


def detect_tag():
    from power_profile import device_tag_no_torch
    return device_tag_no_torch()


def run_one(run, plan, device_dir, profile_path):
    out = os.path.join(device_dir, "runs", run["run_id"])
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "runtime.py"), "--policy", run["policy"],
           "--params", json.dumps(run["params"]), "--trace", run["trace"], "--slo-ms", str(run["slo_ms"]),
           "--profile", profile_path, "--input", run["input"], "--prep", run["prep"], "--out", out,
           "--warmup-s", str(plan.get("warmup_s", config.WARMUP_S)), "--b-max", str(plan.get("b_max", config.B_MAX)),
           "--profile-stat", plan.get("profile_stat", config.PROFILE_STAT), "--power", plan.get("power", "auto"),
           "--extra", json.dumps({"plan_run_id": run["run_id"], "repeat": run["repeat"]})]
    if plan.get("duration_s"):
        cmd += ["--duration", str(plan["duration_s"])]
    if not plan.get("nms", True):
        cmd.append("--no-nms")
    if plan.get("fake"):
        cmd.append("--fake")
    if plan.get("idle_s") is not None:
        cmd += ["--idle-s", str(plan["idle_s"])]
    timeout = trace_seconds(run["trace"], plan.get("duration_s")) + config.DRAIN_TIMEOUT_S + 300
    t0 = time.time()
    with open(os.path.join(out, "stdout.log"), "w") as log:
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return code, time.time() - t0, out


def run_plan(plan, device_tag=None, dry_run=False, deadline=None, log=print):
    tag = device_tag or plan.get("device_tag") or detect_tag()
    device_dir = config.device_dir(tag)
    profile_path = plan.get("profile") or os.path.join(device_dir, "profile.json")
    if not os.path.exists(profile_path):
        raise SystemExit("no profile at %s: run calibrate.py first" % profile_path)
    runs = expand(plan)
    todo = [r for r in runs if not is_complete(os.path.join(device_dir, "runs", r["run_id"]))]
    budget_s = plan.get("budget_min") * 60 if plan.get("budget_min") else None
    if deadline is not None:
        budget_s = min(budget_s or 1e18, deadline - time.time())
    est = [estimate_run_s(r, plan) for r in todo]
    log("%d runs in the plan, %d already done, %d to do: about %.0f min" % (
        len(runs), len(runs) - len(todo), len(todo), sum(est) / 60))
    if budget_s is not None:
        keep, used = [], 0.0
        for r, e in zip(todo, est):
            if used + e <= budget_s:
                keep.append(r)
                used += e
        dropped = [r["run_id"] for r in todo if r not in keep]
        if dropped:
            log("budget %.0f min: dropping %d lowest-priority runs:" % (budget_s / 60, len(dropped)))
            for d in dropped:
                log("  - " + d)
        todo = keep
    if dry_run:
        for r in todo:
            log("  " + r["run_id"])
        return []
    first_temp = None
    done = []
    for i, r in enumerate(todo, 1):
        if deadline is not None and time.time() + estimate_run_s(r, plan) > deadline:
            log("out of time before %s, stopping" % r["run_id"])
            break
        log("[%d/%d] %s" % (i, len(todo), r["run_id"]))
        code, secs, out = run_one(r, plan, device_dir, profile_path)
        status = load_json(os.path.join(out, "run.json")).get("status") if os.path.exists(os.path.join(out, "run.json")) else "FAILED (%s)" % code
        temp = load_json(os.path.join(out, "run.json")).get("temp_start_c") if os.path.exists(os.path.join(out, "run.json")) else None
        log("  %s in %.0f s%s" % (status, secs, "" if temp is None else ", start temp %.1f C" % temp))
        if temp is not None:
            if first_temp is None:
                first_temp = temp
            elif temp - first_temp > config.TEMP_WARN_C:
                log("  WARNING: started %.1f C hotter than the first run (%.1f C)" % (temp - first_temp, first_temp))
        done.append((r["run_id"], status))
        if i < len(todo):
            time.sleep(plan.get("cooldown_s", 0))
    return done


# ---------------------------------------------------------------------- visit
def copyout(tag):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.RESULTS_DIR, "%s_%s.tar.gz" % (tag, stamp))
    with tarfile.open(path, "w:gz") as tar:
        tar.add(config.device_dir(tag), arcname=tag)
        if os.path.isdir(config.TRACES_DIR):
            tar.add(config.TRACES_DIR, arcname=os.path.join(tag, "traces_used"))
    size = os.path.getsize(path) / 2 ** 20
    print("COPYOUT: %s (%.1f MB)" % (path, size))
    return path


def visit(args):
    t_start = time.time()
    deadline = t_start + args.budget_min * 60
    py = sys.executable

    def remaining_min():
        return (deadline - time.time()) / 60

    def step(name, cmd, check=False):
        print("\n=== %s (%.0f min left) ===" % (name, remaining_min()), flush=True)
        code = subprocess.call(cmd, cwd=HERE)
        if check and code != 0:
            raise SystemExit("%s failed (exit %s)" % (name, code))
        return code

    tag = args.device_tag or detect_tag()
    code = step("preflight", [py, "preflight.py", "--device-tag", tag])
    device_dir = config.device_dir(tag)
    if code != 0 and not args.force:
        raise SystemExit("preflight says NOT READY (see above); fix it or pass --force")

    profile_path = os.path.join(device_dir, "profile.json")
    if not os.path.exists(profile_path) or args.recalibrate:
        max_s = max(5.0, min(20.0, args.budget_min * 60 * 0.12 / (len(config.MODELS) * len(config.BATCH_SIZES))))
        step("calibration", [py, "calibrate.py", "--max-s", str(max_s), "--min-iters", "5", "--device-tag", tag],
             check=True)
    profile = Profile.load(profile_path)

    # SLO: 100 ms if the device can make sense of it, else the profile's suggestion
    if args.slo == "auto":
        slowest = max(profile.t(m, 1, "p95") for m in config.MODELS)
        slo = 100.0 if 100.0 >= 1.5 * slowest else float(suggest_slos(profile, config.MODELS)[1])
        if slo != 100.0:
            print("FLAG: 100 ms SLO is below 1.5x the slowest model at b=1 (%.0f ms p95); using %g ms" % (slowest, slo))
    else:
        slo = float(args.slo)

    trace_dir = os.path.join(config.TRACES_DIR, tag)
    if not glob.glob(os.path.join(trace_dir, "periodic_load*.csv")):
        print("\n=== traces ===")
        from traces import build_suite
        build_suite(profile, trace_dir, duration_s=args.run_duration + config.WARMUP_S)

    if not args.skip_power:
        power_min = args.power_min if args.power_min is not None else max(5.0, 0.15 * args.budget_min)
        power_min = min(power_min, remaining_min() - 10)
        if power_min >= 5 and not os.path.exists(os.path.join(device_dir, "power_profile", "summary.csv")):
            step("power profile (%.0f min)" % power_min, [py, "power_profile.py", "--budget-min", str(power_min),
                                                         "--yes", "--profile", profile_path])

    order = []
    for kind_load in ["periodic_load50", "periodic_load80", "periodic_load20", "sticky_load50", "poisson_load50"]:
        path = os.path.join(trace_dir, kind_load + ".csv")
        if os.path.exists(path):
            order.append(path)
    order += sorted(glob.glob(os.path.join(config.TRACES_DIR, "cityflow_*.csv")))
    policies = [{"name": n, "params": p} for n, p in config.VISIT_POLICIES]
    plan = {"device_tag": tag, "profile": profile_path, "policies": policies, "traces": order, "slos_ms": [slo],
            "input": args.input, "prep": "arrival", "duration_s": args.run_duration + config.WARMUP_S,
            "warmup_s": config.WARMUP_S, "repeats": 1, "cooldown_s": args.cooldown, "nms": True,
            "b_max": config.B_MAX, "profile_stat": config.PROFILE_STAT, "power": "auto"}
    save_json(os.path.join(device_dir, "visit_plan.json"), plan)
    print("\n=== policy runs (%.0f min left) ===" % remaining_min())
    run_plan(plan, tag, deadline=deadline - 120)

    try:
        import matplotlib  # noqa: F401
        if remaining_min() > 2:
            step("analysis", [py, "analyze.py", device_dir, "--no-timelines"])
    except ImportError:
        print("matplotlib missing: run analyze.py on the laptop")
    copyout(tag)
    print("visit took %.1f min" % ((time.time() - t_start) / 60))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a plan of edge-runtime experiments.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("example")
    p = sub.add_parser("run")
    p.add_argument("plan")
    p.add_argument("--device-tag")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("visit")
    p.add_argument("--budget-min", type=float, default=90)
    p.add_argument("--slo", default="auto", help="ms, or 'auto'")
    p.add_argument("--input", default="tensor", choices=["tensor", "frames"])
    p.add_argument("--run-duration", type=float, default=60.0, help="seconds of trace per run (after warmup)")
    p.add_argument("--cooldown", type=float, default=10.0)
    p.add_argument("--power-min", type=float, help="minutes for the power profile (default 15%% of the budget)")
    p.add_argument("--skip-power", action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--device-tag")
    p.add_argument("--force", action="store_true", help="go on even if preflight is not ready")
    p = sub.add_parser("copyout")
    p.add_argument("--device-tag")
    args = parser.parse_args(argv)

    if args.cmd == "example":
        print(json.dumps(EXAMPLE, indent=1))
    elif args.cmd == "run":
        run_plan(load_json(args.plan), args.device_tag, args.dry_run)
    elif args.cmd == "visit":
        visit(args)
    else:
        copyout(args.device_tag or detect_tag())


if __name__ == "__main__":
    main()
