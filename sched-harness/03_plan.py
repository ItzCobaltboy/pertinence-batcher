# Runs a grid of runs, each as its own 02_run.py subprocess (a crash kills one run, not the plan).
# Resumable: a run folder whose run.json says status "done" is skipped. Mode and clocks are set once here.
# Tiers are in config.py (ACTIVE_TIERS); runs are listed in priority order and BUDGET_MIN drops from the end.
# Run: sudo python3 03_plan.py
import sys
sys.dont_write_bytecode = True
import json
import os
import signal
import subprocess
import time

import config
from common import device, policies


def on_sigterm(signum, frame):
    # treat kill like Ctrl+C so the finally block still restores the mode and clocks
    raise KeyboardInterrupt()


def build_runs():
    """Priority order: tier first, then seed, mix, lam, policy. Duplicates between tiers appear once."""
    runs = []
    seen = set()
    for tier in sorted(config.ACTIVE_TIERS):
        spec = config.TIERS[tier]
        for seed in spec["seeds"]:
            for mix in spec["mixes"]:
                for lam in spec["lams"][mix]:
                    for name, params in spec["policies"]:
                        run_name = policies.run_name(mix, lam, name, params, seed)
                        if run_name in seen:
                            continue
                        seen.add(run_name)
                        runs.append({"name": run_name, "tier": tier, "mix": mix, "lam": lam, "policy": name,
                                     "params": params, "seed": seed})
    return runs


def is_done(run):
    path = os.path.join(config.RUNS_DIR, run["name"], "run.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return json.load(f).get("status") == "done"


def seconds_per_run():
    return config.SETUP_S_ESTIMATE + config.WARMUP_S + config.DURATION_S + config.DRAIN_S_ESTIMATE \
        + config.SAMPLER_TAIL_S + config.COOLDOWN_S


def apply_budget(runs):
    if config.BUDGET_MIN is None:
        return runs
    keep = int(config.BUDGET_MIN * 60 // seconds_per_run())
    if keep < len(runs):
        print("BUDGET_MIN=%d fits %d runs; dropping the %d lowest-priority ones:" % (config.BUDGET_MIN, keep, len(runs) - keep))
        for r in runs[keep:]:
            print("  dropped", r["name"])
    return runs[:keep]


def command_for(run):
    cmd = [sys.executable, os.path.join(config.HERE, "02_run.py"), "--policy", run["policy"], "--mix", run["mix"],
           "--lam", str(run["lam"]), "--seed", str(run["seed"]), "--session-managed",
           "--out", os.path.join(config.RUNS_DIR, run["name"])]
    for key, value in run["params"].items():
        cmd += ["--param", "%s=%s" % (key, value)]
    return cmd


def main():
    if os.geteuid() != 0:
        sys.exit("run with sudo")
    signal.signal(signal.SIGTERM, on_sigterm)
    runs = apply_budget(build_runs())
    todo = [r for r in runs if not is_done(r)]
    print("%d runs in the plan, %d already done, %d to go" % (len(runs), len(runs) - len(todo), len(todo)))
    print("estimated wall time: %.0f min (%.0f s per run)" % (len(todo) * seconds_per_run() / 60.0, seconds_per_run()))
    if not todo:
        return
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    session = device.start_session()
    failed = []
    child = None
    try:
        for i, run in enumerate(todo):
            print("\n=== run %d of %d: %s ===" % (i + 1, len(todo), run["name"]))
            child = subprocess.Popen(command_for(run))
            code = child.wait()
            child = None
            if code != 0:
                print("RUN FAILED (exit %d): %s" % (code, run["name"]))
                failed.append(run["name"])
            if i + 1 < len(todo):
                time.sleep(config.COOLDOWN_S)
    except KeyboardInterrupt:
        print("\ninterrupted; finished runs are kept, rerun the same command to resume")
        if child is not None:
            child.terminate()
            child.wait()
    finally:
        device.end_session(session)
    print("\nplan finished, %d failed%s" % (len(failed), ": " + ", ".join(failed) if failed else ""))


main()
