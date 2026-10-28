# ONE run: one policy, one load, one mix, one seed. Poisson arrivals feed four queues, the policy decides,
# the GPU serves one batch at a time, tegrastats records power on the same clock. Writes the run folder.
# Run: sudo python3 02_run.py --smoke
#      sudo python3 02_run.py --policy triton --param delay_ms=5 --mix skewed --lam 16 --seed 0
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import datetime
import gzip
import json
import os
import shutil
import subprocess
import time

import torch

import config
from common import device, inputs, metrics, models, policies, powermode, profile as profile_lib, runtime, tegrastats, traffic


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="20 s run, lqf, lam 11 req/s, balanced, then checks 1 to 8")
    ap.add_argument("--policy", default=config.SMOKE_POLICY)
    ap.add_argument("--param", action="append", default=[], help="policy parameter as key=value, repeatable")
    ap.add_argument("--mix", default=config.SMOKE_MIX)
    ap.add_argument("--lam", type=float, default=config.SMOKE_LAM, help="total arrival rate, req/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="run folder (default: results/<tag>/runs/<name>)")
    ap.add_argument("--session-managed", action="store_true", help="03_plan.py already set the mode and clocks")
    return ap.parse_args()


def parse_params(items):
    params = {}
    for item in items:
        key, value = item.split("=")
        params[key] = float(value) if "." in value else int(value)
    return params


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"        # git is not installed on the board


def file_sha1(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def start_sampler(run_dir):
    sampler = tegrastats.Sampler(config.TEGRASTATS_INTERVAL_MS, config.RAILS, os.path.join(run_dir, "power_raw.log.gz"))
    sampler.start()
    for _ in range(30):
        if sampler.samples:
            return sampler
        time.sleep(0.1)
    sampler.stop()
    raise RuntimeError("no tegrastats samples with rails after 3 s (not root?)")


# ---------- writing logs ----------

def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for row in rows:
            w.writerow(row)


def blank(x, fmt):
    return "" if x is None else fmt % x


def write_logs(run_dir, rt, sampler):
    write_csv(os.path.join(run_dir, "requests.csv"),
              ["job_id", "queue", "pool_index", "t_arrival_target", "t_enqueued", "t_batch_start", "t_batch_end",
               "batch_id", "batch_size", "arrival_lateness_ms", "in_measurement_window", "dropped"],
              [[j.job_id, j.queue, j.pool_index, "%.6f" % j.t_target, blank(j.t_enq, "%.6f"), blank(j.t_start, "%.6f"),
                blank(j.t_end, "%.6f"), "" if j.batch_id is None else j.batch_id, "" if j.batch_size is None else j.batch_size,
                blank(j.lateness_ms, "%.3f"), j.in_window, j.dropped] for j in rt.jobs])
    fields = ["batch_id", "queue", "b", "t_start", "t_end", "h2d_ms", "forward_ms", "nms_ms", "n_boxes", "decide_ms", "queue_len_before"]
    write_csv(os.path.join(run_dir, "batches.csv"), fields,
              [[b["batch_id"], b["queue"], b["b"], "%.6f" % b["t_start"], "%.6f" % b["t_end"], "%.3f" % b["h2d_ms"],
                "%.3f" % b["forward_ms"], "%.3f" % b["nms_ms"], b["n_boxes"], "%.4f" % b["decide_ms"], b["queue_len_before"]]
               for b in rt.batches])
    write_csv(os.path.join(run_dir, "waits.csv"), ["t_start", "t_end", "reason"],
              [["%.6f" % s, "%.6f" % e, r] for s, e, r in rt.waits])
    cols = ["t_monotonic"] + ["rail_%s_mw" % r for r in config.RAILS] + list(config.POWER_COLS.values()) + ["gr3d_load", "gr3d_mhz", "thermal_c"]
    rows = []
    for t, parsed, p in list(sampler.samples):
        rows.append(["%.3f" % t] + [p[r] for r in config.RAILS] + [p["total"], p["total_excl_sys5v"], p["gpu"],
                                                                      parsed.get("gr3d_load", ""), parsed.get("gr3d_mhz", ""),
                                                                      parsed["temps"].get("thermal", "")])
    write_csv(os.path.join(run_dir, "power.csv"), cols, rows)


# ---------- main ----------

def main():
    args = parse_args()
    if os.geteuid() != 0:
        sys.exit("run with sudo: tegrastats shows no power rails without root")
    params = parse_params(args.param)
    warmup_s, duration_s = config.WARMUP_S, config.DURATION_S
    if args.smoke:
        warmup_s, duration_s = 0, config.SMOKE_DURATION_S
    if not os.path.exists(config.PROFILE_PATH):
        sys.exit("no profile.json at %s: run 01_calibrate.py first" % config.PROFILE_PATH)
    if args.mix not in config.MIXES:
        sys.exit("unknown mix %r, choose from %s" % (args.mix, list(config.MIXES)))
    profile = profile_lib.load(config.PROFILE_PATH)
    policy = policies.make_policy(args.policy, params)
    name = policies.run_name(args.mix, args.lam, args.policy, params, args.seed)
    run_dir = args.out or os.path.join(config.RESULTS_DIR, "smoke" if args.smoke else "runs", name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)      # a half-written earlier attempt
    os.makedirs(run_dir)

    weights = config.MIXES[args.mix]
    lam_cap = traffic.capacity(profile, weights, config.B_MAX)
    lam = args.lam
    nominal_load = lam / lam_cap
    rates = traffic.queue_rates(weights, lam)
    schedule = traffic.build_schedule(rates, warmup_s + duration_s, config.INPUT_POOL_SIZE, args.seed)
    arrivals_hash = traffic.write_arrivals(os.path.join(run_dir, "arrivals.csv"), schedule)
    print("run %s" % name)
    print("Lambda %.1f req/s (%.0f%% of Lambda_cap %.1f); per queue %s; %d arrivals planned"
          % (lam, nominal_load * 100.0, lam_cap, ["%s %.1f" % (l, r) for l, r in zip(config.QUEUES, rates)], len(schedule)))
    print("power mode setting: SESSION_MODE = %s" % config.SESSION_MODE)

    def service_ms(q, b, stat=None):
        return profile_lib.service_ms(profile, config.QUEUES[q], b, stat or config.PROFILE_STAT)

    info = {"name": name, "policy": args.policy, "params": params, "policy_label": policies.label(args.policy, params),
            "mix": args.mix, "weights": weights, "nominal_load": nominal_load, "lambda_cap": lam_cap, "lambda_total": lam, "rates": rates,
            "seed": args.seed, "warmup_s": warmup_s, "duration_s": duration_s, "smoke": args.smoke,
            "arrivals_hash": arrivals_hash, "n_arrivals": len(schedule), "profile_sha1": file_sha1(config.PROFILE_PATH),
            "profile_stat": config.PROFILE_STAT, "b_max": config.B_MAX, "queue_cap": config.QUEUE_CAP,
            "primary_power": config.PRIMARY_POWER, "git_commit": git_commit(), "python": sys.version.split()[0],
            "torch": torch.__version__, "start": datetime.datetime.now().isoformat(), "status": "aborted", "flags": []}
    import ultralytics
    info["ultralytics"] = ultralytics.__version__

    # The job maker is its own process, forked now, before any CUDA work. Then the main process is kept off its core.
    main_cores, arrival_core = runtime.pin_cores()
    arrivals = runtime.ArrivalProcess(schedule, warmup_s, arrival_core)
    arrivals.start()
    if main_cores is not None:
        os.sched_setaffinity(0, set(main_cores))
    info["pinning"] = ("main process on cores %s, arrival process on core %s" % (sorted(main_cores), arrival_core)
                       if main_cores is not None else "not pinned")
    print(info["pinning"])

    session = None if args.session_managed else device.start_session()
    sampler = None
    rt = None
    try:
        sampler = start_sampler(run_dir)
        pool, info["input_source"] = inputs.build_pool()
        nets = models.load_nets()
        info["nms_backend"] = models.NMS_BACKEND
        buffers = {letter: inputs.BatchBuffer() for letter in config.QUEUES}
        models.warm_up(nets, buffers, pool)
        mode_id = powermode.current_mode()
        info["power_mode"] = {"id": mode_id, "name": device.mode_name(mode_id)}
        info["clocks_locked"] = config.LOCK_CLOCKS
        info["temp_start_c"] = runtime.wait_cool(sampler)
        t0 = time.monotonic() + 1.0
        info["t0"] = t0
        rt = runtime.Runtime(policy, nets, buffers, pool, service_ms, arrivals)
        print("running: %d s warmup + %d s measurement, then drain" % (warmup_s, duration_s))
        arrivals.go(t0)
        rt.run()
        time.sleep(config.SAMPLER_TAIL_S)
        info["status"] = "done"
    except KeyboardInterrupt:
        print("\ninterrupted: writing what has been logged, status = aborted")
    finally:
        arrivals.stop()
        if sampler:
            info["temp_end_c"] = runtime.board_temp(sampler)
            sampler.stop()
        if rt:
            info["overflow_drops"] = rt.overflow
            info["policy_drops"] = sum(1 for j in rt.jobs if j.dropped == 2)
            info["drain_timeout"] = rt.drain_timeout
            info["decide_calls"] = rt.decide_calls
            info["decide_ms_mean"] = rt.decide_total_ms / rt.decide_calls if rt.decide_calls else None
            late = [j.lateness_ms for j in rt.jobs if j.lateness_ms is not None]
            if late:
                info["arrival_lateness_p99_ms"] = metrics.stats(late)["p99"]
                if info["arrival_lateness_p99_ms"] >= config.LATENESS_P99_FLAG_MS:
                    info["flags"].append("arrival_lateness_p99_above_%gms" % config.LATENESS_P99_FLAG_MS)
            if rt.overflow:
                info["flags"].append("queue_overflow_drops")
            if rt.drain_timeout:
                info["flags"].append("drain_timeout")
            if rt.error:
                info["flags"].append("run_thread_error")
                info["error"] = rt.error
            if sampler:
                write_logs(run_dir, rt, sampler)
        if session:
            info["device_state"] = device.describe(session)
            device.end_session(session)
        info["end"] = datetime.datetime.now().isoformat()
        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump(info, f, indent=2, default=str)
        powermode.chown_to_invoker(run_dir)
        print("run folder:", run_dir, "| status:", info["status"], "| flags:", info["flags"] or "none")

    if args.smoke and info["status"] == "done":
        print("\nvalidation checks 1 to 8 (check 9 compares policies, skipped)")
        run = metrics.load_run(run_dir)
        all_ok = True
        for check_name, ok, detail in metrics.run_checks(run, profile):
            all_ok = all_ok and ok
            print("  %s  %s: %s" % ("PASS" if ok else "FAIL", check_name, detail))
        row = metrics.summarize(run, profile)
        print("\nmean latency %.1f ms, p99 %.1f ms, total energy %.1f J over %.1f s, %.0f mJ per request"
              % (row["lat_mean_ms"], row["lat_p99_ms"], row["energy_j"], row["span_s"], row["energy_per_req_mj"]))
        print("SMOKE " + ("PASS" if all_ok else "FAIL"))
    if info["status"] != "done":
        sys.exit(1)


main()
