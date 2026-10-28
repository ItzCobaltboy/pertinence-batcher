# The measurement: for each Jetson power mode, YOLO model and batch size, time the FP16 network
# forward pass while one tegrastats trace runs, then cut the trace with the repetition marks
# to get power and energy. Always restores the original power mode and clocks at the end.
# Run: sudo .venv/bin/python 01_power_sweep.py   (settings are in config.py, SMOKE first)
import sys
sys.dont_write_bytecode = True
import csv
import datetime
import gc
import json
import os
import signal
import time
import types

import numpy as np
import torch

import config
from common import bench, powermode, tegrastats

# power column names in the CSVs, for the three power definitions (the first is the headline)
DEFS = [("", "total", "power_mw"),
        ("_excl_sys5v", "total_excl_sys5v", "power_excl_sys5v_mw"),
        ("_gpu", "gpu", "power_GPU_mw")]
RAIL_COLS = ["power_%s_mw" % r for r in config.RAILS]

RESULT_FIELDS = (["mode_id", "mode_name", "clocks_locked", "model", "batch", "rep", "iterations", "wall_s",
                  "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "throughput_fps"]
                 + [d[2] for d in DEFS[:2]] + RAIL_COLS
                 + ["energy_per_batch%s_mj" % d[0] for d in DEFS]
                 + ["energy_per_frame%s_mj" % d[0] for d in DEFS]
                 + ["dynamic_energy_per_frame%s_mj" % d[0] for d in DEFS]
                 + ["gpu_freq_mhz", "gr3d_load_pct", "temp_start_c", "temp_end_c", "thermal_wait_s",
                    "cadence_hz", "samples_in_window"])
IDLE_FIELDS = (["mode_id", "mode_name", "clocks_locked", "model", "seconds", "power_mw", "power_excl_sys5v_mw"]
               + RAIL_COLS + ["gpu_freq_mhz", "temp_c"])
STATUS_FIELDS = ["mode_id", "model", "batch", "status", "detail"]


# ---------- small helpers ----------

def append_row(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def power_columns(p):
    """powers() dict -> CSV columns for the two totals and the six rails."""
    cols = {"power_mw": p["total"], "power_excl_sys5v_mw": p["total_excl_sys5v"]}
    for r in config.RAILS:
        cols["power_%s_mw" % r] = p[r]
    return cols


def mean_over(samples, t0, t1):
    """Time-weighted mean of every power column over [t0, t1]."""
    keys = config.RAILS + ["total", "total_excl_sys5v", "gpu"]
    return {k: tegrastats.mean_power(samples, k, t0, t1) for k in keys}


def mean_gpu_freq(samples, t0, t1):
    f = [s[1].get("gr3d_mhz") for s in samples if t0 <= s[0] <= t1 and s[1].get("gr3d_mhz") is not None]
    return round(float(np.mean(f)), 1) if f else ""


def mean_gpu_load(samples, t0, t1):
    f = [s[1]["gr3d_load"] for s in samples if t0 <= s[0] <= t1 and "gr3d_load" in s[1]]
    return round(float(np.mean(f)), 1) if f else ""


def thermal_now(sampler):
    s = sampler.latest()
    return s[1]["temps"].get("thermal") if s else None


def wait_cool(sampler):
    """Pause while thermal > 80 C until it is below 65 C. Returns (seconds waited, start temp)."""
    temp = thermal_now(sampler)
    if temp is None or temp <= config.THERMAL_PAUSE_ABOVE:
        return 0.0, temp
    print("   thermal %.1f C > %d: pausing until below %d C" % (temp, config.THERMAL_PAUSE_ABOVE, config.THERMAL_RESUME_BELOW))
    start = time.monotonic()
    while thermal_now(sampler) > config.THERMAL_RESUME_BELOW:
        time.sleep(5)
    return time.monotonic() - start, temp


# ---------- plan and time estimate ----------

def estimate_seconds(n_modes, n_models, n_batches, reps, min_seconds, idle_s, settle_s):
    cell = 2 + reps * (min_seconds + config.GAP_S)
    per_model = idle_s + 10 + n_batches * cell          # idle with model + loading + cells
    per_mode = settle_s + idle_s + 5 + n_models * per_model + config.SAMPLER_TAIL_S
    return n_modes * per_mode


def pick_grid(budget_min, n_modes, models, args):
    """Largest of a few shrinking grids that fits the budget (the first one is the requested grid)."""
    options = [(args.batches, args.reps), ([1, 2, 4, 8, 16, 32], args.reps), ([1, 2, 4, 8, 16, 32], 2),
               ([1, 4, 8, 16], 2), ([1, 4, 8, 16], 1), ([1, 8], 1)]
    for batches, reps in options:
        total = estimate_seconds(n_modes, len(models), len(batches), reps, args.min_seconds, args.idle_s, args.settle_s)
        if total <= budget_min * 60:
            return batches, reps
    print("WARNING: even the smallest grid does not fit %d min; using it anyway" % budget_min)
    return options[-1]


# ---------- measurement ----------

def measure_idle(sampler, seconds):
    t0 = time.monotonic()
    time.sleep(seconds)
    t1 = time.monotonic()
    time.sleep(0.6)             # let a sample after t1 arrive
    samples = sampler.window(t0, t1)
    return mean_over(samples, t0, t1), mean_gpu_freq(samples, t0, t1), thermal_now(sampler)


def idle_row(mode, locked, model, seconds, power, gpu_freq, temp):
    row = {"mode_id": mode[0], "mode_name": mode[1], "clocks_locked": locked, "model": model,
           "seconds": seconds, "gpu_freq_mhz": gpu_freq, "temp_c": temp}
    row.update({k: round(v, 1) for k, v in power_columns(power).items()})
    return row


def make_row(sampler, mode, locked, letter, b, rep, latencies, t0, t1, idle, temp_start, temp_end, waited):
    samples = sampler.window(t0, t1, pad=1.0)
    inside = [s for s in samples if t0 <= s[0] <= t1]
    if len(inside) < 2:
        raise RuntimeError("only %d tegrastats samples inside the repetition window" % len(inside))
    times = [s[0] for s in samples]
    wall = t1 - t0
    iters = len(latencies)
    per_batch_s = wall / iters
    mean_p = mean_over(samples, t0, t1)
    lat = np.array(latencies)
    row = {"mode_id": mode[0], "mode_name": mode[1], "clocks_locked": locked, "model": letter, "batch": b,
           "rep": rep, "iterations": iters, "wall_s": round(wall, 3),
           "latency_ms_mean": round(float(lat.mean()), 4), "latency_ms_p50": round(float(np.percentile(lat, 50)), 4),
           "latency_ms_p95": round(float(np.percentile(lat, 95)), 4),
           "throughput_fps": round(b * iters / wall, 2)}
    row.update({k: round(v, 1) for k, v in power_columns(mean_p).items()})
    for suffix, key, col in DEFS:
        energy = tegrastats.energy_mj([s[0] for s in samples], [s[2][key] for s in samples], t0, t1)
        row["energy_per_batch%s_mj" % suffix] = round(energy / iters, 3)
        row["energy_per_frame%s_mj" % suffix] = round(energy / iters / b, 3)
        idle_mw = float(idle[col])
        row["dynamic_energy_per_frame%s_mj" % suffix] = round((mean_p[key] - idle_mw) * per_batch_s / b, 3)
    row.update({"gpu_freq_mhz": mean_gpu_freq(samples, t0, t1), "gr3d_load_pct": mean_gpu_load(samples, t0, t1),
                "temp_start_c": temp_start, "temp_end_c": temp_end, "thermal_wait_s": round(waited, 1),
                "cadence_hz": round(len(inside) / wall, 2), "samples_in_window": len(inside)})
    return row


def run_cell(net, images, b, sampler, mode, locked, letter, idle, args):
    """All repetitions of one (mode, model, batch). Returns the rows (written by the caller)."""
    x = bench.make_batch(b, images)
    bench.warmup(net, x, config.WARMUP_ITERS)
    rows = []
    waited_total = 0.0
    for rep in range(args.reps):
        waited, _ = wait_cool(sampler)
        waited_total += waited
        temp_start = thermal_now(sampler)
        latencies, t0, t1 = bench.timed_run(net, x, args.min_seconds, config.MIN_ITERS, config.CHUNK_ITERS)
        temp_end = thermal_now(sampler)
        time.sleep(config.GAP_S)        # also lets the samples after t1 arrive
        rows.append(make_row(sampler, mode, locked, letter, b, rep, latencies, t0, t1, idle,
                             temp_start, temp_end, waited))
        r = rows[-1]
        print("   b=%-2d rep %d: %.1f ms/batch, %.0f fps, %.0f mW, %.1f mJ/frame"
              % (b, rep, r["latency_ms_mean"], r["throughput_fps"], r["power_mw"], r["energy_per_frame_mj"]))
    return rows, waited_total


def run_model(letter, images, sampler, mode, locked, args, out, done):
    print("  model yolov8%s" % letter)
    batches = [b for b in args.batches if (mode[0], letter, b) not in done]
    oom_at = min([b for (m, l, b), s in done.items() if m == mode[0] and l == letter and s == "OOM"], default=None)
    if oom_at is not None:
        batches = [b for b in batches if b < oom_at]
    if not batches:
        print("   all cells done already")
        return
    net = bench.load_net(letter)
    try:
        idle = idle_for(sampler, mode, locked, letter, args, out)
        for b in batches:
            status_row = {"mode_id": mode[0], "model": letter, "batch": b}
            try:
                rows, waited = run_cell(net, images, b, sampler, mode, locked, letter, idle, args)
            except Exception as e:
                gc.collect()
                torch.cuda.empty_cache()
                if bench.is_oom(e):
                    print("   OOM at b=%d, stopping this model" % b)
                    append_row(out["status"], STATUS_FIELDS, dict(status_row, status="OOM", detail=str(e)[:200]))
                else:
                    print("   ERROR at b=%d: %s" % (b, e))
                    append_row(out["status"], STATUS_FIELDS, dict(status_row, status="error", detail=str(e)[:200]))
                break
            for row in rows:
                append_row(out["results"], RESULT_FIELDS, row)
            if waited > 0:
                append_row(out["status"], STATUS_FIELDS, dict(status_row, status="thermal_pause", detail="%.0f s" % waited))
            append_row(out["status"], STATUS_FIELDS, dict(status_row, status="done", detail=""))
            torch.cuda.empty_cache()
    finally:
        del net
        gc.collect()
        torch.cuda.empty_cache()


def idle_for(sampler, mode, locked, letter, args, out):
    """Idle power with this model loaded; measured once per (mode, model), reused on resume."""
    for r in read_rows(out["idle"]):
        if int(r["mode_id"]) == mode[0] and r["model"] == letter:
            return r
    print("   idle with model loaded (%d s)" % args.idle_s)
    power, gpu_freq, temp = measure_idle(sampler, args.idle_s)
    row = idle_row(mode, locked, letter, args.idle_s, power, gpu_freq, temp)
    append_row(out["idle"], IDLE_FIELDS, row)
    return row


def run_mode(mode, images, args, out, done, meta):
    locked = args.lock_clocks
    print("\n=== mode %d %s ===" % mode)
    try:
        powermode.set_mode(mode[0])
    except RuntimeError as e:
        print("  cannot set mode, skipping:", e)
        append_row(out["status"], STATUS_FIELDS, {"mode_id": mode[0], "model": "-", "batch": 0, "status": "mode_error", "detail": str(e)[:200]})
        return
    time.sleep(args.settle_s)
    if locked:
        powermode.lock_clocks()
    clocks = powermode.show_clocks()
    with open(os.path.join(out["raw"], "mode%d_jetson_clocks_show.txt" % mode[0]), "w") as f:
        f.write(clocks)

    sampler = tegrastats.Sampler(config.TEGRASTATS_INTERVAL_MS, config.RAILS,
                                 os.path.join(out["raw"], "mode%d_tegrastats.log.gz" % mode[0]))
    sampler.start()
    try:
        time.sleep(2)
        if not sampler.samples:
            raise RuntimeError("no tegrastats samples with rails (not root?)")
        if not any(r["model"] == "none" and int(r["mode_id"]) == mode[0] for r in read_rows(out["idle"])):
            print("  idle baseline, CUDA initialised, no model (%d s)" % args.idle_s)
            power, gpu_freq, temp = measure_idle(sampler, args.idle_s)
            append_row(out["idle"], IDLE_FIELDS, idle_row(mode, locked, "none", args.idle_s, power, gpu_freq, temp))
        for letter in args.models:
            run_model(letter, images, sampler, mode, locked, args, out, done)
        time.sleep(config.SAMPLER_TAIL_S)
    finally:
        sampler.stop()
        meta["mode_cadence_hz"][str(mode[0])] = round(sampler.cadence_hz(), 2)
        print("  tegrastats cadence in this mode: %.2f samples/s" % meta["mode_cadence_hz"][str(mode[0])])


# ---------- main ----------

def settings():
    """Everything the sweep needs, from config.py (there are no command line flags)."""
    return types.SimpleNamespace(
        modes=config.SWEEP_MODES, models=list(config.SWEEP_MODELS), batches=list(config.BATCHES),
        reps=config.REPS, min_seconds=config.MIN_SECONDS, lock_clocks=config.LOCK_CLOCKS,
        budget_min=config.BUDGET_MIN, smoke=config.SMOKE, idle_s=config.IDLE_S, settle_s=config.MODE_SETTLE_S)


def on_sigterm(signum, frame):
    # treat kill like Ctrl+C so the finally block still restores the mode and clocks
    raise KeyboardInterrupt()


def main():
    if os.geteuid() != 0:
        sys.exit("run with sudo: tegrastats shows no power rails without root, and nvpmodel needs it")
    signal.signal(signal.SIGTERM, on_sigterm)
    args = settings()

    all_modes = powermode.list_modes(config.NVPMODEL_CONF)
    original_mode = powermode.current_mode()
    if args.smoke:
        args.modes = [original_mode]
        args.models, args.batches, args.reps = ["n"], [1, 4], 1
        args.min_seconds, args.idle_s, args.settle_s = 5, 5, 3
    by_name = {name: mode_id for mode_id, name in all_modes}
    wanted = []
    for m in (args.modes if args.modes else [mid for mid, _ in all_modes]):
        wanted.append(by_name[m] if m in by_name else m)
    unknown = [m for m in wanted if m not in dict(all_modes)]
    if unknown:
        sys.exit("modes %s are not in %s: %s" % (unknown, config.NVPMODEL_CONF, all_modes))
    modes = [(m, dict(all_modes)[m]) for m in wanted]
    for letter in args.models:
        if letter not in config.MODELS:
            sys.exit("unknown model %r, choose from %s" % (letter, list(config.MODELS)))
    if args.budget_min:
        args.batches, args.reps = pick_grid(args.budget_min, len(modes), args.models, args)

    results_dir = config.RESULTS_DIR + ("_smoke" if args.smoke else "")
    out = {"dir": results_dir, "raw": os.path.join(results_dir, "raw")}
    for name in ["results", "idle", "status"]:
        out[name] = os.path.join(results_dir, name + ".csv")
    os.makedirs(out["raw"], exist_ok=True)
    done = {(int(r["mode_id"]), r["model"], int(r["batch"])): r["status"]
            for r in read_rows(out["status"]) if r["status"] in ("done", "OOM")}

    est = estimate_seconds(len(modes), len(args.models), len(args.batches), args.reps, args.min_seconds, args.idle_s, args.settle_s)
    print("modes %s | models %s | batches %s | reps %d | min %g s | clocks %s"
          % ([m[0] for m in modes], args.models, args.batches, args.reps, args.min_seconds,
             "locked" if args.lock_clocks else "DVFS"))
    print("estimated time: %.0f min (cells already done are skipped on resume)" % (est / 60))
    if args.smoke:
        print("*** SMOKE RUN (config.SMOKE = True): set it to False in config.py for the full sweep ***")
    print("results ->", results_dir)

    meta = {"command": " ".join(sys.argv), "start": datetime.datetime.now().isoformat(), "original_mode": original_mode,
            "modes": modes, "models": args.models, "batches": args.batches, "reps": args.reps,
            "min_seconds": args.min_seconds, "lock_clocks": args.lock_clocks, "torch": torch.__version__,
            "mode_cadence_hz": {}}
    preflight = os.path.join(config.RESULTS_DIR, "preflight.json")
    if os.path.exists(preflight):
        with open(preflight) as f:
            meta["preflight"] = json.load(f)

    clocks_backup = os.path.join(out["raw"], "jetson_clocks_original.conf")
    powermode.store_clocks(clocks_backup)
    torch.zeros(1).cuda()               # initialise CUDA so the no-model idle baseline includes the context
    images = bench.sample_images()
    try:
        for mode in modes:
            run_mode(mode, images, args, out, done, meta)
        print("\nsweep finished")
    except KeyboardInterrupt:
        print("\ninterrupted; finished cells are kept, rerun the same command to resume")
    finally:
        # mode first: switching modes resets clocks, then the saved clock state goes back on top
        print("restoring power mode %d and clocks" % original_mode)
        try:
            powermode.set_mode(original_mode)
        except RuntimeError as e:
            print("COULD NOT RESTORE MODE:", e, "-> run: sudo nvpmodel -m %d" % original_mode)
        try:
            powermode.restore_clocks(clocks_backup)
        except RuntimeError as e:
            print("COULD NOT RESTORE CLOCKS:", e, "-> run: sudo jetson_clocks --restore " + clocks_backup)
        meta["end"] = datetime.datetime.now().isoformat()
        with open(os.path.join(results_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        powermode.chown_to_invoker(results_dir)
        powermode.chown_to_invoker(config.MODELS_DIR)
        print("outputs in", results_dir)


main()
