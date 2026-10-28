# Measures the service time T_i(b) of the whole serving pipeline (stack + H2D + forward + NMS) for every model
# and batch size, the cost of switching models, and the idle power with all four nets loaded.
# Run: sudo python3 01_calibrate.py   -> results/<device_tag>/profile.json
import sys
sys.dont_write_bytecode = True
import datetime
import json
import os
import time

import numpy as np
import torch

import config
from common import device, inputs, models, tegrastats


def part_stats(values):
    v = np.array(values)
    return {"mean_ms": float(v.mean()), "std_ms": float(v.std()), "p50_ms": float(np.percentile(v, 50)),
            "p95_ms": float(np.percentile(v, 95)), "p99_ms": float(np.percentile(v, 99))}


def batch_arrays(pool, k, b):
    """Different inputs on every call so the batch is not the same bytes each time."""
    return [pool[(k * b + i) % len(pool)] for i in range(b)]


def calibrate_cell(net, buffer, pool, b):
    for k in range(config.CALIB_WARMUP):
        models.run_batch(net, buffer, batch_arrays(pool, k, b))
    parts = {"h2d": [], "forward": [], "nms": [], "total": []}
    boxes = []
    for k in range(config.CALIB_TIMED):
        h2d, fwd, nms, n_boxes = models.run_batch(net, buffer, batch_arrays(pool, k, b))
        parts["h2d"].append(h2d)
        parts["forward"].append(fwd)
        parts["nms"].append(nms)
        parts["total"].append(h2d + fwd + nms)
        boxes.append(n_boxes)
    cell = {name: part_stats(values) for name, values in parts.items()}
    cell["boxes_mean"] = float(np.mean(boxes))
    return cell


def fit_line(table, stat):
    """T(b) = a + c*b by least squares on the chosen statistic; returns a, c and the worst relative residual."""
    sizes = sorted(int(k) for k in table)
    y = np.array([table[str(b)]["total"][stat + "_ms"] for b in sizes])
    c, a = np.polyfit(np.array(sizes, dtype=float), y, 1)
    resid = np.abs(a + c * np.array(sizes) - y) / y
    return float(a), float(c), float(resid.max()), [float(r) for r in resid]


def measure_switch_cost(nets, buffers, pool):
    """Alternate n and l at b=1 versus the same number of back-to-back batches on one model each."""
    n_batches = config.SWITCH_BATCHES
    alternating = []
    for k in range(n_batches):
        letter = "n" if k % 2 == 0 else "l"
        h2d, fwd, nms, _ = models.run_batch(nets[letter], buffers[letter], batch_arrays(pool, k, 1))
        alternating.append(h2d + fwd + nms)
    same = {}
    for letter in ["n", "l"]:
        times = []
        for k in range(n_batches):
            h2d, fwd, nms, _ = models.run_batch(nets[letter], buffers[letter], batch_arrays(pool, k, 1))
            times.append(h2d + fwd + nms)
        same[letter] = float(np.mean(times))
    alt_mean = float(np.mean(alternating))
    same_mean = (same["n"] + same["l"]) / 2.0
    return {"alternating_n_l_mean_ms": alt_mean, "same_model_mean_ms": same_mean, "difference_ms": alt_mean - same_mean,
            "same_model_n_mean_ms": same["n"], "same_model_l_mean_ms": same["l"]}


def measure_idle_power(seconds):
    sampler = tegrastats.Sampler(config.TEGRASTATS_INTERVAL_MS, config.RAILS)
    sampler.start()
    try:
        time.sleep(2)
        if not sampler.samples:
            raise RuntimeError("no tegrastats samples with rails (not root?)")
        t0 = time.monotonic()
        time.sleep(seconds)
        t1 = time.monotonic()
        time.sleep(0.6)
        samples = sampler.window(t0, t1)
    finally:
        sampler.stop()
    return {key: tegrastats.mean_power(samples, key, t0, t1) for key in config.POWER_COLS}


def main():
    assert max(config.CALIB_BATCHES) == config.B_MAX, "the largest CALIB_BATCHES entry must equal B_MAX"
    session = device.start_session()
    try:
        print("loading the four nets")
        pool, pool_source = inputs.build_pool()
        nets = models.load_nets()
        buffers = {letter: inputs.BatchBuffer() for letter in config.QUEUES}
        profile = {"meta": {"device_tag": config.DEVICE_TAG, "date": datetime.datetime.now().isoformat(),
                            "device_state": device.describe(session), "torch": torch.__version__,
                            "input_pool": pool_source, "nms_backend": models.NMS_BACKEND, "warmup": config.CALIB_WARMUP, "timed": config.CALIB_TIMED},
                   "models": {}}
        for letter in config.QUEUES:
            table = {}
            print("yolov8%s" % letter)
            for b in config.CALIB_BATCHES:
                try:
                    table[str(b)] = calibrate_cell(nets[letter], buffers[letter], pool, b)
                except Exception as e:
                    if models.is_oom(e):
                        sys.exit("FLAG: out of memory for yolov8%s at b=%d with all four nets loaded. B_MAX=%d does not fit; "
                                 "stopping instead of guessing." % (letter, b, config.B_MAX))
                    raise
                t = table[str(b)]
                print("  b=%-2d total p50 %.1f ms (h2d %.1f, forward %.1f, nms %.1f)"
                      % (b, t["total"]["p50_ms"], t["h2d"]["p50_ms"], t["forward"]["p50_ms"], t["nms"]["p50_ms"]))
            a, c, worst, resid = fit_line(table, config.PROFILE_STAT)
            profile["models"][letter] = {"b": table, "fit": {"a_ms": a, "c_ms_per_frame": c, "max_rel_err": worst,
                                                              "rel_err_per_b": dict(zip([str(b) for b in config.CALIB_BATCHES], resid))}}
            print("  batch time grows by %.2f ms per frame, floor %.2f ms (worst fit error %.1f%%)" % (c, a, worst * 100))

        nms_share = (profile["models"]["n"]["b"]["1"]["nms"]["p50_ms"] / profile["models"]["n"]["b"]["1"]["total"]["p50_ms"])
        profile["meta"]["nms_share_n_b1"] = nms_share
        print("NMS share of yolov8n service time at b=1: %.0f%%" % (nms_share * 100))
        if nms_share > config.NMS_DOMINATES_SHARE:
            print("FLAG: NMS dominates yolov8n service time (above %.0f%%)" % (config.NMS_DOMINATES_SHARE * 100))

        print("switch cost (alternate n/l at b=1 vs back-to-back)")
        profile["switch_cost"] = measure_switch_cost(nets, buffers, pool)
        print("  alternating %.2f ms vs same model %.2f ms -> %+.2f ms" % (
            profile["switch_cost"]["alternating_n_l_mean_ms"], profile["switch_cost"]["same_model_mean_ms"],
            profile["switch_cost"]["difference_ms"]))

        print("idle power with all four nets loaded (%d s)" % config.CALIB_IDLE_S)
        torch.cuda.synchronize()
        profile["p_idle_loaded_mw"] = measure_idle_power(config.CALIB_IDLE_S)
        print("  idle: " + ", ".join("%s %.0f mW" % (k, v) for k, v in profile["p_idle_loaded_mw"].items()))

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(config.PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2)
        print("saved", config.PROFILE_PATH)
    finally:
        device.end_session(session)


main()
