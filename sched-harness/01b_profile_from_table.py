# Builds profile.json from the measured dvfs_runtime table instead of running the long GPU calibration (01_calibrate.py), which is the
# longest continuous GPU run in the setup and the step that kept powering the board off. Service times are the table's, at the clock
# ceiling (config.DVFS_MAX_MHZ). Only the idle power is measured here (20 s, light load).
# Run: sudo -E python3 01b_profile_from_table.py
import sys
sys.dont_write_bytecode = True
import csv
import datetime
import json
import os
import time

import torch

import config
from common import device, dvfs, inputs, models, tegrastats


def table_rows():
    rows = {}
    with open(config.DVFS_TABLE, newline="") as fh:
        for r in csv.DictReader(fh):
            f = int(round(float(r["freq_mhz"])))
            if f == config.DVFS_MAX_MHZ:
                rows[(r["model"], int(r["batch"]))] = r
    return rows


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
    rows = table_rows()
    profile = {"meta": {"device_tag": config.DEVICE_TAG, "date": datetime.datetime.now().isoformat(), "torch": torch.__version__,
                        "source": "built from %s at %d MHz, not calibrated in this harness" % (config.DVFS_TABLE, config.DVFS_MAX_MHZ),
                        "p99_ms_is": "lat_max_ms of the table (an upper bound)"},
               "models": {}}
    for label in config.QUEUES:
        cells = {}
        for b in config.CALIB_BATCHES:
            r = rows[(label, b)]
            cells[str(b)] = {"total": {"p50_ms": float(r["lat_p50_ms"]), "p95_ms": float(r["lat_p95_ms"]), "p99_ms": float(r["lat_max_ms"]),
                                       "mean_ms": float(r["lat_mean_ms"])}}
        profile["models"][label] = {"b": cells}
        print("%s b=1 %.1f ms, b=16 %.1f ms (p50)" % (config.MODELS[label], cells["1"]["total"]["p50_ms"], cells["16"]["total"]["p50_ms"]))
    profile["switch_cost"] = {"note": "model switching cost was +0.23 ms in the 16:50 calibration run; the GPU clock switch costs about 88 ms (DVFS_SWITCH_MS)"}

    session = device.start_session()
    clock = None
    try:
        clock = dvfs.Clock(os.path.join(config.RESULTS_DIR, "gpu_clock_backup.json"), config.DVFS_MAX_MHZ)
        clock.set(max(f for f in clock.available_mhz if f <= config.DVFS_MAX_MHZ + 1))
        print("GPU pinned at %d MHz, loading the nets, then %d s idle" % (clock.current_mhz, config.CALIB_IDLE_S))
        models.load_nets()
        torch.cuda.synchronize()
        profile["p_idle_loaded_mw"] = measure_idle_power(config.CALIB_IDLE_S)
        print("idle: " + ", ".join("%s %.0f mW" % (k, v) for k, v in profile["p_idle_loaded_mw"].items()))
    finally:
        if clock is not None:
            clock.restore()
        device.end_session(session)

    from common import traffic
    for name, weights in config.MIXES.items():
        print("Lambda_cap %s: %.0f req/s" % (name, traffic.capacity(profile, weights, config.B_MAX)))
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.PROFILE_PATH, "w") as fh:
        json.dump(profile, fh, indent=2)
    print("saved", config.PROFILE_PATH)


main()
