# Power sweep for YOLOv8 n/s/m/l on the GPU. This is the only script you run.
#
#   python power_sweep.py              measure everything, then aggregate
#   python power_sweep.py --aggregate  only redo the tables and plots from saved data
#
# How it works:
#   1. start nvidia-smi in the background. It writes one power sample every
#      SAMPLE_MS into ONE file for the whole sweep (one long power trace).
#   2. baseline: sleep BASELINE_S with nothing loaded on the GPU
#   3. every (model, batch) combo, in random order: run worker.py in a new
#      process. The worker writes down when each of its phases started and ended.
#   4. stop nvidia-smi, turn its timestamps into seconds, then aggregate.py cuts
#      the trace into phases using those start/end times.

import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime

import settings
import aggregate


def combo_name(model, batch):
    # "yolov8n.pt", 8 -> "yolov8n_bs8"
    stem = os.path.splitext(os.path.basename(model))[0]
    return f"{stem}_bs{batch}"


def start_sampler(raw_path):
    cmd = settings.NVIDIA_SMI + [
        "-i", settings.GPU_INDEX,
        "--query-gpu=timestamp," + settings.POWER_FIELD,
        "--format=csv,noheader,nounits",
        "-lms", str(settings.SAMPLE_MS),
        "-f", raw_path,
    ]
    print("starting sampler:", " ".join(cmd))
    return subprocess.Popen(cmd)


def stop_sampler(proc):
    # Ctrl-C makes nvidia-smi exit cleanly (and flush its file); kill it if it doesn't
    if os.name == "posix":
        proc.send_signal(signal.SIGINT)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def convert_samples(raw_path, out_path):
    # nvidia-smi lines look like "2026/10/09 12:34:56.789, 85.23".
    # The timestamp is local time on this machine, so turn it into the same
    # seconds-since-1970 that time.time() gives (and that the workers used).
    rows = 0
    with open(raw_path) as f_in, open(out_path, "w") as f_out:
        f_out.write("time,power_w\n")
        for line in f_in:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                t = datetime.strptime(parts[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                watts = float(parts[1])
            except ValueError:
                continue  # "[N/A]" or a broken line
            f_out.write(f"{t:.3f},{watts}\n")
            rows += 1
    print(f"converted {rows} power samples -> {out_path}")


def run_worker(model, batch, combos_dir):
    name = combo_name(model, batch)
    job = {
        "model": model,
        "batch": batch,
        "out": os.path.join(combos_dir, name + ".json"),
        "models_dir": settings.MODELS_DIR,
        "images_dir": settings.IMAGES_DIR,
        "img_size": settings.IMG_SIZE,
        "seed": settings.SEED,
        "idle_s": settings.IDLE_S,
        "num_run_phases": settings.NUM_RUN_PHASES,
        "forwards_per_phase": settings.FORWARDS_PER_PHASE,
        "buffer_s": settings.BUFFER_S,
    }
    cmd = [sys.executable, os.path.join(settings.HERE, "worker.py"), json.dumps(job)]
    log_path = os.path.join(combos_dir, name + ".log")
    with open(log_path, "w") as log:
        try:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=settings.WORKER_TIMEOUT_S)
            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode}), see {log_path}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {settings.WORKER_TIMEOUT_S} s, see {log_path}")


def measure():
    out_dir = settings.OUT_DIR
    combos_dir = os.path.join(out_dir, "combos")
    os.makedirs(combos_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "samples_raw.csv")

    # every (model, batch) combo once, in a random (but repeatable) order
    combos = []
    for model in settings.MODELS:
        for batch in settings.BATCH_SIZES:
            combos.append((model, batch))
    random.Random(settings.SEED).shuffle(combos)

    # rough time estimate: ~15 s to start a worker and load the model, the idle phase and the
    # buffers. The run phases come on top and depend on the batch size (about 1 h on the A100
    # with the default 84 combos and 200 forwards per phase).
    per_combo = 15 + settings.IDLE_S + settings.NUM_RUN_PHASES * settings.BUFFER_S
    fixed_min = (settings.BASELINE_S + len(combos) * per_combo) / 60
    print(f"{len(combos)} combos: about {fixed_min:.0f} min of loading/idle/buffers, plus the run phases")

    sampler = start_sampler(raw_path)
    try:
        time.sleep(2)  # give nvidia-smi a moment to start

        # baseline: nothing loaded
        print(f"baseline: {settings.BASELINE_S:.0f} s with nothing loaded")
        baseline = {"start": time.time()}
        time.sleep(settings.BASELINE_S)
        baseline["end"] = time.time()
        with open(os.path.join(out_dir, "baseline.json"), "w") as f:
            json.dump(baseline, f, indent=1)

        for i, (model, batch) in enumerate(combos, start=1):
            print(f"[{i}/{len(combos)}] {model} batch {batch}")
            run_worker(model, batch, combos_dir)

        time.sleep(settings.END_TAIL_S)
    finally:
        stop_sampler(sampler)  # always stop nvidia-smi, even after Ctrl-C or an error

    convert_samples(raw_path, os.path.join(out_dir, "samples.csv"))


def main():
    if "--aggregate" not in sys.argv:
        measure()
    aggregate.aggregate(settings.OUT_DIR)


if __name__ == "__main__":
    main()
