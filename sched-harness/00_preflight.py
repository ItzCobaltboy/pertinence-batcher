# Checks the board before anything else: versions, root, a 3 s parsed tegrastats sample, power mode and clocks,
# RAM and disk, the input pool, all four nets resident together, one real forward + NMS at b=1 and b=8.
# Run: sudo python3 00_preflight.py   -> results/<device_tag>/preflight.json, ends with READY or NOT READY.
import sys
sys.dont_write_bytecode = True
import json
import os
import platform
import shutil
import time

import config
from common import powermode, tegrastats

info = {}
problems = []


def read_text(path):
    try:
        with open(path) as f:
            return f.read().replace("\x00", "").strip()
    except OSError as e:
        return "unreadable: %s" % e


def check_system():
    info["device_model"] = read_text("/proc/device-tree/model")
    info["l4t_release"] = read_text("/etc/nv_tegra_release")
    info["python"] = platform.python_version()
    info["root"] = os.geteuid() == 0
    print("board:", info["device_model"])
    print("L4T:", info["l4t_release"].splitlines()[0] if info["l4t_release"] else "?")
    print("python:", info["python"], "| root:", info["root"])
    if not info["python"].startswith("3.8"):
        problems.append("python is %s, the board stack is 3.8" % info["python"])


def check_torch():
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    print("torch:", torch.__version__, "| cuda available:", info["cuda_available"])
    if not info["cuda_available"]:
        problems.append("torch.cuda.is_available() is False")
        return
    import cv2
    import ultralytics
    info["ultralytics"] = ultralytics.__version__
    info["opencv"] = cv2.__version__
    print("ultralytics:", ultralytics.__version__, "| opencv:", cv2.__version__)
    try:
        from ultralytics.utils import ops
        ops.non_max_suppression
    except Exception as e:
        problems.append("FLAG: ultralytics NMS import failed on %s: %s" % (ultralytics.__version__, e))


def check_tegrastats():
    if not info["root"]:
        problems.append("not root: tegrastats shows no power rails and nvpmodel needs it. Run with sudo.")
        return
    if shutil.which("tegrastats") is None:
        problems.append("tegrastats not found on PATH")
        return
    sampler = tegrastats.Sampler(config.TEGRASTATS_INTERVAL_MS, config.RAILS)
    sampler.start()
    time.sleep(3.5)
    sampler.stop()
    if not sampler.samples:
        problems.append("no tegrastats sample with all six rails in 3 s (%d lines lacked rails)" % sampler.bad_lines)
        return
    _, parsed, p = sampler.samples[-1]
    info["rails_found"] = sorted(parsed["rails"])
    info["power_now_mw"] = p
    info["cadence_hz"] = sampler.cadence_hz()
    print("rails:", info["rails_found"])
    print("power now (mW):", p)
    print("cadence: %.2f samples/s (asked for %.1f)" % (info["cadence_hz"], 1000.0 / config.TEGRASTATS_INTERVAL_MS))


def check_modes():
    info["nvpmodel_q"] = powermode.run(["nvpmodel", "-q"])[1].strip()
    info["session_mode_setting"] = config.SESSION_MODE
    clocks = powermode.show_clocks()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "jetson_clocks_show.txt"), "w") as f:
        f.write(clocks)
    print("power mode now:", info["nvpmodel_q"].replace("\n", " | "))
    print("SESSION_MODE in config.py:", config.SESSION_MODE, "(None = leave the mode as it is)")
    print("clocks saved to jetson_clocks_show.txt")


def check_resources():
    disk = shutil.disk_usage(config.HERE)
    mem = {}
    for line in read_text("/proc/meminfo").splitlines():
        key, value = line.split(":")
        mem[key] = int(value.split()[0])
    info["disk_free_gb"] = round(disk.free / 1e9, 1)
    info["ram_available_mb"] = mem["MemAvailable"] // 1024
    print("disk free: %.1f GB | RAM available: %d MB" % (info["disk_free_gb"], info["ram_available_mb"]))


def check_pipeline():
    import torch
    from common import inputs, models
    pool, source = inputs.build_pool()
    info["input_pool"] = source
    print("input pool: %d arrays, source: %s" % (len(pool), source))
    try:
        nets = models.load_nets()
    except Exception as e:
        problems.append("loading the four nets together failed: %s" % e)
        return
    buffers = {letter: inputs.BatchBuffer() for letter in config.QUEUES}
    info["gpu_mem_after_load_mb"] = round(torch.cuda.memory_allocated() / 1e6)
    print("all four nets resident, GPU memory allocated: %d MB" % info["gpu_mem_after_load_mb"])
    info["pipeline_ms"] = {}
    for letter in config.QUEUES:
        for b in [1, 8, config.B_MAX]:
            try:
                models.run_batch(nets[letter], buffers[letter], pool[:b])           # first call is a warm-up
                h2d, fwd, nms, boxes = models.run_batch(nets[letter], buffers[letter], pool[:b])
            except Exception as e:
                if models.is_oom(e):
                    problems.append("FLAG: out of memory for yolov8%s at b=%d with all four nets loaded" % (letter, b))
                else:
                    problems.append("yolov8%s b=%d failed: %s" % (letter, b, e))
                torch.cuda.empty_cache()
                continue
            info["pipeline_ms"]["%s_b%d" % (letter, b)] = {"h2d": h2d, "forward": fwd, "nms": nms, "boxes": boxes}
            print("yolov8%s b=%-2d: h2d %.1f ms, forward %.1f ms, nms %.1f ms (%d boxes)" % (letter, b, h2d, fwd, nms, boxes))
    info["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6)
    print("GPU memory peak allocated: %d MB" % info["gpu_mem_peak_mb"])


def main():
    check_system()
    check_torch()
    if not problems:
        check_tegrastats()
    if not problems:
        check_modes()
        check_resources()
        check_pipeline()
    info["problems"] = problems
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "preflight.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    powermode.chown_to_invoker(config.RESULTS_DIR)
    powermode.chown_to_invoker(config.MODELS_DIR)
    print("\nsaved", path)
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nREADY")


main()
