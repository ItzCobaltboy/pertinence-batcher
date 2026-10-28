# Checks the board before a sweep: versions, root, tegrastats rails and cadence, power modes,
# clocks, disk/RAM/thermal, and that all four YOLOv8 models run one FP16 forward on the GPU.
# Run: sudo .venv/bin/python 00_preflight.py   -> results/<device_tag>/preflight.json
import sys
sys.dont_write_bytecode = True
import json
import os
import platform
import shutil
import subprocess
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
    info["machine"] = platform.machine()
    info["root"] = os.geteuid() == 0
    print("board:", info["device_model"])
    print("L4T:", info["l4t_release"].splitlines()[0] if info["l4t_release"] else "?")
    print("python:", info["python"], info["machine"], "| root:", info["root"])
    if "Xavier" not in info["device_model"]:
        print("WARNING: this does not look like a Jetson AGX Xavier; check DEVICE_TAG in config.py")


def check_torch():
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    print("torch:", torch.__version__, "| cuda available:", info["cuda_available"])
    if not info["cuda_available"]:
        problems.append("torch.cuda.is_available() is False: this is a CPU-only torch build or a driver problem")
        return
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["torch_cuda"] = torch.version.cuda
    print("gpu:", info["gpu_name"], "| torch cuda:", info["torch_cuda"])
    import torchvision
    import ultralytics
    info["torchvision"] = torchvision.__version__
    info["ultralytics"] = ultralytics.__version__
    print("torchvision:", torchvision.__version__, "| ultralytics:", ultralytics.__version__)


def check_tegrastats():
    if not info["root"]:
        problems.append("not root: tegrastats shows no power rails. Run with sudo.")
        return
    if shutil.which("tegrastats") is None:
        problems.append("tegrastats not found on PATH")
        return
    sampler = tegrastats.Sampler(config.TEGRASTATS_INTERVAL_MS, config.RAILS)
    sampler.start()
    time.sleep(3 + 0.5)
    sampler.stop()
    if not sampler.samples:
        problems.append("no tegrastats sample with all six rails in 3 s (%d lines lacked rails): run with sudo"
                        % sampler.bad_lines)
        return
    _, parsed, p = sampler.samples[-1]
    info["rails_found"] = sorted(parsed["rails"])
    info["power_now_mw"] = p
    info["cadence_hz"] = sampler.cadence_hz()
    info["thermal_now_c"] = parsed["temps"].get("thermal")
    info["ram_used_mb"] = parsed.get("ram_used_mb")
    print("rails:", info["rails_found"])
    print("power now (mW):", p)
    print("sampling cadence: %.2f samples/s (asked for %.1f)" % (info["cadence_hz"], 1000.0 / config.TEGRASTATS_INTERVAL_MS))
    print("thermal now:", info["thermal_now_c"], "C")


def check_modes():
    info["nvpmodel_q"] = powermode.run(["nvpmodel", "-q"])[1].strip()
    info["modes"] = powermode.list_modes(config.NVPMODEL_CONF)
    print("current:", info["nvpmodel_q"].replace("\n", " | "))
    print("modes in %s:" % config.NVPMODEL_CONF)
    for mode_id, name in info["modes"]:
        print("  %d  %s" % (mode_id, name))
    clocks = powermode.show_clocks()
    with open(os.path.join(config.RESULTS_DIR, "jetson_clocks_show.txt"), "w") as f:
        f.write(clocks)
    print("jetson_clocks --show saved (%d lines)" % len(clocks.splitlines()))


def check_resources():
    disk = shutil.disk_usage(config.HERE)
    info["disk_free_gb"] = round(disk.free / 1e9, 1)
    mem = {}
    for line in read_text("/proc/meminfo").splitlines():
        key, value = line.split(":")
        mem[key] = int(value.split()[0])
    info["ram_available_mb"] = mem["MemAvailable"] // 1024
    info["swap_total_mb"] = mem["SwapTotal"] // 1024
    print("disk free: %.1f GB | RAM available: %d MB | swap: %d MB"
          % (info["disk_free_gb"], info["ram_available_mb"], info["swap_total_mb"]))


def check_models():
    import torch
    from common import bench
    images = bench.sample_images()
    info["models"] = {}
    for letter in config.MODELS:
        try:
            net = bench.load_net(letter)
            for b in [1, 8]:
                bench.forward(net, bench.make_batch(b, images))
                torch.cuda.synchronize()
            info["models"][letter] = "ok"
            print("yolov8%s: loaded, FP16 forward ok at b=1 and b=8" % letter)
            del net
            torch.cuda.empty_cache()
        except Exception as e:
            info["models"][letter] = "FAILED: %s" % e
            problems.append("yolov8%s failed: %s" % (letter, e))
            print("yolov8%s: FAILED: %s" % (letter, e))


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    check_system()
    check_torch()
    if problems:
        return finish()
    check_tegrastats()
    if problems:
        return finish()
    check_modes()
    check_resources()
    check_models()
    finish()


def finish():
    path = os.path.join(config.RESULTS_DIR, "preflight.json")
    info["problems"] = problems
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    powermode.chown_to_invoker(config.RESULTS_DIR)
    powermode.chown_to_invoker(config.MODELS_DIR)
    print("\nsaved", path)
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nPREFLIGHT OK")


main()
