# Read-only look at the board before anything is pinned: power mode, GPU devfreq node, the clock list against
# settings.FREQS_HZ, thermal zones, tegrastats output and the sample rate it really delivers, whether tegrastats
# GR3D_FREQ agrees with cur_freq, and which INA3221 power nodes exist. Writes nothing to sysfs.
# Run from the repo root (never from inside dvfs/: this file's name would shadow the stdlib inspect):
#   sudo python3 -m dvfs.inspect
import sys
sys.dont_write_bytecode = True
import glob
import json
import os
import platform
import time

from dvfs import gpu, settings, tegrastats
from dvfs.guard import run


def read(path):
    try:
        with open(path) as f:
            return f.read().replace("\x00", "").strip()
    except OSError as e:
        return "unreadable: %s" % e


def main():
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(), "root": os.geteuid() == 0}
    info["board"] = read("/proc/device-tree/model")
    info["l4t"] = read("/etc/nv_tegra_release")
    info["nvpmodel_q"] = run(["nvpmodel", "-q"])[1].strip()
    info["jetson_clocks_show"] = run(["jetson_clocks", "--show"])[1].strip()
    info["gpu"] = gpu.read_state()
    available = sorted(int(x) for x in info["gpu"]["available_frequencies"].split())
    cap = int(info["gpu"]["max_freq"])
    info["sweepable_hz"] = [f for f in available if f <= cap]
    info["settings_freqs_ok"] = {str(f): (f in available and f <= cap) for f in settings.FREQS_HZ}
    info["thermal_zones"] = {os.path.basename(z): {"type": read(z + "/type"), "temp_mC": read(z + "/temp")}
                             for z in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*"))}
    info["oc_counters"] = {p: read(p) for p in glob.glob(settings.HWMON_OC_GLOB)}
    info["ina3221_hwmon"] = {p: read(p) for p in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"))}
    try:
        import torch
        info["torch"], info["torch_cuda"] = torch.__version__, torch.version.cuda
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError as e:
        info["python_packages"] = "missing: %s" % e

    print("board:", info["board"])
    print("L4T:", info["l4t"].splitlines()[0] if info["l4t"] else "?")
    print("nvpmodel:", info["nvpmodel_q"].replace("\n", " | "))
    print("GPU devfreq:", {k: info["gpu"][k] for k in ("min_freq", "max_freq", "cur_freq", "governor", "enable_3d_scaling")})
    print("available_frequencies (MHz):", [f / 1e6 for f in available])
    print("sweepable up to the mode cap (MHz):", [f / 1e6 for f in info["sweepable_hz"]])
    print("settings.FREQS_HZ (MHz):", [f / 1e6 for f in settings.FREQS_HZ], "->",
          "all usable" if all(info["settings_freqs_ok"].values()) else "NOT USABLE: " + str([k for k, v in info["settings_freqs_ok"].items() if not v]))
    print("thermal zones:", {v["type"]: int(v["temp_mC"]) / 1000.0 for v in info["thermal_zones"].values() if v["temp_mC"].lstrip("-").isdigit()})
    print("over-current counters:", info["oc_counters"] or "none found")
    print("INA3221 rail labels:", list(info["ina3221_hwmon"].values()) or "none found at the JetPack 5 path (report only)")

    if not info["root"]:
        print("\nnot root: skipping the tegrastats checks (no power rails without root)")
    else:
        rates = {}
        changing = {}       # how often the total power reading actually differs from the previous sample
        for interval in (100, 50, 20):
            sampler = tegrastats.Sampler(interval, settings.RAILS)
            sampler.start()
            time.sleep(6)
            rates[interval] = sampler.cadence_hz()
            snap = list(sampler.samples)
            changes = sum(1 for a, b in zip(snap, snap[1:]) if a[2]["total"] != b[2]["total"])
            changing[interval] = changes / (snap[-1][0] - snap[0][0]) if len(snap) > 1 else 0.0
            last = sampler.latest()
            if interval == 100 and last is not None:
                info["tegrastats_sample"] = last[1]["raw"]
                tegra_mhz = last[1].get("gr3d_mhz")
                cur = gpu.read_attr(gpu.attr("cur_freq"))
                info["gr3d_vs_cur_freq"] = {"tegrastats_mhz": tegra_mhz, "cur_freq_hz": cur}
            sampler.stop()
        info["tegrastats_hz_by_interval_ms"] = rates
        info["power_value_changes_hz_by_interval_ms"] = changing
        print("\ntegrastats sample:", info.get("tegrastats_sample", "none (no rails?)"))
        print("achieved sample rate by --interval (ms):", {k: round(v, 2) for k, v in rates.items()})
        print("rate at which the total power value actually changes (idle, so a floor):", {k: round(v, 2) for k, v in changing.items()})
        print("-> pick the smallest interval whose sample rate still grows; if the changing rate stops growing, the sensor is the limit")
        print("idle GR3D_FREQ vs cur_freq:", info.get("gr3d_vs_cur_freq"))

    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    path = os.path.join(settings.RESULTS_DIR, "board_info_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("saved", path)


if __name__ == "__main__":
    main()
