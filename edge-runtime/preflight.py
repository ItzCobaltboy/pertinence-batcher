# Environment check before a (possibly short) device visit. Under a minute.
#
#   python preflight.py            full check, writes results/<device_tag>/preflight.json
#   python preflight.py --quick    skip loading the models
#
# Reports: board / JetPack / CUDA / Python / package versions, power mode, sudo
# (non-interactive check, never prompts), the power sampler with a 3 s parsed sample,
# disk / RAM / GPU memory, the four YOLOv8 weights (each loaded and run at b = 1 and
# b = 8), traces and frame pool. Ends with "READY" or "NOT READY" and what is missing.
# Installs nothing.

import argparse
import glob
import importlib
import os
import shutil
import sys
import time

import config
from common import device_tag, is_jetson, power_mode, read_text, run_cmd, save_json

ULTRALYTICS_HINT = ("pip3 install --no-deps ultralytics ultralytics-thop && "
                    "pip3 install pyyaml requests scipy tqdm psutil py-cpuinfo   "
                    "(--no-deps: never let pip replace the JetPack torch/torchvision)")


def check_packages(report, problems):
    pk = {}
    for name in ["numpy", "pandas", "matplotlib", "PIL", "cv2", "torch", "torchvision", "ultralytics"]:
        try:
            mod = importlib.import_module(name)
            pk[name] = getattr(mod, "__version__", "?")
        except Exception as e:
            pk[name] = None
            report.setdefault("import_errors", {})[name] = str(e)[:300]
    report["packages"] = pk
    for name in ["numpy", "pandas", "PIL", "torch", "torchvision"]:
        if pk[name] is None:
            problems.append("%s missing%s" % (name, " (use the JetPack wheel, never pip's)" if name.startswith("torch") else ""))
    if pk["ultralytics"] is None:
        problems.append("ultralytics missing: " + ULTRALYTICS_HINT)
    if pk["matplotlib"] is None:
        report.setdefault("warnings", []).append("matplotlib missing: analyze.py must run on the laptop")
    if pk["cv2"] is None:
        report.setdefault("warnings", []).append("opencv missing: only needed for inputs.py export-frames (laptop)")


def check_torch(report, problems):
    try:
        import torch
    except Exception:
        return None
    info = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        info["gpu_free_mb"] = free / 2 ** 20
        info["gpu_total_mb"] = total / 2 ** 20
        try:
            import torchvision
            boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device="cuda")
            torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
            info["torchvision_nms_cuda"] = True
        except Exception as e:
            info["torchvision_nms_cuda"] = False
            problems.append("torchvision.ops.nms fails on CUDA (ultralytics NMS needs it): %s" % str(e)[:200])
    else:
        problems.append("torch.cuda.is_available() is False")
    report["torch"] = info
    return info


def check_power(report, problems):
    import power
    backend = "tegrastats" if is_jetson() else "nvidia-smi"
    if not shutil.which(backend):
        problems.append("%s not on PATH: no power measurements" % backend)
        report["power"] = {"backend": None}
        return
    tmp = os.path.join(config.RESULTS_DIR, "_preflight_power")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "power.csv")
    sampler = power.make_sampler(backend)
    sampler.start(path)
    try:
        time.sleep(3)
    finally:
        sampler.stop()
    df = power.read(path)
    info = {"backend": backend, "meta": power.read_meta(path), "samples": int(len(df))}
    info.update(power.cadence(df))
    if len(df):
        info["power_w_mean"] = float(df["power_w"].mean())
        info["columns"] = list(df.columns)
        if "power_source" in df:
            info["power_source"] = str(df["power_source"].iloc[-1])
            info["rails"] = [c for c in df.columns if c.startswith("rail_")]
            if df["unknown"].astype(str).str.len().max() > 0:
                info["unknown_tokens_example"] = str(df["unknown"].iloc[-1])[:300]
    else:
        problems.append("%s produced no parsable samples in 3 s" % backend)
    report["power"] = info
    shutil.rmtree(tmp, ignore_errors=True)


def check_models(report, problems, quick):
    from models import find_weights
    found = {m: find_weights(m) for m in config.MODELS}
    report["weights"] = found
    missing = [m for m, p in found.items() if p is None]
    if missing:
        problems.append("weights missing: %s (copy %s into %s)" % (
            missing, ", ".join(m + ".pt" for m in missing), os.path.join(config.YOLO_DISPATCHER_DIR, "models")))
        return
    if quick:
        return
    try:
        from inputs import TensorInputs
        from models import Runner
        runner = Runner(config.MODELS, nms=True, log=lambda *a: None)
        inputs = TensorInputs(config.FRAMES_DIR, runner.device, size=8, log=lambda *a: None)
        timing = {}
        for m in config.MODELS:
            timing[m] = {}
            for b in (1, 8):
                out, ms = runner.forward(m, inputs.sample_batch(b))      # first call: includes cuDNN setup
                out, ms2 = runner.forward(m, inputs.sample_batch(b))
                timing[m]["b%d_first_ms" % b] = ms
                timing[m]["b%d_ms" % b] = ms2
                timing[m]["b%d_nms_ms" % b] = runner.nms(out)
        report["model_check"] = {"timing": timing, "info": runner.info()}
    except Exception as e:
        problems.append("loading/running the models failed: %s" % str(e)[:300])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check this machine before a run or a visit.")
    parser.add_argument("--quick", action="store_true", help="skip loading and running the models")
    parser.add_argument("--device-tag", help="override the detected device tag (results/<tag>/)")
    args = parser.parse_args(argv)
    t0 = time.time()
    problems = []
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
              "jetson": is_jetson(), "board": read_text("/proc/device-tree/model") or None,
              "l4t": read_text("/etc/nv_tegra_release") or None, "power_mode": power_mode()}
    code, out = run_cmd(["nvcc", "--version"])
    report["nvcc"] = out.strip().splitlines()[-1] if code == 0 and out.strip() else None
    code, _ = run_cmd(["sudo", "-n", "true"], timeout=5)
    report["sudo_noninteractive"] = code == 0
    if sys.version_info < (3, 8):
        problems.append("Python %s < 3.8" % report["python"])

    check_packages(report, problems)
    tinfo = check_torch(report, problems)
    tag = args.device_tag or device_tag((tinfo or {}).get("gpu_name"))
    report["device_tag"] = tag
    check_power(report, problems)

    usage = shutil.disk_usage(config.HERE)
    report["disk_free_gb"] = usage.free / 2 ** 30
    if report["disk_free_gb"] < 2:
        problems.append("less than 2 GB free disk")
    meminfo = read_text("/proc/meminfo")
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable"):
            report["ram_available_mb"] = int(line.split()[1]) / 1024.0

    check_models(report, problems, args.quick)
    traces = sorted(glob.glob(os.path.join(config.TRACES_DIR, "*.csv")))
    frames = glob.glob(os.path.join(config.FRAMES_DIR, "*.jpg"))
    report["traces"] = [os.path.basename(t) for t in traces]
    report["frames"] = len(frames)
    if not frames:
        problems.append("no frame pool in %s (python inputs.py export-frames on the laptop)" % config.FRAMES_DIR)
    if not traces:
        report.setdefault("warnings", []).append(
            "no traces in %s yet (plan.py visit builds synthetic ones after calibration)" % config.TRACES_DIR)

    report["problems"] = problems
    report["ready"] = not problems
    report["seconds"] = time.time() - t0
    out_dir = config.device_dir(tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "preflight.json"), report)

    print("board      :", report["board"] or "-", "| L4T:", (report["l4t"] or "-")[:60])
    print("python     :", report["python"], "| packages:", ", ".join("%s %s" % kv for kv in report["packages"].items()))
    print("torch      :", report.get("torch"))
    print("power mode :", report["power_mode"], "| sudo -n:", report["sudo_noninteractive"])
    print("power      :", {k: report["power"].get(k) for k in ("backend", "samples", "median_interval_ms",
                                                               "power_w_mean", "power_source")})
    print("disk free  : %.1f GB | RAM available: %.0f MB" % (report["disk_free_gb"], report.get("ram_available_mb", 0)))
    print("weights    :", {m: bool(p) for m, p in report["weights"].items()})
    if "model_check" in report:
        for m, t in report["model_check"]["timing"].items():
            print("  %s: b1 %.1f ms, b8 %.1f ms (nms %.1f ms)" % (m, t["b1_ms"], t["b8_ms"], t["b8_nms_ms"]))
    print("traces     : %d | frames: %d" % (len(traces), len(frames)))
    for w in report.get("warnings", []):
        print("warning    :", w)
    print("device_tag :", tag, "->", os.path.join(out_dir, "preflight.json"), "(%.0f s)" % report["seconds"])
    if problems:
        print("NOT READY:")
        for p in problems:
            print("  -", p)
    else:
        print("READY")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
