"""
nvidia-smi snapshots, so interference and throttling on the shared GPU show
up in the data. No torch import. Every function returns something usable
even when nvidia-smi is missing (CPU test box): the fields are just None.
"""

import os
import shutil
import subprocess

GPU_FIELDS = [
    "index", "uuid", "name", "driver_version", "pstate",
    "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
    "temperature.gpu", "power.draw", "power.limit",
    "memory.used", "memory.total", "utilization.gpu",
    "clocks_throttle_reasons.active",
]


def _run(args, timeout=20):
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    if out.returncode != 0:
        return None, (out.stderr or out.stdout).strip()
    return out.stdout, None


def available():
    return shutil.which("nvidia-smi") is not None


def gpu_index():
    """Physical GPU index to query. CUDA_VISIBLE_DEVICES=3 means torch's cuda:0
    is nvidia-smi's GPU 3. A UUID or anything else falls back to "all GPUs"."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible.isdigit():
        return visible
    if visible.startswith("GPU-"):
        return visible
    return "0" if visible == "" else None


def _to_number(value):
    value = value.strip()
    if value in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def snapshot():
    """One dict of GPU state plus the list of compute processes on that GPU."""
    snap = {"available": available(), "gpu": None, "compute_apps": None, "error": None}
    if not snap["available"]:
        return snap

    index = gpu_index()
    select = ["-i", index] if index is not None else []
    fields = list(GPU_FIELDS)
    text, err = _run(["nvidia-smi"] + select + ["--query-gpu=" + ",".join(fields),
                                                "--format=csv,noheader,nounits"])
    if text is None:
        # Older/newer drivers name the throttle field differently; retry without it.
        fields = [f for f in fields if not f.startswith("clocks_throttle")]
        text, err = _run(["nvidia-smi"] + select + ["--query-gpu=" + ",".join(fields),
                                                    "--format=csv,noheader,nounits"])
    if text is None:
        snap["error"] = err
        return snap

    line = text.strip().splitlines()[0]
    values = [v.strip() for v in line.split(",")]
    gpu = {}
    for field, value in zip(fields, values):
        if field in ("uuid", "name", "driver_version", "pstate", "clocks_throttle_reasons.active"):
            gpu[field] = value
        else:
            gpu[field] = _to_number(value)
    snap["gpu"] = gpu

    apps, err = _run(["nvidia-smi"] + select + ["--query-compute-apps=pid,process_name,used_memory",
                                                "--format=csv,noheader,nounits"])
    if apps is not None:
        rows = []
        for app_line in apps.strip().splitlines():
            parts = [p.strip() for p in app_line.split(",")]
            if len(parts) >= 3:
                rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": _to_number(parts[2])})
        snap["compute_apps"] = rows
    return snap


def other_processes(snap, own_pid=None):
    """Compute processes on the GPU that are not us (other users on the shared box)."""
    if not snap or snap.get("compute_apps") is None:
        return []
    own = str(own_pid if own_pid is not None else os.getpid())
    return [a for a in snap["compute_apps"] if a["pid"] != own]


def full_report():
    """Plain `nvidia-smi` + `nvidia-smi -q` text, saved at sweep start and end."""
    if not available():
        return "nvidia-smi not found on this machine\n"
    parts = []
    for args in (["nvidia-smi"], ["nvidia-smi", "-q", "-d", "CLOCK,TEMPERATURE,PERFORMANCE,POWER,MEMORY"]):
        text, err = _run(args, timeout=60)
        parts.append("$ " + " ".join(args) + "\n" + (text if text is not None else f"ERROR: {err}\n"))
    return "\n".join(parts)


# Throttle reason bits (nvml). GpuIdle and the applications-clocks setting are harmless.
HARMLESS_THROTTLE_BITS = 0x1 | 0x2
THROTTLE_NAMES = {
    0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
    0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake",
}


def throttle_names(snap):
    """Active, non-harmless throttle reasons at the time of the snapshot."""
    if not snap or not snap.get("gpu"):
        return []
    raw = snap["gpu"].get("clocks_throttle_reasons.active")
    if not raw or not isinstance(raw, str):
        return []
    try:
        mask = int(raw, 16)
    except ValueError:
        return []
    mask &= ~HARMLESS_THROTTLE_BITS
    return [name for bit, name in THROTTLE_NAMES.items() if mask & bit] or ([hex(mask)] if mask else [])
