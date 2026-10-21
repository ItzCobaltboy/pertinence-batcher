# Small shared pieces: the profile table, the job record, JSON/CSV helpers and
# environment info. No torch import (analyze.py and the tests use this).

import hashlib
import json
import os
import platform
import subprocess
import sys

STATS = ["p50", "p95", "p99", "mean"]


class Profile:
    """T_i(b) table from calibrate.py's profile.json.

    t(model, b) returns milliseconds. Between profiled sizes it interpolates
    linearly; past the largest profiled size it extrapolates from the last two
    points (and warn_extrapolation() says so once). `part` picks what a batch
    costs: "total" (forward + NMS, the default when the run does NMS) or
    "forward"."""

    def __init__(self, data, stat="p50", part="total"):
        self.data = data
        self.stat = stat
        self.part = part
        self.models = list(data["models"].keys())
        self.batch_sizes = sorted(int(b) for b in data["batch_sizes"])
        self._extrapolated = set()

    @classmethod
    def load(cls, path, stat="p50", part="total"):
        with open(path) as f:
            return cls(json.load(f), stat=stat, part=part)

    @classmethod
    def from_table(cls, table, stat="p50"):
        """table = {model: {b: ms}}; every statistic set to the same value.
        Used by the tests and the fake runner."""
        models = {}
        sizes = set()
        for model, row in table.items():
            models[model] = {}
            for b, ms in row.items():
                sizes.add(int(b))
                entry = {s + "_ms": float(ms) for s in STATS}
                entry["std_ms"] = 0.0
                models[model][str(int(b))] = {"forward": entry, "nms": {s + "_ms": 0.0 for s in STATS},
                                              "total": entry}
        return cls({"models": models, "batch_sizes": sorted(sizes)}, stat=stat)

    def point(self, model, b, stat=None, part=None):
        entry = self.data["models"][model][str(int(b))][part or self.part]
        return float(entry[(stat or self.stat) + "_ms"])

    def t(self, model, b, stat=None, part=None):
        sizes = self.batch_sizes
        if b in sizes:
            return self.point(model, b, stat, part)
        if b < sizes[0]:
            return self.point(model, sizes[0], stat, part)
        for lo, hi in zip(sizes, sizes[1:]):
            if lo < b < hi:
                t_lo = self.point(model, lo, stat, part)
                t_hi = self.point(model, hi, stat, part)
                return t_lo + (t_hi - t_lo) * (b - lo) / (hi - lo)
        # past the largest profiled size
        self._extrapolated.add((model, b))
        if len(sizes) == 1:
            return self.point(model, sizes[0], stat, part) * b / sizes[0]
        lo, hi = sizes[-2], sizes[-1]
        t_lo = self.point(model, lo, stat, part)
        t_hi = self.point(model, hi, stat, part)
        return t_hi + (t_hi - t_lo) * (b - hi) / (hi - lo)

    def extrapolated(self):
        return sorted(self._extrapolated)

    def table(self, stat=None, part=None):
        return {m: {b: self.t(m, b, stat, part) for b in self.batch_sizes} for m in self.models}


def capacity_fps(profile, models, split, b):
    """Frames per second the device sustains if every queue always runs batches of
    b and frames are routed by `split`: one frame costs sum_i p_i T_i(b) / b."""
    ms_per_frame = 0.0
    for model, p in zip(models, split):
        ms_per_frame += p * profile.t(model, b) / b
    return 1000.0 / ms_per_frame


class Job:
    """One frame. Times are seconds on the run clock (see runtime.py)."""
    __slots__ = ["job_id", "camera", "queue", "t_target", "t_arrival", "t_actual", "deadline", "slo_ms",
                 "frame_ref", "t_enqueued", "payload", "decode_ms", "prep_ms", "model_run",
                 "promoted_from", "batch_id", "t_start", "t_end", "dropped", "t_drop"]

    def __init__(self, job_id, camera, queue, t_target, slo_ms, frame_ref):
        self.job_id = job_id
        self.camera = camera
        self.queue = queue              # queue it was routed to
        self.t_target = t_target        # when the camera produced it (trace time, run clock)
        self.t_arrival = t_target       # what policies see as the arrival time
        self.t_actual = None            # when the arrival thread actually woke up for it
        self.deadline = t_target + slo_ms / 1000.0
        self.slo_ms = slo_ms
        self.frame_ref = frame_ref
        self.t_enqueued = None          # when it actually entered its queue
        self.payload = None             # prepared tensor (prep arrival) or None
        self.decode_ms = 0.0
        self.prep_ms = 0.0
        self.model_run = queue          # after any promotion
        self.promoted_from = ""
        self.batch_id = -1
        self.t_start = None
        self.t_end = None
        self.dropped = False
        self.t_drop = None


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_cmd(cmd, timeout=10):
    """(returncode, stdout+stderr) of a command; (None, error text) if it can't start."""
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=timeout, universal_newlines=True)
        return out.returncode, out.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)


def git_commit():
    code, out = run_cmd(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"])
    return out.strip() if code == 0 else "unknown"


def versions():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["numpy", "pandas", "matplotlib", "torch", "torchvision", "ultralytics", "cv2", "PIL"]:
        mod = sys.modules.get(name)
        if mod is not None:
            info[name] = getattr(mod, "__version__", "?")
    return info


def is_jetson():
    return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/proc/device-tree/model") and \
        "jetson" in read_text("/proc/device-tree/model").lower()


def read_text(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\x00", "").strip()
    except OSError:
        return ""


def power_mode():
    """Jetson power mode (nvpmodel -q) if there is one, else None."""
    code, out = run_cmd(["nvpmodel", "-q"])
    if code != 0:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    mode = {"raw": " | ".join(lines)}
    for line in lines:
        if "Power Mode" in line:
            mode["name"] = line.split(":", 1)[1].strip()
        elif line.isdigit():
            mode["id"] = int(line)
    return mode
