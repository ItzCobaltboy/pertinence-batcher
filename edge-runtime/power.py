# Power backends. One interface for all of them:
#
#   sampler = make_sampler("auto" | "nvidia-smi" | "tegrastats" | "fake" | "none")
#   sampler.start(path)     start sampling in the background, raw data goes next to `path`
#   sampler.stop()          stop it (idempotent; also runs at exit)
#   read(path)              -> DataFrame [t, power_w, ...] on the host time.time() clock
#
# Cleanup: stop() is called from try/finally by the callers and registered with
# atexit. Sampler subprocesses are started with PR_SET_PDEATHSIG (Linux), so the
# kernel sends them SIGTERM if the process that started them dies, even on SIGKILL.
# PDEATHSIG follows the *thread* that started the child: start samplers from the
# main thread.

import atexit
import ctypes
import gzip
import json
import math
import os
import pty
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

import config


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _die_with_parent():
    # runs in the child between fork and exec
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass


def popen_tied(cmd, **kwargs):
    """subprocess.Popen whose child dies with this process (Linux)."""
    if sys.platform.startswith("linux"):
        kwargs["preexec_fn"] = _die_with_parent
    return subprocess.Popen(cmd, **kwargs)


def _stop_process(proc, sig=signal.SIGINT):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(sig)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except OSError:
        pass


def _meta_path(path):
    return path + ".meta.json"


def _write_meta(path, meta):
    with open(_meta_path(path), "w") as f:
        json.dump(meta, f, indent=1)


def read_meta(path):
    try:
        with open(_meta_path(path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def compress_raw(path):
    """<path>.raw -> <path>.raw.gz (tegrastats traces are ~1 MB per minute raw)."""
    raw = path + ".raw"
    if os.path.exists(raw):
        with open(raw, "rb") as f_in, gzip.open(raw + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(raw)


def _open_raw(path):
    if os.path.exists(path + ".raw.gz"):
        return gzip.open(path + ".raw.gz", "rt")
    return open(path + ".raw")


def _has_raw(path):
    return os.path.exists(path + ".raw") or os.path.exists(path + ".raw.gz")


def _env_cmd(var, default):
    value = os.environ.get(var)
    return shlex.split(value) if value else list(default)


# ---------------------------------------------------------------------------
# nvidia-smi (laptop, A100)
# ---------------------------------------------------------------------------
NVSMI_COLUMNS = {
    "power.draw.instant": "power_w",
    "power.draw": "power_w",
    "clocks.sm": "sm_clock_mhz",
    "temperature.gpu": "temp_c",
    "utilization.gpu": "util_pct",
    "clocks_throttle_reasons.active": "throttle",
    "clocks_event_reasons.active": "throttle",
}


class NvidiaSmiSampler:
    tail_s = config.NVIDIA_SMI_TAIL_S
    backend = "nvidia-smi"

    def __init__(self, gpu_index="0", sample_ms=config.SAMPLE_MS):
        self.cmd = _env_cmd("EDGE_RUNTIME_NVIDIA_SMI", ["nvidia-smi"])
        self.gpu_index = str(gpu_index)
        self.sample_ms = sample_ms
        self.proc = None
        self.path = None
        self.fields = None

    def _field_works(self, field):
        try:
            out = subprocess.run(self.cmd + ["-i", self.gpu_index, "--query-gpu=" + field,
                                             "--format=csv,noheader,nounits"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        text = out.stdout.strip()
        return out.returncode == 0 and text != "" and "valid" not in text.lower() and "N/A" not in text

    def probe(self):
        """Which fields this driver supports. power.draw.instant is preferred
        (power.draw is a ~1 s average on some GPUs)."""
        fields = []
        fields.append("power.draw.instant" if self._field_works("power.draw.instant") else "power.draw")
        for f in ["clocks.sm", "temperature.gpu", "utilization.gpu"]:
            if self._field_works(f):
                fields.append(f)
        for f in ["clocks_throttle_reasons.active", "clocks_event_reasons.active"]:
            if self._field_works(f):
                fields.append(f)
                break
        return fields

    def start(self, path):
        self.path = path
        self.fields = self.probe()
        cmd = self.cmd + ["-i", self.gpu_index, "--query-gpu=timestamp," + ",".join(self.fields),
                          "--format=csv,noheader,nounits", "-lms", str(self.sample_ms), "-f", path + ".raw"]
        self.proc = popen_tied(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.stop)
        _write_meta(path, {"backend": self.backend, "fields": self.fields, "cmd": cmd,
                           "power_field": self.fields[0], "sample_ms": self.sample_ms})

    def stop(self):
        _stop_process(self.proc, signal.SIGINT)   # SIGINT makes nvidia-smi flush its file
        self.proc = None
        if self.path:
            compress_raw(self.path)


def _read_nvidia_smi(path, meta):
    fields = meta.get("fields", ["power.draw"])
    cols = [NVSMI_COLUMNS.get(f, f) for f in fields]
    rows = []
    if not _has_raw(path):
        return _empty()
    with _open_raw(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(fields) + 1:
                continue
            try:
                # local time on this machine -> the same seconds time.time() gives
                t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            row = {"t": t}
            for col, value in zip(cols, parts[1:]):
                if col == "throttle":
                    row[col] = value
                    continue
                try:
                    row[col] = float(value)
                except ValueError:
                    row[col] = float("nan")
            rows.append(row)
    return _frame(rows)


# ---------------------------------------------------------------------------
# tegrastats (Jetson)
# ---------------------------------------------------------------------------
_PAIR = re.compile(r"^(\d+)(mW)?/(\d+)(mW)?$")
_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FREQ = re.compile(r"^(\d+)%(?:@\[?([\d,]+)\]?)?$")
_TEMP = re.compile(r"^([A-Za-z0-9_]+)@(-?[\d.]+)C$")
_DATE = re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$|^\d{1,2}:\d{2}:\d{2}$")
_MB = re.compile(r"^(\d+)/(\d+)MB$")
RAIL_PREFIXES = ("VDD", "POM_", "VIN")
BOARD_RAILS = ["VDD_IN", "POM_5V_IN"]   # whole-board input rail, newer / older boards


def parse_tegrastats_line(line):
    """One tegrastats line -> dict. Power rails become rail_<NAME>_w (current
    value, watts); the board power is VDD_IN (or POM_5V_IN on older boards), else
    the sum of all rails, and power_source says which. Unknown tokens are kept as
    text in `unknown`, never an error."""
    tokens = line.split()
    out = {}
    rails = {}
    unknown = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        pair = _PAIR.match(nxt)
        if pair and _NAME.match(tok) and (pair.group(2) or tok.startswith(RAIL_PREFIXES)):
            rails[tok] = int(pair.group(1)) / 1000.0
            i += 2
            continue
        if tok in ("GR3D_FREQ", "EMC_FREQ") and _FREQ.match(nxt):
            m = _FREQ.match(nxt)
            key = "gpu" if tok == "GR3D_FREQ" else "emc"
            out[key + "_util_pct"] = float(m.group(1))
            if m.group(2):
                out[key + "_freq_mhz"] = float(m.group(2).split(",")[0])
            i += 2
            continue
        if tok == "CPU" and nxt.startswith("["):
            loads = []
            freqs = []
            for entry in nxt.strip("[]").split(","):
                m = re.match(r"^(\d+)%@(\d+)$", entry)
                if m:
                    loads.append(float(m.group(1)))
                    freqs.append(float(m.group(2)))
            if loads:
                out["cpu_load_pct"] = sum(loads) / len(loads)
                out["cpu_freq_mhz"] = sum(freqs) / len(freqs)
                out["cpu_cores_on"] = len(loads)
            i += 2
            continue
        if tok in ("RAM", "SWAP") and _MB.match(nxt):
            out[tok.lower() + "_used_mb"] = float(_MB.match(nxt).group(1))
            i += 2
            continue
        if tok.startswith("("):
            # "(lfb 2x4MB)", "(cached 0MB)": skip the whole parenthesised group
            while i < len(tokens) and not tokens[i].endswith(")"):
                i += 1
            i += 1
            continue
        temp = _TEMP.match(tok)
        if temp:
            value = float(temp.group(2))
            if value > -200:          # -256C = sensor off
                out["temp_" + temp.group(1).lower() + "_c"] = value
            i += 1
            continue
        if _DATE.match(tok):
            i += 1
            continue
        unknown.append(tok)
        i += 1

    for name, watts in rails.items():
        out["rail_" + name + "_w"] = watts
    board = [r for r in BOARD_RAILS if r in rails]
    if board:
        out["power_w"] = rails[board[0]]
        out["power_source"] = board[0]
    elif rails:
        out["power_w"] = sum(rails.values())
        out["power_source"] = "sum:" + "+".join(sorted(rails))
    else:
        out["power_w"] = float("nan")
        out["power_source"] = "none"
    if out.get("temp_gpu_c") is None:
        for k in ("temp_tj_c", "temp_soc0_c"):
            if k in out:
                out["temp_c"] = out[k]
                break
    else:
        out["temp_c"] = out["temp_gpu_c"]
    out["unknown"] = " ".join(unknown)
    return out


class TegrastatsSampler:
    """Runs `tegrastats --interval N` on a pseudo-terminal (so it line-buffers
    instead of block-buffering into a pipe) and stamps every line with
    time.time() when it is read."""
    tail_s = config.TEGRASTATS_TAIL_S
    backend = "tegrastats"

    def __init__(self, sample_ms=config.SAMPLE_MS):
        self.cmd = _env_cmd("EDGE_RUNTIME_TEGRASTATS", ["tegrastats"])
        self.sample_ms = sample_ms
        self.proc = None
        self.thread = None
        self.master = None
        self.path = None

    def start(self, path):
        self.path = path
        cmd = self.cmd + ["--interval", str(self.sample_ms)]
        master, slave = pty.openpty()
        self.proc = popen_tied(cmd, stdout=slave, stderr=slave, stdin=subprocess.DEVNULL)
        os.close(slave)
        self.master = master
        self.thread = threading.Thread(target=self._reader, args=(master, path + ".raw"), daemon=True)
        self.thread.start()
        atexit.register(self.stop)
        _write_meta(path, {"backend": self.backend, "cmd": cmd, "sample_ms": self.sample_ms})

    def _reader(self, fd, raw_path):
        buf = b""
        with open(raw_path, "w") as out:
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break          # EIO once the child is gone
                if not chunk:
                    break
                now = time.time()
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", "replace").strip()
                    if text:
                        out.write("%.4f\t%s\n" % (now, text))
                out.flush()

    def stop(self):
        _stop_process(self.proc, signal.SIGTERM)
        self.proc = None
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None
        if self.master is not None:
            try:
                os.close(self.master)
            except OSError:
                pass
            self.master = None
        if self.path and _has_raw(self.path):
            meta = read_meta(self.path)
            df = read(self.path)
            meta.update(cadence(df))
            meta["columns_seen"] = list(df.columns)     # what this board / image reports
            _write_meta(self.path, meta)
            compress_raw(self.path)


def _read_tegrastats(path):
    rows = []
    if not _has_raw(path):
        return _empty()
    with _open_raw(path) as f:
        for line in f:
            stamp, _, text = line.rstrip("\n").partition("\t")
            try:
                t = float(stamp)
            except ValueError:
                continue
            row = parse_tegrastats_line(text)
            row["t"] = t
            rows.append(row)
    return _frame(rows)


# ---------------------------------------------------------------------------
# fake (tests): a thread writing power = power_fn(t)
# ---------------------------------------------------------------------------
class FakeSampler:
    tail_s = 0.0
    backend = "fake"

    def __init__(self, power_fn=None, sample_ms=config.SAMPLE_MS):
        self.power_fn = power_fn or (lambda t: 5.0 + 2.0 * math.sin(t))
        self.sample_ms = sample_ms
        self._stop = threading.Event()
        self.thread = None
        self.path = None

    def start(self, path):
        self.path = path
        self._stop.clear()
        _write_meta(path, {"backend": self.backend, "sample_ms": self.sample_ms})
        self.thread = threading.Thread(target=self._loop, args=(path + ".raw",), daemon=True)
        self.thread.start()
        atexit.register(self.stop)

    def _loop(self, raw_path):
        with open(raw_path, "w") as f:
            f.write("t,power_w\n")
            while not self._stop.is_set():
                t = time.time()
                f.write("%.6f,%.6f\n" % (t, self.power_fn(t)))
                f.flush()
                self._stop.wait(self.sample_ms / 1000.0)

    def stop(self):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None
        if self.path:
            compress_raw(self.path)


class NoSampler:
    """No power source (CPU-only machine): writes the meta file, nothing else."""
    tail_s = 0.0
    backend = "none"

    def start(self, path):
        _write_meta(path, {"backend": self.backend})

    def stop(self):
        pass


def make_sampler(kind="auto", gpu_index="0"):
    if kind == "auto":
        from common import is_jetson
        if is_jetson() and shutil.which("tegrastats"):
            kind = "tegrastats"
        elif shutil.which("nvidia-smi"):
            kind = "nvidia-smi"
        else:
            kind = "none"
    if kind == "nvidia-smi":
        return NvidiaSmiSampler(gpu_index)
    if kind == "tegrastats":
        return TegrastatsSampler()
    if kind == "fake":
        return FakeSampler()
    if kind == "none":
        return NoSampler()
    raise ValueError("unknown power backend: %s" % kind)


# ---------------------------------------------------------------------------
# reading + energy
# ---------------------------------------------------------------------------
def _empty():
    import pandas as pd
    return pd.DataFrame({"t": [], "power_w": []})


def _frame(rows):
    import pandas as pd
    if not rows:
        return _empty()
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def read(path):
    """The power trace written by any backend at `path`, as a DataFrame sorted by t."""
    meta = read_meta(path)
    backend = meta.get("backend")
    if backend == "nvidia-smi":
        return _read_nvidia_smi(path, meta)
    if backend == "tegrastats":
        return _read_tegrastats(path)
    if backend == "fake" and _has_raw(path):
        import pandas as pd
        with _open_raw(path) as f:
            return pd.read_csv(f)
    return _empty()


def cadence(df):
    """Effective sampling interval of a trace, and how bursty the stamps are."""
    if len(df) < 3:
        return {"samples": int(len(df))}
    dt = np.diff(df["t"].values) * 1000.0
    return {"samples": int(len(df)), "median_interval_ms": float(np.median(dt)),
            "p95_interval_ms": float(np.percentile(dt, 95)),
            "burst_fraction": float(np.mean(dt < 5.0))}


def energy_j(t, p, t0, t1):
    """Integral of power over [t0, t1] (trapezoid, samples linearly
    interpolated at the window edges). NaN without samples in or around the window."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    t, p = t[ok], p[ok]
    if t1 <= t0 or len(t) < 2 or t[0] > t1 or t[-1] < t0:
        return float("nan")
    inside = (t > t0) & (t < t1)
    grid = np.concatenate([[t0], t[inside], [t1]])
    values = np.interp(grid, t, p)
    return float(np.sum((values[1:] + values[:-1]) * np.diff(grid)) / 2.0)


def mean_power(t, p, t0, t1):
    e = energy_j(t, p, t0, t1)
    return e / (t1 - t0) if t1 > t0 else float("nan")
