# Copied from sched-harness/common/tegrastats.py, then extended for dvfs: the Sampler also measures, in parallel with
# the power lines, the GPU clock actually held (sysfs cur_freq, polled fast) and any new dmesg lines, flagging the ones
# that look like throttling. Nothing here stops a run; it only records (samples, raw aux log, window queries).
# tegrastats sampler (runs the tool in a pty so lines arrive one by one), line parser,
# and the power/energy helpers that cut the trace with wall-clock marks.
# Every line is stamped with time.monotonic() when read; tegrastats' own timestamp is ignored.
import bisect
import gzip
import os
import re
import subprocess
import threading
import time

import numpy as np

RAIL_RE = re.compile(r"\b([A-Z][A-Z0-9_]*) (\d+)mW/(\d+)mW")
GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+))?")
EMC_RE = re.compile(r"EMC_FREQ (\d+)%(?:@(\d+))?")
CPU_RE = re.compile(r"CPU \[([^\]]*)\]")
TEMP_RE = re.compile(r"\b([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C")
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")


def parse_line(line):
    """One tegrastats line -> dict. Missing fields are simply absent; nothing raises."""
    out = {"raw": line}
    # name -> (current_mw, average_mw). Without root there are no rail fields at all.
    out["rails"] = {n: (int(c), int(a)) for n, c, a in RAIL_RE.findall(line)}
    m = GR3D_RE.search(line)
    if m:
        out["gr3d_load"] = int(m.group(1))
        out["gr3d_mhz"] = int(m.group(2)) if m.group(2) else None
    m = EMC_RE.search(line)
    if m:
        out["emc_load"] = int(m.group(1))
        out["emc_mhz"] = int(m.group(2)) if m.group(2) else None
    m = CPU_RE.search(line)
    if m:
        out["cpu"] = m.group(1).split(",")      # e.g. "1%@1180" or "off", kept as text
    out["temps"] = {n: float(v) for n, v in TEMP_RE.findall(line)}
    m = RAM_RE.search(line)
    if m:
        out["ram_used_mb"] = int(m.group(1))
        out["ram_total_mb"] = int(m.group(2))
    return out


def powers(parsed, rails):
    """Parsed line -> dict of mW columns: each rail, total, total_excl_sys5v, gpu.
    Returns None when a rail is missing (not root, or an unexpected line)."""
    cur = parsed["rails"]
    if any(r not in cur for r in rails):
        return None
    out = {r: cur[r][0] for r in rails}
    out["total"] = sum(out[r] for r in rails)
    out["total_excl_sys5v"] = out["total"] - out.get("SYS5V", 0)
    out["gpu"] = out["GPU"]
    return out


def max_temp(parsed, ignore=()):
    """Highest temperature in one parsed line, names in `ignore` left out. None if the line has none."""
    temps = [v for k, v in parsed.get("temps", {}).items() if k not in ignore]
    return max(temps) if temps else None


class Sampler:
    def __init__(self, interval_ms, rails, raw_log_path=None, cur_freq_path=None, cur_freq_interval_s=0.02,
                 dmesg_pattern=None, dmesg_interval_s=1.0, aux_log_path=None):
        self.interval_ms = interval_ms
        self.rails = rails
        self.raw_log_path = raw_log_path
        # parallel measurement of the held clock and of throttle messages (both optional)
        self.cur_freq_path = cur_freq_path
        self.cur_freq_interval_s = cur_freq_interval_s
        self.dmesg_re = re.compile(dmesg_pattern, re.IGNORECASE) if dmesg_pattern else None
        self.dmesg_interval_s = dmesg_interval_s
        self.aux_log_path = aux_log_path
        self.freq_samples = []   # list of (t_monotonic, cur_freq_hz)
        self.dmesg_lines = []    # list of (t_monotonic seen, line, matches the throttle pattern)
        self._dmesg_seen = set()
        self.samples = []        # list of (t_monotonic, parsed dict, powers dict)
        self.bad_lines = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        # a pty makes tegrastats line-buffer; a plain pipe would deliver ~4 KB chunks
        master, slave = os.openpty()
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=slave, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        os.close(slave)
        self._master = master
        self._log = gzip.open(self.raw_log_path, "at") if self.raw_log_path else None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._aux_log = None
        self._aux_thread = None
        if self.cur_freq_path or self.dmesg_re:
            self._aux_log = gzip.open(self.aux_log_path, "at") if self.aux_log_path else None
            if self.dmesg_re:
                self._dmesg_seen = set(self._dmesg())     # lines already there at the start are not events
            self._aux_thread = threading.Thread(target=self._aux_loop, daemon=True)
            self._aux_thread.start()

    def _dmesg(self):
        p = subprocess.run(["dmesg"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
        return p.stdout.splitlines() if p.returncode == 0 else []

    def _aux_write(self, t, kind, value):
        if self._aux_log:
            self._aux_log.write("%.4f %s %s\n" % (t, kind, value))

    def _aux_loop(self):
        next_dmesg = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic()
            if self.cur_freq_path:
                try:
                    with open(self.cur_freq_path) as f:
                        value = int(f.read().strip())
                except (OSError, ValueError):
                    value = None
                if value is not None:
                    with self._lock:
                        self.freq_samples.append((t, value))
                    self._aux_write(t, "cur_freq", value)
            if self.dmesg_re and t >= next_dmesg:
                next_dmesg = t + self.dmesg_interval_s
                for line in self._dmesg():
                    if line in self._dmesg_seen:
                        continue
                    self._dmesg_seen.add(line)
                    hit = bool(self.dmesg_re.search(line))
                    with self._lock:
                        self.dmesg_lines.append((t, line, hit))
                    self._aux_write(t, "dmesg_hit" if hit else "dmesg", line)
                if self._aux_log:
                    self._aux_log.flush()
            self._stop.wait(self.cur_freq_interval_s)

    def _read_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = os.read(self._master, 4096)
            except OSError:           # pty closed
                break
            if not chunk:
                break
            t = time.monotonic()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._add(t, line.decode("ascii", "replace").strip())

    def _add(self, t, line):
        if not line:
            return
        if self._log:
            self._log.write("%.3f %s\n" % (t, line))
            self._log.flush()       # so a power cut loses at most the last line, not a buffered chunk
        parsed = parse_line(line)
        p = powers(parsed, self.rails)
        if p is None:
            self.bad_lines += 1
            return
        with self._lock:
            self.samples.append((t, parsed, p))

    def stop(self):
        self._stop.set()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        os.close(self._master)
        self._thread.join(timeout=5)
        if self._log:
            self._log.close()
        if self._aux_thread:
            self._aux_thread.join(timeout=5)
        if self._aux_log:
            self._aux_log.close()

    def window(self, t0, t1, pad=0.5):
        """Samples with t0-pad <= t <= t1+pad (the pad gives neighbours for interpolation)."""
        with self._lock:
            return [s for s in self.samples if t0 - pad <= s[0] <= t1 + pad]

    def freq_window(self, t0, t1):
        """cur_freq reads (t, hz) with t0 <= t <= t1."""
        with self._lock:
            return [x for x in self.freq_samples if t0 <= x[0] <= t1]

    def dmesg_hits(self, t0, t1):
        """New dmesg lines that matched the throttle pattern, seen in [t0, t1]. dmesg is polled every dmesg_interval_s."""
        with self._lock:
            return [x for x in self.dmesg_lines if t0 <= x[0] <= t1 and x[2]]

    def trim(self, before_t):
        """Drop samples older than before_t from memory (the raw logs on disk keep everything)."""
        with self._lock:
            i = bisect.bisect_left([s[0] for s in self.samples], before_t)
            del self.samples[:i]
            j = bisect.bisect_left([s[0] for s in self.freq_samples], before_t)
            del self.freq_samples[:j]

    def latest(self):
        with self._lock:
            return self.samples[-1] if self.samples else None

    def cadence_hz(self, t0=None, t1=None):
        with self._lock:
            ts = [s[0] for s in self.samples if (t0 is None or s[0] >= t0) and (t1 is None or s[0] <= t1)]
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])


def energy_mj(times, mw, t0, t1):
    """Trapezoid integral of power (mW) over [t0, t1] -> mJ. The curve is interpolated
    at t0 and t1, and no sample outside the window is used except as an interpolation
    neighbour. Needs at least two samples."""
    times = np.asarray(times, dtype=float)
    mw = np.asarray(mw, dtype=float)
    inside = (times > t0) & (times < t1)
    xs = np.concatenate([[t0], times[inside], [t1]])
    ys = np.concatenate([[np.interp(t0, times, mw)], mw[inside], [np.interp(t1, times, mw)]])
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))


def mean_power(samples, key, t0, t1):
    """Time-weighted mean power (mW) of column `key` over [t0, t1]."""
    times = [s[0] for s in samples]
    values = [s[2][key] for s in samples]
    return energy_mj(times, values, t0, t1) / (t1 - t0)
