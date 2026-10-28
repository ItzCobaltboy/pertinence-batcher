# Safety for a profiling session: refuse to start next to another benchmark, and abort (the session then
# restores the clocks) on high temperature, a clock that moved off the pinned value, a new over-current
# count, or new thermal/throttle lines in dmesg. NVIDIA documents that hardware throttling is not reported to
# the OS, so a changed cur_freq and the over-current counters are the only signals besides temperature.
import glob
import os
import re
import subprocess
import time

from dvfs import settings


class SafetyAbort(Exception):
    pass


def run(cmd):
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def hot_c(parsed):
    """Highest temperature in one parsed tegrastats line, PMIC excluded. None if the line has none."""
    temps = [v for k, v in parsed.get("temps", {}).items() if k not in settings.TEMP_IGNORE]
    return max(temps) if temps else None


def busy_processes():
    """Other python processes and tegrastats already running (own pid and the sudo parent are skipped)."""
    mine = {os.getpid(), os.getppid()}
    found = []
    for cmd in (["pgrep", "-af", "python"], ["pgrep", "-af", "tegrastats"]):
        for line in run(cmd)[1].splitlines():
            pid = line.split(" ", 1)[0]
            if pid.isdigit() and int(pid) not in mine and line not in found:
                found.append(line)
    return found


class Guard:
    def __init__(self, sampler, gpu):
        self.sampler = sampler
        self.gpu = gpu
        self.target_hz = None       # while set, cur_freq must equal it
        self.pattern = re.compile(settings.DMESG_PATTERN, re.IGNORECASE)
        self.oc_base = self._oc()
        self.dmesg_base = set(self._dmesg())
        self.last_slow = time.monotonic()

    def set_target(self, freq_hz):
        self.target_hz = freq_hz

    def _oc(self):
        counts = {}
        for path in glob.glob(settings.HWMON_OC_GLOB):
            try:
                with open(path) as f:
                    counts[path] = int(f.read().strip())
            except (OSError, ValueError):
                pass
        return counts

    def _dmesg(self):
        code, text = run(["dmesg"])
        return text.splitlines() if code == 0 else []

    def check(self):
        now = time.monotonic()
        latest = self.sampler.latest()
        if latest is None or now - latest[0] > settings.SAMPLER_STALE_S:
            raise SafetyAbort("tegrastats stopped delivering samples (temperature is unwatched)")
        temp = hot_c(latest[1])
        if temp is not None and temp > settings.TEMP_CAP_C:
            raise SafetyAbort("temperature %.1f C above the cap of %s C" % (temp, settings.TEMP_CAP_C))
        if self.target_hz is not None:
            cur = self.gpu.cur_freq()
            if cur != self.target_hz:
                raise SafetyAbort("GPU clock moved: cur_freq %d while %d is pinned (throttle?)" % (cur, self.target_hz))
        if now - self.last_slow >= settings.GUARD_PERIOD_S:
            self.last_slow = now
            for path, count in self._oc().items():
                if count > self.oc_base.get(path, count):
                    raise SafetyAbort("over-current counter rose: %s %d -> %d" % (path, self.oc_base[path], count))
            new = [l for l in self._dmesg() if l not in self.dmesg_base and self.pattern.search(l)]
            if new:
                raise SafetyAbort("new dmesg lines: " + " | ".join(new[:3]))

    def sleep(self, seconds):
        """Sleep in short slices, checking safety between them."""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.5, left))
            self.check()
