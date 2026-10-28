# Run-time checks of a profiling session, only two: the GPU clock stays where it was pinned, and dmesg shows no new
# over-current / throttle / thermal lines. Either one aborts the run (the session then restores the clocks). Both are
# also counted, so every subrun row records that the checks ran and what cur_freq did. Also: refuse to start next
# to another benchmark.
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
    """Highest temperature in one parsed tegrastats line, PMIC excluded. None if the line has none. Recorded, never acted on."""
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
    def __init__(self, gpu):
        self.gpu = gpu
        self.target_hz = None       # while set, cur_freq must equal it
        self.pattern = re.compile(settings.DMESG_PATTERN, re.IGNORECASE)
        self.dmesg_base = set(self._dmesg())
        self.last_dmesg = time.monotonic()
        self.dmesg_checks = 0
        self.reset_stats()

    def set_target(self, freq_hz):
        self.target_hz = freq_hz

    def reset_stats(self):
        """Start counting cur_freq reads for the next subrun."""
        self.freq_reads = 0
        self.freq_min = None
        self.freq_max = None
        self.dmesg_checks_at_reset = self.dmesg_checks

    def stats(self):
        return {"n_cur_freq_reads": self.freq_reads, "cur_freq_min_hz": self.freq_min, "cur_freq_max_hz": self.freq_max,
                "n_dmesg_checks": self.dmesg_checks - self.dmesg_checks_at_reset}

    def _dmesg(self):
        code, text = run(["dmesg"])
        return text.splitlines() if code == 0 else []

    def check_dmesg(self):
        self.last_dmesg = time.monotonic()
        self.dmesg_checks += 1
        new = [l for l in self._dmesg() if l not in self.dmesg_base and self.pattern.search(l)]
        if new:
            raise SafetyAbort("new dmesg lines: " + " | ".join(new[:3]))

    def check(self):
        if self.target_hz is not None:
            cur = self.gpu.cur_freq()
            self.freq_reads += 1
            self.freq_min = cur if self.freq_min is None else min(self.freq_min, cur)
            self.freq_max = cur if self.freq_max is None else max(self.freq_max, cur)
            if cur != self.target_hz:
                raise SafetyAbort("GPU clock moved: cur_freq %d while %d is pinned" % (cur, self.target_hz))
        if time.monotonic() - self.last_dmesg >= settings.GUARD_PERIOD_S:
            self.check_dmesg()

    def sleep(self, seconds):
        """Sleep in short slices, checking between them."""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.5, left))
            self.check()
