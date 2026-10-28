# GPU frequency control through the devfreq sysfs node (AGX Xavier; the nvpmodel mode is never touched).
# pin(f) writes min_freq and max_freq so both equal f; restore() writes the recorded originals back.
# Needs root. Governor, 3D scaling, fan, nvpmodel and jetson_clocks are only read, never written.
# Manual use:  sudo python3 -m dvfs.gpu --show | --restore
import atexit
import json
import os
import sys
import time

from dvfs import settings

DEV = settings.GPU_DEVFREQ


def attr(name):
    return os.path.join(DEV, name)


def read_attr(path):
    with open(path) as f:
        return f.read().strip()


def write_attr(path, value):
    """One sysfs write. Returns (t_start, t_end) on time.monotonic(); the kernel validates it on close."""
    t0 = time.monotonic()
    with open(path, "w") as f:
        f.write("%s\n" % value)
    return t0, time.monotonic()


def read_state():
    state = {k: read_attr(attr(k)) for k in ("min_freq", "max_freq", "cur_freq", "governor", "available_frequencies")}
    state["enable_3d_scaling"] = read_attr(settings.GPU_3D_SCALING) if os.path.exists(settings.GPU_3D_SCALING) else None
    state["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return state


def restore_state(original):
    """max_freq first (the ceiling can only go up or stay, so min can follow), then min_freq. True if the read-back matches."""
    ok = True
    for name in ("max_freq", "min_freq"):
        try:
            write_attr(attr(name), original[name])
        except OSError as e:
            ok = False
            print("COULD NOT WRITE %s: %s" % (name, e))
    now = read_state()
    for key in ("min_freq", "max_freq", "governor", "enable_3d_scaling"):
        if str(now[key]) != str(original[key]):
            ok = False
            print("AFTER RESTORE %s is %s, original was %s" % (key, now[key], original[key]))
    if not ok:
        print("MANUAL FIX: echo %s > %s ; echo %s > %s" % (original["max_freq"], attr("max_freq"), original["min_freq"], attr("min_freq")))
    return ok


class Gpu:
    """Owns the GPU clock for one session. The original state goes to backup_path before the first write.
    An existing backup means an earlier session died with clocks pinned: it is kept, it holds the real original."""

    def __init__(self, backup_path):
        self.backup_path = backup_path
        self.restored = True        # nothing to restore until the backup exists
        self.available = sorted(int(x) for x in read_attr(attr("available_frequencies")).split())
        if os.path.exists(backup_path):
            with open(backup_path) as f:
                self.original = json.load(f)
            print("NOTICE: %s exists (an earlier session did not restore); using it as the original state" % backup_path)
        else:
            self.original = read_state()
            with open(backup_path, "w") as f:
                json.dump(self.original, f, indent=2)
        self.cap_hz = int(self.original["max_freq"])
        self.restored = False
        atexit.register(self.restore)

    def check_freqs(self, freqs_hz):
        bad = [f for f in freqs_hz if f not in self.available or f > self.cap_hz]
        if bad:
            raise ValueError("FREQS_HZ %s are not usable: available %s, mode cap (original max_freq) %d"
                             % (bad, self.available, self.cap_hz))

    def cur_freq(self):
        return int(read_attr(attr("cur_freq")))

    def pin(self, freq_hz):
        """min_freq = max_freq = freq_hz. Raising writes max first, lowering writes min first, so min never exceeds max."""
        self.check_freqs([freq_hz])
        order = ["max_freq", "min_freq"] if freq_hz >= int(read_attr(attr("max_freq"))) else ["min_freq", "max_freq"]
        writes = [(name,) + write_attr(attr(name), freq_hz) for name in order]
        return {"order": order, "t_start": writes[0][1], "t_end": writes[-1][2],
                "write_ms": [(w[2] - w[1]) * 1000.0 for w in writes]}

    def wait_cur(self, freq_hz, timeout_s):
        """Poll cur_freq until it equals freq_hz. Returns (monotonic time of the first match or None, last value read)."""
        t0 = time.monotonic()
        while True:
            value = self.cur_freq()
            t = time.monotonic()
            if value == freq_hz:
                return t, value
            if t - t0 > timeout_s:
                return None, value
            time.sleep(0.0002)

    def restore(self):
        if self.restored:
            return True
        self.restored = True
        print("restoring GPU clocks: min %s max %s" % (self.original["min_freq"], self.original["max_freq"]))
        ok = restore_state(self.original)
        if ok:
            os.replace(self.backup_path, "%s.restored_%s" % (self.backup_path, time.strftime("%Y%m%d_%H%M%S")))
        return ok


def main():
    if "--show" in sys.argv:
        print(json.dumps(read_state(), indent=2))
    elif "--restore" in sys.argv:
        if os.geteuid() != 0:
            sys.exit("run with sudo: writing devfreq files needs root")
        backup = os.path.join(settings.RESULTS_DIR, "gpu_state_backup.json")
        if not os.path.exists(backup):
            sys.exit("no backup at %s: nothing to restore (current state: python3 -m dvfs.gpu --show)" % backup)
        with open(backup) as f:
            original = json.load(f)
        if restore_state(original):
            os.replace(backup, "%s.restored_%s" % (backup, time.strftime("%Y%m%d_%H%M%S")))
            print("restored")
    else:
        sys.exit("usage: python3 -m dvfs.gpu --show | --restore")


if __name__ == "__main__":
    main()
