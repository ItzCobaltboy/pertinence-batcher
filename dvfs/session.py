# One profiling session: checks (root, mode, nothing else running), backs up the GPU clock state, validates the
# frequency list, starts the tegrastats sampler and the safety guard, and on any exit (normal, error, Ctrl-C,
# SIGTERM, ssh hangup, abort) stops the sampler and puts the clocks back. Used by profile_switch and profile_models.
import json
import os
import platform
import signal
import sys
import time

from dvfs import settings, tegrastats
from dvfs.gpu import Gpu
from dvfs.guard import Guard, SafetyAbort, busy_processes, run


def current_mode():
    """`nvpmodel -q` prints 'NV Power Mode: NAME' then the id on the last line (read only)."""
    code, text = run(["nvpmodel", "-q"])
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if code != 0 or not lines or not lines[-1].isdigit():
        raise RuntimeError("could not read the power mode from nvpmodel -q:\n" + text)
    return int(lines[-1]), text.strip()


def chown_to_invoker(path):
    """After running under sudo, give files back to the user who called sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None or not os.path.exists(path):
        return
    os.chown(path, int(uid), int(gid))
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            os.chown(os.path.join(root, name), int(uid), int(gid))


class Session:
    def __init__(self, out_dir, ignore_busy=False):
        self.out_dir = out_dir
        self.ignore_busy = ignore_busy
        self.gpu = None
        self.sampler = None
        self.guard = None

    def __enter__(self):
        if os.geteuid() != 0:
            sys.exit("run with sudo: writing the GPU clock files and reading tegrastats rails need root")
        mode_id, mode_text = current_mode()
        if settings.EXPECTED_MODE_ID is not None and mode_id != settings.EXPECTED_MODE_ID:
            sys.exit("nvpmodel mode is %d, expected %d (settings.EXPECTED_MODE_ID). This tool never changes the mode."
                     % (mode_id, settings.EXPECTED_MODE_ID))
        busy = busy_processes()
        if busy and not self.ignore_busy:
            sys.exit("another benchmark may be running (use --ignore-busy if these are harmless):\n  " + "\n  ".join(busy))
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)
        try:
            self.gpu = Gpu(os.path.join(settings.RESULTS_DIR, "gpu_state_backup.json"))
            self.gpu.check_freqs(settings.FREQS_HZ)
            for sig in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, lambda s, f: sys.exit(128 + s))      # SystemExit unwinds through __exit__
            self.sampler = tegrastats.Sampler(settings.TEGRASTATS_INTERVAL_MS, settings.RAILS,
                                              os.path.join(self.out_dir, "tegrastats_raw.log.gz"))
            self.sampler.start()
            deadline = time.monotonic() + 5
            while self.sampler.latest() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if self.sampler.latest() is None:
                raise RuntimeError("no tegrastats samples with power rails in 5 s (%d lines without rails)" % self.sampler.bad_lines)
            self.guard = Guard(self.sampler, self.gpu)
            self.write_meta(mode_id, mode_text)
        except BaseException:
            self.close()
            raise
        return self

    def write_meta(self, mode_id, mode_text):
        meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "python": platform.python_version(),
                "nvpmodel": mode_text, "mode_id": mode_id, "gpu_original_state": self.gpu.original,
                "available_frequencies": self.gpu.available, "cap_hz": self.gpu.cap_hz,
                "jetson_clocks_show": run(["jetson_clocks", "--show"])[1],
                "settings": {k: v for k, v in vars(settings).items() if k.isupper()}}
        try:
            import torch
            meta["torch"] = torch.__version__
        except ImportError:
            pass
        with open(os.path.join(self.out_dir, "meta_%s.json" % time.strftime("%Y%m%d_%H%M%S")), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def close(self):
        if self.sampler is not None:
            try:
                self.sampler.stop()
            except Exception as e:
                print("could not stop tegrastats cleanly:", e)
        if self.gpu is not None:
            self.gpu.restore()
        chown_to_invoker(self.out_dir)
        chown_to_invoker(settings.RESULTS_DIR)

    def __exit__(self, exc_type, exc, tb):
        if isinstance(exc, SafetyAbort):
            print("SAFETY ABORT:", exc)
            with open(os.path.join(self.out_dir, "abort.txt"), "a") as f:
                f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), exc))
        self.close()
        return False
