# Finds what kills the board: walks the GPU clock up one step at a time under load and logs every step to probe.log
# with fsync, so after a crash the LAST LINE of probe.log says exactly where it died.
# Run (MAXN set by hand first):   sudo -E /home/agx2/cs24bt033/jetson-power/.venv/bin/python3 probe_clock.py
# Then, after a crash and reboot:  tail -5 probe.log
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.join(HERE, "probe.log")


def log(text):
    line = "%s %s" % (time.strftime("%H:%M:%S"), text)
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def hot():
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone1/temp") as fh:
            return int(fh.read()) / 1000.0
    except OSError:
        return -1.0


def burn(net, x, seconds, label):
    import torch
    t_end = time.time() + seconds
    n = 0
    last = time.time()
    while time.time() < t_end:
        with torch.no_grad():
            net(x)
        torch.cuda.synchronize()
        n += 1
        if time.time() - last > 2.0:
            log("  %s: %d batches so far, %.1f C" % (label, n, hot()))
            last = time.time()
    log("  %s: done, %d batches" % (label, n))


def main():
    from common import dvfs
    import torch
    import torchvision
    log("=== probe start ===")
    clock = dvfs.Clock(os.path.join(HERE, "probe_clock_backup.json"))
    log("available MHz %s, cap %d, original %s" % (clock.available_mhz, clock.cap_mhz, clock.original))
    try:
        log("S1 load resnet152 FP16 on the GPU (clock not pinned)")
        net = torchvision.models.resnet152(weights=None).half().eval().cuda()
        x = torch.rand(16, 3, 224, 224, device="cuda").half()
        log("S2 b=16 for 6 s, unpinned")
        burn(net, x, 6, "unpinned")
        for f in [829, 1198, 1377]:
            log("S3 pin %d MHz" % f)
            ms = clock.set(f)
            log("  pinned %d MHz in %.0f ms, cur_freq now %s" % (f, ms, dvfs._read("cur_freq")))
            time.sleep(2)
            log("S4 b=16 for 8 s at %d MHz" % f)
            burn(net, x, 8, "%d MHz" % f)
        log("S5 all clocks passed")
    finally:
        log("restoring clocks")
        clock.restore()
        log("=== probe end ===")


main()
