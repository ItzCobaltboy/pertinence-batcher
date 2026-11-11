# Does sustained load at a high clock kill the board? Holds ResNet50 b=16 flat out at each "clock:seconds" stage, with a cooldown
# between stages, and writes the elapsed time plus the raw tegrastats line to stress.log with fsync every second.
# After a crash and reboot:   tail -8 stress.log     (the last line is the board's state in its last second)
# Run as root, MAXN set by hand:   python probe_stress.py                      (default stages 905:60 1198:60 1377:90)
#                                  python probe_stress.py 1377:30 1377:120     (your own stages)
#   --cpu N   also run N busy processes (float32 matmul loops) for the whole test, e.g. --cpu 8 for every core
#   --jc      also force CPU/GPU/EMC to max with jetson_clocks for the test (restored at the end)
#   worst case:  python probe_stress.py --cpu 8 --jc 1377:120
import multiprocessing
import os
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.join(HERE, "stress.log")
lock = threading.Lock()
stage = {"name": "idle", "t0": time.time(), "batches": 0}


def log(text):
    line = "%s %s" % (time.strftime("%H:%M:%S"), text)
    with lock:
        print(line, flush=True)
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def cpu_burn():
    import numpy as np
    a = np.random.rand(256, 256).astype(np.float32)
    b = np.random.rand(256, 256).astype(np.float32)
    while True:
        a = a.dot(b)
        a /= (np.abs(a).max() + 1e-9)


def watch_tegrastats():
    proc = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, universal_newlines=True)
    for line in proc.stdout:
        log("[%s +%3.0fs b=%d] %s" % (stage["name"], time.time() - stage["t0"], stage["batches"], line.strip()))


def main():
    if os.geteuid() != 0:
        sys.exit("run as root")
    args = sys.argv[1:]
    n_cpu = 0
    use_jc = "--jc" in args
    if "--cpu" in args:
        n_cpu = int(args[args.index("--cpu") + 1])
    stages = [a for i, a in enumerate(args) if ":" in a]
    stages = stages or ["905:60", "1198:60", "1377:90"]
    from common import dvfs
    import torch
    import torchvision
    log("=== stress start, stages %s, cpu burners %d, jetson_clocks %s ===" % (stages, n_cpu, use_jc))
    clock = dvfs.Clock(os.path.join(HERE, "stress_clock_backup.json"))
    threading.Thread(target=watch_tegrastats, daemon=True).start()
    burners = []
    jc_file = os.path.join(HERE, "stress_jetson_clocks.conf")
    try:
        if use_jc:
            subprocess.check_call(["jetson_clocks", "--store", jc_file])
            subprocess.check_call(["jetson_clocks"])
            log("jetson_clocks on (CPU, GPU, EMC at max)")
        for _ in range(n_cpu):
            p = multiprocessing.Process(target=cpu_burn)
            p.daemon = True
            p.start()
            burners.append(p)
        if burners:
            log("%d CPU burners running" % len(burners))
            time.sleep(5)
        net = torchvision.models.resnet50(weights=None).half().eval().cuda()
        x = torch.rand(16, 3, 224, 224, device="cuda").half()
        for item in stages:
            mhz, seconds = item.split(":")
            clock.set(int(mhz))
            stage.update(name="%sMHz" % mhz, t0=time.time(), batches=0)
            log("STAGE %s MHz for %s s" % (mhz, seconds))
            t_end = time.time() + float(seconds)
            while time.time() < t_end:
                with torch.no_grad():
                    net(x)
                torch.cuda.synchronize()
                stage["batches"] += 1
            log("STAGE %s MHz passed, %d batches" % (mhz, stage["batches"]))
            stage.update(name="cooldown", t0=time.time())
            time.sleep(20)
        log("ALL STAGES PASSED")
    finally:
        for p in burners:
            p.terminate()
        if use_jc:
            subprocess.call(["jetson_clocks", "--restore", jc_file])
            log("jetson_clocks restored")
        clock.restore()
        log("=== stress end, clocks restored ===")


main()
