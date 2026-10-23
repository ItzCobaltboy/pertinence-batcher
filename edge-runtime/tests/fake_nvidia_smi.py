# Stands in for nvidia-smi in the tests: answers field probes, and with -lms writes
# samples to the -f file until it gets SIGINT/SIGTERM. Writes its pid to
# $FAKE_SMI_PIDFILE so a test can check it is gone.
import os
import signal
import sys
import time
from datetime import datetime

args = sys.argv[1:]
query = [a for a in args if a.startswith("--query-gpu=")][0].split("=", 1)[1].split(",")
if "-lms" not in args:
    print("[N/A]" if query[0] == "power.draw.instant" else "42.0")
    sys.exit(0)
pidfile = os.environ.get("FAKE_SMI_PIDFILE")
if pidfile:
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
interval = int(args[args.index("-lms") + 1]) / 1000.0
out = args[args.index("-f") + 1]
with open(out, "w") as f:
    while True:
        stamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        f.write(", ".join([stamp] + ["42.0"] * (len(query) - 1)) + "\n")
        f.flush()
        time.sleep(interval)
