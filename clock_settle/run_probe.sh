#!/bin/bash
# Builds and runs clock_probe on the board over a few clock pairs, busy and idle, then analyses the dumps.
# On the Jetson: first build as your normal user (nvcc -O2 -arch=sm_72 -o clock_probe clock_probe.cu -lpthread),
# then run as root with nothing else using the GPU:   sudo ./run_probe.sh            (about 15 minutes)
# Shorter test run:                                              sudo REPEATS=10 ./run_probe.sh
# The original min/max clocks are restored by clock_probe at the end of each call.
set -e
cd "$(dirname "$0")"
OUT=probe_results/$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
REPEATS=${REPEATS:-100}

# Build as your normal user BEFORE using sudo:   nvcc -O2 -arch=sm_72 -o clock_probe clock_probe.cu -lpthread
# (sudo resets PATH, so root often cannot find nvcc). If the binary is missing or older than the source, try to build it here.
if [ ! -x ./clock_probe ] || [ clock_probe.cu -nt ./clock_probe ]; then
    NVCC=${NVCC:-$(command -v nvcc || true)}
    [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
    if [ -z "$NVCC" ]; then
        echo "nvcc not found for this user (sudo has its own PATH)."
        echo "Build it as your normal user first:  nvcc -O2 -arch=sm_72 -o clock_probe clock_probe.cu -lpthread"
        echo "then run again with sudo."
        exit 1
    fi
    "$NVCC" -O2 -arch=sm_72 -o clock_probe clock_probe.cu -lpthread
fi

# busy: the kernel is running when the clock changes. idle200: the GPU was idle for 200 ms before the change.
for pair in 829:1237 421:1237 829:1033 1033:1198 1198:1237; do
    from=${pair%%:*}
    to=${pair##*:}
    ./clock_probe --from "$from" --to "$to" --mode busy --repeats "$REPEATS" --out "$OUT/busy_${from}_${to}"
    ./clock_probe --from "$from" --to "$to" --mode idle --idle-ms 200 --repeats "$REPEATS" --out "$OUT/idle200_${from}_${to}"
done

# long idle (3 s), one pair only: does a GPU that has been idle for a while pay the 88 ms?
./clock_probe --from 829 --to 1237 --mode idle --idle-ms 3000 --repeats 30 --out "$OUT/idle3000_829_1237"

echo "dumps are in $OUT"
if python3 -c "import numpy, pandas, matplotlib" 2>/dev/null; then
    for f in "$OUT"/*_events.csv; do
        python3 analyze_probe.py "${f%_events.csv}" | tee -a "$OUT/summary.txt"
    done
else
    echo "numpy/pandas/matplotlib missing here: copy $OUT to the PC and run  python3 analyze_probe.py <prefix>  there"
fi
