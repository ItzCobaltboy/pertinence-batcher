#!/usr/bin/env bash
# The whole ResNet DVFS study in one command, from sched-harness/ inside tmux:   sudo -E bash run_resnet.sh
# It prints the plan (runs, hours) and stops. Start for real with GO=1:          GO=1 sudo -E bash run_resnet.sh
# Steps (stops at the first failure): preflight -> calibrate (skipped when profile.json exists) -> two smoke runs -> plan.
# The plan is the alpha x SLO x load x seed grid of tier 8 plus fixed-clock baselines; it resumes, so after a crash run the same line again.
# Override the grid without editing files (space separated):
#   DVFS_ALPHAS="0 0.5 1" DVFS_SLOS="300 600" DVFS_LAMS="80 160" DVFS_SEEDS="0" GO=1 sudo -E bash run_resnet.sh
#   PYTHON=/path/to/venv/bin/python3 GO=1 sudo -E bash run_resnet.sh     if the venv is somewhere else
#   CPU_MAX_KHZ=0 ...   turn the CPU cap off (default cap 1190400 kHz). CALIB=full ...   use the long GPU calibration instead of the table-based profile.
#   STEPS="plan" GO=1 sudo -E bash run_resnet.sh        run only some steps (default: preflight calibrate smoke plan)
# Defaults: 5 alphas (0 to 1 by 0.25) + fixed clocks 829 1033 1198 MHz, SLO 900 and 300 ms, loads 100 200 300 req/s, skewed mix light3 (60/30/10), seeds 0 1 2, expired jobs dropped; GPU clock ceiling 1237 MHz (the PSU limit).
# Log: runlog_resnet.txt (appended, timestamped).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
ROOT="$(dirname "$HERE")"
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then                 # same interpreter rule as dvfs_runtime/run_all.sh: sudo does not see the venv
    for cand in "$ROOT/jetson-power/.venv/bin/python3" "$HERE/.venv/bin/python3" "$ROOT/.venv/bin/python3"; do
        if [ -x "$cand" ]; then PYTHON="$cand"; break; fi
    done
    PYTHON="${PYTHON:-python3}"
fi
LOG="$HERE/runlog_resnet.txt"
export DVFS_SLOS="${DVFS_SLOS:-900 300}"
STEPS="${STEPS:-preflight calibrate smoke plan}"

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo -E: GPU clock files and tegrastats rails need root" >&2
    exit 1
fi
exec > >(while IFS= read -r line; do printf '%s %s\n' "$(date '+%F %T')" "$line"; done | tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
# tee does not fsync: a hard power-off would lose the last lines of the log. Flush the disk every 2 s so the last line is the real one.
( while true; do sync; sleep 2; done ) &
SYNC_PID=$!
TEGRA_PID=""
trap 'kill $SYNC_PID $TEGRA_PID 2>/dev/null; cpu_restore; sync; sleep 1' EXIT

step() {
    local name="$1"; shift
    echo "--- $name: $* ---"
    "$@"
    local code=$?
    if [ $code -ne 0 ]; then
        echo "=== FAILED (exit $code) at: $name. Fix it and run the same line again (finished plan runs are kept). ==="
        exit $code
    fi
}

echo "=== run_resnet: steps [$STEPS], SLOs [$DVFS_SLOS], python $PYTHON, log $LOG ==="
"$PYTHON" -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)" || { echo "this python has no torch: set PYTHON=/path/to/venv/bin/python3"; exit 1; }
# Cap the CPU clock for the whole study. The harness is GPU-bound (the CPU only runs the arrival process and the main loop), and CPU plus GPU
# load together is what browned out the board. CPU_MAX_KHZ=0 turns the cap off. The original limits are put back when the script exits.
CPU_SAVE="/tmp/cpu_max_saved.$$"
cpu_cap() {
    local want="${CPU_MAX_KHZ:-1190400}"
    [ "$want" = "0" ] && { echo "CPU cap off"; return; }
    : > "$CPU_SAVE"
    for d in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
        [ -w "$d/scaling_max_freq" ] || continue
        echo "$d $(cat "$d/scaling_max_freq")" >> "$CPU_SAVE"
        local best=0 f
        for f in $(cat "$d/scaling_available_frequencies"); do
            if [ "$f" -le "$want" ] && [ "$f" -gt "$best" ]; then best="$f"; fi
        done
        [ "$best" -gt 0 ] && echo "$best" > "$d/scaling_max_freq"
    done
    echo "CPU max clock capped: cpu0 now $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz (asked <= $want)"
}
cpu_restore() {
    [ -s "$CPU_SAVE" ] || return
    while read -r d v; do echo "$v" > "$d/scaling_max_freq"; done < "$CPU_SAVE"
    rm -f "$CPU_SAVE"
}

MODE="$(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
echo "power mode now: $MODE"
if [ "${ANY_MODE:-0}" != "1" ] && ! echo "$MODE" | grep -q "MAXN"; then
    echo "Not in MAXN. The table was measured in MAXN. Set it by hand:  sudo nvpmodel -m 0   then wait ~30 s and run this again."
    echo "(ANY_MODE=1 skips this check.)"
    exit 1
fi
echo "--- plan size ---"
"$PYTHON" 03_plan.py --dry-run || exit 1
if [ "${GO:-0}" != "1" ]; then
    echo "Dry run only. Nothing started. Add GO=1 to run it for real."
    exit 0
fi

cpu_cap
# Board power, temperatures and clocks once a second into a file that the 2 s sync flushes: after a crash its last line is the board's state.
stdbuf -oL tegrastats --interval 1000 >> "$HERE/tegrastats_session.log" 2>&1 &
TEGRA_PID=$!
echo "tegrastats session log: $HERE/tegrastats_session.log"
for s in $STEPS; do
    case "$s" in
        preflight) step preflight "$PYTHON" 00_preflight.py ;;
        calibrate)
            if [ -f "results/agx_xavier_jp511_resnet/profile.json" ]; then
                echo "--- calibrate: profile.json exists, skipped (delete it to redo) ---"
            else
                if [ "${CALIB:-table}" = "full" ]; then
                    step calibrate "$PYTHON" 01_calibrate.py          # the long GPU run that powered the board off
                else
                    step profile_from_table "$PYTHON" 01b_profile_from_table.py
                fi
            fi ;;
        smoke)
            step smoke_lqf "$PYTHON" 02_run.py --smoke
            step smoke_dvfs "$PYTHON" 02_run.py --smoke --policy dvfs_alpha --param alpha=0.5 --param slo_ms=300 ;;
        plan) step plan "$PYTHON" 03_plan.py ;;
        *) echo "unknown step $s"; exit 2 ;;
    esac
done
echo "=== run_resnet finished ==="
