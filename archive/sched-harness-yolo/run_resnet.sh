#!/usr/bin/env bash
# The whole ResNet DVFS study in one command, from sched-harness/ inside tmux:   sudo -E bash run_resnet.sh
# It prints the plan (runs, hours) and stops. Start for real with GO=1:          GO=1 sudo -E bash run_resnet.sh
# Steps (stops at the first failure): preflight -> calibrate (skipped when profile.json exists) -> two smoke runs -> plan.
# The plan is the alpha x SLO x load x seed grid of tier 8 plus fixed-clock baselines; it resumes, so after a crash run the same line again.
# Override the grid without editing files (space separated):
#   DVFS_ALPHAS="0 0.5 1" DVFS_SLOS="300 600" DVFS_LAMS="80 160" DVFS_SEEDS="0" GO=1 sudo -E bash run_resnet.sh
#   PYTHON=/path/to/venv/bin/python3 GO=1 sudo -E bash run_resnet.sh     if the venv is somewhere else
#   STEPS="plan" GO=1 sudo -E bash run_resnet.sh        run only some steps (default: preflight calibrate smoke plan)
# Defaults: 11 alphas (0 to 1 by 0.1), SLO 150 300 600 ms, loads 40 80 120 160 190 req/s, seeds 0 1 2, baselines 829 and 1377 MHz.
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
export SCHED_FAMILY=resnet
export DVFS_SLOS="${DVFS_SLOS:-150 300 600}"
STEPS="${STEPS:-preflight calibrate smoke plan}"

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo -E: GPU clock files and tegrastats rails need root" >&2
    exit 1
fi
exec > >(while IFS= read -r line; do printf '%s %s\n' "$(date '+%F %T')" "$line"; done | tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
trap 'sleep 1' EXIT

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

for s in $STEPS; do
    case "$s" in
        preflight) step preflight "$PYTHON" 00_preflight.py ;;
        calibrate)
            if [ -f "results/agx_xavier_jp511_resnet/profile.json" ]; then
                echo "--- calibrate: profile.json exists, skipped (delete it to redo) ---"
            else
                step calibrate "$PYTHON" 01_calibrate.py
            fi ;;
        smoke)
            step smoke_lqf "$PYTHON" 02_run.py --smoke
            step smoke_dvfs "$PYTHON" 02_run.py --smoke --policy dvfs_alpha --param alpha=0.5 --param slo_ms=300 ;;
        plan) step plan "$PYTHON" 03_plan.py ;;
        *) echo "unknown step $s"; exit 2 ;;
    esac
done
echo "=== run_resnet finished ==="
