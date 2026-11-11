#!/usr/bin/env bash
# Runs the whole chain on the Jetson, stopping at the first problem:
#   preflight -> calibrate -> smoke run (must print SMOKE PASS) -> Tier 1 plan.
# Start it once, inside tmux, and leave it:   tmux new -s harness   then   sudo bash run_all.sh
# Optional: PY=/path/to/python sudo -E bash run_all.sh   (default: the jetson-power venv of the invoking user)
# Logs go to logs/ next to this script. A finished step is not skipped on rerun except plan runs (03_plan.py resumes).
set -u -o pipefail
cd "$(dirname "$0")"

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo (tegrastats rails need root): sudo bash run_all.sh"
    exit 1
fi

USER_HOME=$(eval echo "~${SUDO_USER:-root}")
PY="${PY:-$USER_HOME/jetson-power/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "python not found at $PY (set PY=... to the venv python that has torch and ultralytics)"
    exit 1
fi

mkdir -p logs
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

step() {
    # step <name> <command...> : runs the command, logs to logs/<name>.log, stops the chain on a non-zero exit
    name="$1"; shift
    echo "[$(stamp)] START $name"
    "$@" 2>&1 | tee "logs/$name.log"
    code=${PIPESTATUS[0]}
    if [ "$code" -ne 0 ]; then
        echo "[$(stamp)] FAILED $name (exit $code). See logs/$name.log. Stopping."
        exit "$code"
    fi
    echo "[$(stamp)] DONE $name"
}

step 00_preflight "$PY" 00_preflight.py
step 01_calibrate "$PY" 01_calibrate.py
step 02_smoke "$PY" 02_run.py --smoke

if ! grep -q "^SMOKE PASS" logs/02_smoke.log; then
    echo "[$(stamp)] smoke run did not print SMOKE PASS, so the plan is NOT started. Read logs/02_smoke.log (the FAIL lines)."
    exit 2
fi

step 03_plan "$PY" 03_plan.py
echo "[$(stamp)] ALL DONE. Copy back results/ and logs/, then run 04_analyze.py on the laptop."
