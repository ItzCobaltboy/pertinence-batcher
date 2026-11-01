#!/usr/bin/env bash
# One command for the whole DVFS chain, run from this folder inside tmux:   sudo bash run_all.sh
# Steps (stops at the first failure): inspect -> profile_switch -> profile_models. Everything printed goes to runlog.txt
# (appended, with timestamps) and to the terminal. The python modules restore the GPU clocks themselves on every exit path;
# the clock state is printed before and after as a visible check.
#   STEPS="switch models" sudo -E bash run_all.sh     run only some steps (default: inspect switch models)
#   RESUME=results/<tag>/models_<stamp> sudo -E bash run_all.sh     continue an interrupted profile_models folder
#   DVFS_FAMILIES="resnet yolo" STEPS=models ...        one profile_models run per family, in that order (needs sudo -E or sudo env ...)
#   PYTHON=/path/to/python                            override the interpreter (default: ../.venv/bin/python, else python3)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"                 # python -m dvfs_runtime.* must run from the repo root (inspect.py would shadow the stdlib in here)
LOG="$HERE/runlog.txt"
STEPS="${STEPS:-inspect switch models}"
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    VENV_PY="$ROOT/jetson-power/.venv/bin/python3"
    if [ -x "$VENV_PY" ]; then PYTHON="$VENV_PY"; else PYTHON="python3"; fi
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo: writing the GPU clock files and reading tegrastats rails need root" >&2
    exit 1
fi

exec > >(while IFS= read -r line; do printf '%s %s\n' "$(date '+%F %T')" "$line"; done | tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
trap 'sleep 1' EXIT                       # let the log writer above flush its last lines
cd "$ROOT" || exit 1

echo "=== dvfs run_all: steps [$STEPS], python $PYTHON, log $LOG ==="
trap 'echo "=== interrupted: the python step restores the clocks; state below ==="; "$PYTHON" -m dvfs_runtime.gpu --show; exit 130' INT TERM

step() {
    local name="$1"
    shift
    echo "--- $name: $* ---"
    "$@"
    local code=$?
    if [ $code -ne 0 ]; then
        echo "=== FAILED (exit $code) at: $name ==="
        "$PYTHON" -m dvfs_runtime.gpu --show
        if [ "$name" = "profile_models" ]; then
            echo "To continue: RESUME=<the models_<stamp> folder above> STEPS=models sudo -E bash run_all.sh"
        fi
        exit $code
    fi
}

echo "--- GPU clock state before ---"
"$PYTHON" -m dvfs_runtime.gpu --show
step "plan (dry runs)" bash -c "'$PYTHON' -m dvfs_runtime.profile_switch --dry-run && '$PYTHON' -m dvfs_runtime.profile_models --dry-run"

for s in $STEPS; do
    case "$s" in
        inspect) step inspect "$PYTHON" -m dvfs_runtime.board_info ;;
        switch)  step profile_switch "$PYTHON" -m dvfs_runtime.profile_switch ;;
        models)
            if [ -n "${RESUME:-}" ]; then
                step profile_models "$PYTHON" -m dvfs_runtime.profile_models --resume "$RESUME"
            elif [ -n "${DVFS_FAMILIES:-}" ]; then
                for fam in $DVFS_FAMILIES; do      # e.g. DVFS_FAMILIES="resnet yolo": one models run per family, in that order
                    step "profile_models ($fam)" env DVFS_FAMILY="$fam" "$PYTHON" -m dvfs_runtime.profile_models
                done
            else
                step profile_models "$PYTHON" -m dvfs_runtime.profile_models
            fi ;;
        *) echo "unknown step '$s' (inspect | switch | models)"; exit 2 ;;
    esac
done

echo "--- GPU clock state after (min/max/governor must equal the 'before' block) ---"
"$PYTHON" -m dvfs_runtime.gpu --show
echo "=== dvfs run_all: done ==="
