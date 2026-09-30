#!/usr/bin/env bash
# One-shot launcher for the YOLOv8 batch sweep. Run it inside tmux:
#
#   tmux new -s batchsweep
#   bash yolo-analysis/batch_sweep/run_sweep.sh              # full default sweep
#   SESSION=main bash run_sweep.sh --runs 3                  # extra args go to `sweep.py measure`
#   env knobs: SESSION (default main), PYTHON (default python), DATASET_ROOT, CUDA_VISIBLE_DEVICES
#
# Steps: GPU/disk snapshot -> preflight -> measure (resumable) -> aggregate.
# Each step's full output is appended (&>>) to its own log file under
# yolo-analysis/results/batch_sweep/logs/<session>/, so nothing lives only in
# tmux scrollback. Rerunning this script resumes where the last run stopped.
# Watch progress from another pane with:  tail -f <log dir>/measure.log

set -u

SESSION="${SESSION:-main}"
PYTHON="${PYTHON:-python}"
DATA_ARGS=()   # DATASET_ROOT=/path overrides yolo-analysis/dataset for both preflight and measure
if [ -n "${DATASET_ROOT:-}" ]; then DATA_ARGS=(--dataset-root "$DATASET_ROOT"); fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$(cd "$HERE/.." && pwd)/results/batch_sweep"
OUT="${BATCH_SWEEP_OUT_DIR:-$OUT}"
LOG_DIR="$OUT/logs/$SESSION"
mkdir -p "$LOG_DIR"
cd "$HERE"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo "===== $(stamp) $* =====" &>> "$LOG_DIR/launcher.log"; echo "[$(stamp)] $*"; }

banner "launch: session=$SESSION args=$* host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# 1. Snapshot of the shared machine before anything runs.
{
    echo "===== $(stamp) environment snapshot ====="
    df -h "$REPO"
    nvidia-smi || echo "nvidia-smi not available"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
} &>> "$LOG_DIR/environment.log"

# 2. Preflight: versions, CUDA, torch_tensorrt, data folders, weights, disk.
banner "preflight -> $LOG_DIR/preflight_stdout.log"
if ! "$PYTHON" sweep.py preflight --session "$SESSION" "${DATA_ARGS[@]}" &>> "$LOG_DIR/preflight_stdout.log"; then
    banner "preflight FAILED, not starting the sweep (see preflight_stdout.log)"
    exit 1
fi

# 3. The sweep itself. Resumable: finished runs are skipped on a rerun.
banner "measure -> $LOG_DIR/measure.log"
"$PYTHON" sweep.py measure --session "$SESSION" "${DATA_ARGS[@]}" "$@" &>> "$LOG_DIR/measure.log"
MEASURE_STATUS=$?
banner "measure exited with $MEASURE_STATUS"

# 4. Aggregate again (measure already does it; this also runs if measure died).
banner "aggregate -> $LOG_DIR/aggregate.log"
"$PYTHON" sweep.py aggregate --session "$SESSION" &>> "$LOG_DIR/aggregate.log"

{
    echo "===== $(stamp) environment snapshot (end) ====="
    nvidia-smi || echo "nvidia-smi not available"
} &>> "$LOG_DIR/environment.log"

banner "done (measure status $MEASURE_STATUS). Sanity flags: $OUT/summary/$SESSION/sanity_flags.txt"
exit "$MEASURE_STATUS"
