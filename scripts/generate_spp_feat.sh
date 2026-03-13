#!/bin/sh

PID0=""
PID1=""
TASK_PID=""
USE_SETSID=0
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU0="${GPU0:-4}"
GPU1="${GPU1:-5}"
CONFIG0="${CONFIG0:-configs/stpls3d.yaml}"
CONFIG1="${CONFIG1:-configs/stpls3d_1.yaml}"

cd "$REPO_ROOT" || exit 1

if command -v setsid >/dev/null 2>&1; then
  USE_SETSID=1
fi

start_task() {
  gpu_id="$1"
  config_path="$2"
  shift 2

  if [ "$USE_SETSID" -eq 1 ]; then
    setsid env \
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:$PYTHONPATH" \
      "$PYTHON_BIN" tools/generate_point_feat_utonia.py --config "$config_path" "$@" &
  else
    env \
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:$PYTHONPATH" \
      "$PYTHON_BIN" tools/generate_point_feat_utonia.py --config "$config_path" "$@" &
  fi
  TASK_PID=$!
}

stop_task() {
  pid="$1"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

  retry=0
  while [ "$retry" -lt 5 ] && kill -0 "$pid" 2>/dev/null; do
    sleep 1
    retry=$((retry + 1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo
  echo "Interrupt received. Stopping background tasks..."
  stop_task "$PID0"
  stop_task "$PID1"
  wait "$PID0" 2>/dev/null || true
  wait "$PID1" 2>/dev/null || true
  exit 130
}

trap cleanup INT TERM HUP

echo "Starting GPU ${GPU0} task..."
start_task "$GPU0" "$CONFIG0" "$@"
PID0="$TASK_PID"

echo "Starting GPU ${GPU1} task..."
start_task "$GPU1" "$CONFIG1" "$@"
PID1="$TASK_PID"

echo "Waiting for both tasks to complete..."
wait "$PID0"
STATUS0=$?
echo "GPU ${GPU0} task finished."

wait "$PID1"
STATUS1=$?
echo "GPU ${GPU1} task finished."

if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
  exit 1
fi

echo "All done. Point features saved to configured output directories."
