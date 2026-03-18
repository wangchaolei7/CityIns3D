#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

GPU_IDS="${GPU_IDS:-3,4}"
CONFIG_PATH="${CONFIG_PATH:-configs/stpls3d.yaml}"
SCENE_LIST="${SCENE_LIST:-./open3dis/dataset_outdoor/stpls3d_val_ori.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

declare -a EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --scene-list)
      SCENE_LIST="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_WORKERS="${#GPU_ARRAY[@]}"
if [[ "$NUM_WORKERS" -le 0 ]]; then
  echo "No GPU ids provided."
  exit 1
fi

USE_SETSID=0
if command -v setsid >/dev/null 2>&1; then
  USE_SETSID=1
fi

declare -a PIDS=()

start_task() {
  local worker_id="$1"
  local gpu_id="$2"

  if [[ "$USE_SETSID" -eq 1 ]]; then
    setsid env \
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:${PYTHONPATH:-}" \
      "$PYTHON_BIN" tools/grounding_2d_stpls3d.py \
      --config "$CONFIG_PATH" \
      --scene-list "$SCENE_LIST" \
      --worker-id "$worker_id" \
      --num-workers "$NUM_WORKERS" \
      "${EXTRA_ARGS[@]}" &
  else
    env \
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:${PYTHONPATH:-}" \
      "$PYTHON_BIN" tools/grounding_2d_stpls3d.py \
      --config "$CONFIG_PATH" \
      --scene-list "$SCENE_LIST" \
      --worker-id "$worker_id" \
      --num-workers "$NUM_WORKERS" \
      "${EXTRA_ARGS[@]}" &
  fi
  PIDS+=("$!")
}

stop_task() {
  local pid="$1"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

  local retry=0
  while [[ "$retry" -lt 5 ]] && kill -0 "$pid" 2>/dev/null; do
    sleep 1
    retry=$((retry + 1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo
  echo "🛑 Interrupt received. Stopping background tasks..."
  for pid in "${PIDS[@]}"; do
    stop_task "$pid"
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit 130
}

trap cleanup INT TERM HUP

echo "📄 Scene list: $SCENE_LIST"
echo "⚙️  Config: $CONFIG_PATH"
echo "🖥️  GPUs: $GPU_IDS"
echo "🚀 Launching $NUM_WORKERS workers..."

for idx in "${!GPU_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$idx]}"
  echo "🚀 Starting worker $idx on GPU $gpu_id..."
  start_task "$idx" "$gpu_id"
done

echo "⏳ Waiting for all workers to complete..."
STATUS=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "$pid"; then
    echo "✅ Worker $idx finished."
  else
    echo "❌ Worker $idx failed."
    STATUS=1
  fi
done

exit "$STATUS"
