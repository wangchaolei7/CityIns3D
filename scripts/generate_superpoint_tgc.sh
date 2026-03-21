#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

GPU_IDS="${GPU_IDS:-1,2}"
CONFIG_PATH="${CONFIG_PATH:-configs/stpls3d.yaml}"
SCENE_LIST="${SCENE_LIST:-}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SINGLE_SCENE=""

declare -a PIDS=()
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
    --original-root)
      ORIGINAL_ROOT="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --scene)
      SINGLE_SCENE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --worker-id|--num-workers)
      echo "Arguments $1 are managed by this script."
      exit 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_WORKERS="${#GPU_ARRAY[@]}"
if [[ "${NUM_WORKERS}" -le 0 ]]; then
  echo "No GPU ids provided."
  exit 1
fi

LAUNCH_WORKERS="${NUM_WORKERS}"
if [[ -n "${SINGLE_SCENE}" ]]; then
  LAUNCH_WORKERS=1
fi

USE_SETSID=0
if command -v setsid >/dev/null 2>&1; then
  USE_SETSID=1
fi

start_task() {
  local worker_id="$1"
  local gpu_id="$2"
  local -a cmd=(
    "${PYTHON_BIN}"
    tools/generate_superpoint_tgc.py
    --config "${CONFIG_PATH}"
  )

  if [[ -z "${SINGLE_SCENE}" ]]; then
    if [[ -n "${SCENE_LIST}" ]]; then
      cmd+=(--scene-list "${SCENE_LIST}")
    fi
    if [[ -n "${ORIGINAL_ROOT}" ]]; then
      cmd+=(--original-root "${ORIGINAL_ROOT}")
    fi
    cmd+=(
      --worker-id "${worker_id}"
      --num-workers "${NUM_WORKERS}"
    )
  elif [[ -n "${ORIGINAL_ROOT}" ]]; then
    cmd+=(--original-root "${ORIGINAL_ROOT}")
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  echo "Starting worker ${worker_id} on GPU ${gpu_id}..."
  if [[ "${USE_SETSID}" -eq 1 ]]; then
    setsid env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:${PYTHONPATH:-}" \
      "${cmd[@]}" &
  else
    env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:${PYTHONPATH:-}" \
      "${cmd[@]}" &
  fi

  PIDS+=("$!")
}

stop_task() {
  local pid="$1"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return
  fi

  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

  local retry=0
  while [[ "${retry}" -lt 5 ]] && kill -0 "${pid}" 2>/dev/null; do
    sleep 1
    retry=$((retry + 1))
  done

  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  trap - INT TERM HUP
  for pid in "${PIDS[@]}"; do
    stop_task "${pid}"
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

trap 'echo "Interrupt received. Stopping tasks..."; cleanup; exit 130' INT TERM HUP

echo "Config: ${CONFIG_PATH}"
echo "GPUs: ${GPU_IDS}"
if [[ -n "${SINGLE_SCENE}" ]]; then
  echo "Scene: ${SINGLE_SCENE}"
else
  if [[ -n "${SCENE_LIST}" ]]; then
    echo "Scene list: ${SCENE_LIST}"
  elif [[ -n "${ORIGINAL_ROOT}" ]]; then
    echo "Scene list: auto-discover from ${ORIGINAL_ROOT}"
  else
    echo "Scene list: ${SCENE_LIST:-<cfg.data.split_path or cfg.data.original_ply>}"
  fi
fi
echo "Launching ${LAUNCH_WORKERS} worker(s)..."

for idx in "${!GPU_ARRAY[@]}"; do
  if [[ "${idx}" -ge "${LAUNCH_WORKERS}" ]]; then
    break
  fi
  start_task "${idx}" "${GPU_ARRAY[$idx]}"
done

echo "Waiting for tasks to complete..."

status=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "Worker ${idx} finished."
  else
    echo "Worker ${idx} failed."
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  cleanup
  exit "${status}"
fi

echo "All done. TGC superpoint labels saved."
