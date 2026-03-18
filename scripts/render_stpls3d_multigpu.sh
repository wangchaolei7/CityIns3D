#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wangcl/CityIns3D"
PY_SCRIPT="${ROOT_DIR}/segmenter2d/Render/STPLS3D/snap_open3dis.py"
GPU_IDS="${GPU_IDS:-0,1}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
NUM_WORKERS="${#GPU_LIST[@]}"

if [[ "${NUM_WORKERS}" -le 0 ]]; then
  echo "No GPUs specified. Set GPU_IDS, e.g. GPU_IDS=0,1"
  exit 1
fi

pids=()

cleanup() {
  trap - INT TERM HUP
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}

trap cleanup INT TERM HUP

for worker_id in "${!GPU_LIST[@]}"; do
  gpu_id="${GPU_LIST[$worker_id]}"
  echo "🚀 Starting render worker ${worker_id}/${NUM_WORKERS} on GPU ${gpu_id}..."
  (
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    exec python "${PY_SCRIPT}" \
      --worker-id "${worker_id}" \
      --num-workers "${NUM_WORKERS}" \
      "$@"
  ) &
  pids+=("$!")
done

echo "⏳ Waiting for render workers to complete..."
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "❌ Rendering failed."
  exit "${status}"
fi

echo "✅ Rendering completed."
