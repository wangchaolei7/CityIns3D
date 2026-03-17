#!/bin/bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PIDS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

start_task() {
  local gpu_id="$1"
  local config_path="$2"

  echo "🚀 Starting GPU ${gpu_id} task..."
  if command -v setsid >/dev/null 2>&1; then
    setsid env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:${PYTHONPATH:-}" \
      python3 tools/generate_3d_inst_lifted_part.py --config "${config_path}" "${EXTRA_ARGS[@]}" &
  else
    env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONWARNINGS="ignore" \
      PYTHONPATH="./:${PYTHONPATH:-}" \
      python3 tools/generate_3d_inst_lifted_part.py --config "${config_path}" "${EXTRA_ARGS[@]}" &
  fi
  PIDS+=("$!")
}

cleanup() {
  trap - INT TERM HUP

  for pid in "${PIDS[@]}"; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  sleep 1

  for pid in "${PIDS[@]}"; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done

  wait "${PIDS[@]}" 2>/dev/null || true
}

trap 'echo "⛔ Interrupt received. Stopping tasks..."; cleanup; exit 130' INT TERM HUP

start_task 1 configs/stpls3d.yaml
# start_task 0 configs/stpls3d_1.yaml

echo "⏳ Waiting for tasks to complete..."

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}"
  rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    status="${rc}"
    break
  fi
done

if [[ "${status}" -ne 0 ]]; then
  cleanup
  exit "${status}"
fi

echo "🎉 All done! Lifted-part 3D proposals saved."
