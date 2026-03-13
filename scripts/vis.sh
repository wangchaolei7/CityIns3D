#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

SCENE_ID="${SCENE_ID:-05_points_GTv3_00}"
CONFIG_PATH="${CONFIG_PATH:-configs/stpls3d.yaml}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  SCENE_ID="$1"
  shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="viz/${SCENE_ID}"
fi

cd "$REPO_ROOT" || exit 1

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python3 visualization/visualize_stpls3d.py \
  --config "${CONFIG_PATH}" \
  --scene "${SCENE_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
