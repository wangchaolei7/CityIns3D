#!/bin/bash
# for 室外数据集

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:${PYTHONPATH:-}
NUM_WORKERS="${NUM_WORKERS:-0}"
PROTOCOL="${PROTOCOL:-cityins3d}"

export PYTHONPATH
#kitti360
# CUDA_VISIBLE_DEVICES=1 python3 open3dis/evaluation/eval_class_agnostic_kitti360.py --config configs/kitti360_zbuffer.yaml --type 2D
# CUDA_VISIBLE_DEVICES=1 python3 open3dis/evaluation/eval_class_agnostic_kitti360.py --config configs/kitti360.yaml --type 2D

# stpls3d
CUDA_VISIBLE_DEVICES=1 python3 open3dis/evaluation/eval_class_agnostic_stpls3d.py \
  --config configs/stpls3d.yaml \
  --type 2D \
  --protocol "${PROTOCOL}" \
  --num-workers "${NUM_WORKERS}" \
  "$@"


#laion2b_s39b_b160k
