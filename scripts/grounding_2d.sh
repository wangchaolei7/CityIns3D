#!/bin/bash

# dataset_cfg=${1:-'configs/replica.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=1 python3 tools/grounding_2d_replica.py --config $dataset_cfg

# dataset_cfg=${1:-'configs/scannet200.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python3 tools/grounding_2d.py --config $dataset_cfg

# kitti360
# dataset_cfg=${1:-'configs/kitti360.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python3 tools/grounding_2d_kitti360.py --config $dataset_cfg

# scannet200_nodepth
# dataset_cfg=${1:-'configs/scannet200_nodepth.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python3 tools/grounding_2d_scannet200_nodepth.py --config $dataset_cfg

# stpls3d
# dataset_cfg=${1:-'configs/stpls3d.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python3 tools/grounding_2d_stpls3d.py --config $dataset_cfg


# 启动 GPU 0 任务
echo "🚀 Starting GPU 0 task..."
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS="ignore" PYTHONPATH=./:$PYTHONPATH \
python3 tools/grounding_2d_stpls3d.py --config configs/stpls3d.yaml &

PID0=$!

# 启动 GPU 1 任务
echo "🚀 Starting GPU 1 task..."
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" PYTHONPATH=./:$PYTHONPATH \
python3 tools/grounding_2d_stpls3d.py --config configs/stpls3d_1.yaml &

PID1=$!

# 等待两个任务完成
echo "⏳ Waiting for both tasks to complete..."
wait $PID0
echo "✅ GPU 0 task finished."

wait $PID1
echo "✅ GPU 1 task finished."

echo "🎉 All done! Results saved in respective GPU directories."