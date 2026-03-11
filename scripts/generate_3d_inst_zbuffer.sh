#!/bin/bash
# Scannet200
# dataset_cfg=${1:-'configs/scannet200_zbuffer.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python3 tools/generate_3d_inst_zbuffer.py --config $dataset_cfg


# Scannetpp
# dataset_cfg=${1:-'configs/scannetpp_zbuffer.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python3 tools/generate_3d_inst_zbuffer.py --config $dataset_cfg


# Kitti360
# dataset_cfg=${1:-'configs/kitti360_zbuffer.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=0 python3 tools/generate_3d_inst_zbuffer.py --config $dataset_cfg

#----------------------------------------------------
# Stpls3d 默认使用z_buffer生成深度
# dataset_cfg=${1:-'configs/stpls3d.yaml'}
# export PYTHONWARNINGS="ignore"
# PYTHONPATH=./:$PYTHONPATH
# export PYTHONPATH
# CUDA_VISIBLE_DEVICES=1 python3 tools/generate_3d_inst_zbuffer.py --config $dataset_cfg


# 启动 GPU 0 任务
echo "🚀 Starting GPU 0 task..."
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS="ignore" PYTHONPATH=./:$PYTHONPATH \
python3 tools/generate_3d_inst_zbuffer.py --config configs/stpls3d.yaml &

PID0=$!

# 启动 GPU 1 任务
echo "🚀 Starting GPU 1 task..."
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" PYTHONPATH=./:$PYTHONPATH \
python3 tools/generate_3d_inst_zbuffer.py --config configs/stpls3d_1.yaml &

PID1=$!

# 等待两个任务完成
echo "⏳ Waiting for both tasks to complete..."
wait $PID0
echo "✅ GPU 0 task finished."

wait $PID1
echo "✅ GPU 1 task finished."

echo "🎉 All done! Results saved in respective GPU directories."
