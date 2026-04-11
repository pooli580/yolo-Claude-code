#!/bin/bash
# YOLOv8-Strip 服务器训练脚本
# 用法：./server_train.sh

set -e

echo "=============================================="
echo "YOLOv8-Strip 服务器训练脚本"
echo "=============================================="

# 1. 环境检查
echo -e "\n[1/5] 检查环境..."
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CUDA 版本：$(nvcc --version | grep release)"

# 2. 激活虚拟环境
echo -e "\n[2/5] 激活虚拟环境..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "虚拟环境已激活"
else
    echo "创建虚拟环境..."
    python -m venv .venv
    source .venv/bin/activate
    echo "安装依赖..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics
fi

# 3. 检查数据集
echo -e "\n[3/5] 检查数据集..."
if [ -d "datasets/fabric-defect/images/train" ]; then
    echo "✓ 数据集已就绪"
    DATASET="fabric-defect.yaml"
else
    echo "⚠ 数据集不存在，使用 COCO8 测试..."
    DATASET="ultralytics/datasets/coco8.yaml"
fi

# 4. 启动训练
echo -e "\n[4/5] 启动训练..."
echo "模型：yolov8s-strip.yaml"
echo "数据集：$DATASET"
echo "设备：双卡 (0,1)"

# 训练参数
MODEL="yolov8s-strip.yaml"
EPOCHS=200
BATCH=64
IMGSZ=1024
DEVICE="0,1"
WORKERS=8
NAME="fabric-2gpu-exp"

# 使用 nohup 后台运行
nohup python train_fabric.py \
    --model $MODEL \
    --data $DATASET \
    --epochs $EPOCHS \
    --batch $BATCH \
    --imgsz $IMGSZ \
    --device $DEVICE \
    --workers $WORKERS \
    --name $NAME \
    > train_${NAME}.log 2>&1 &

PID=$!
echo "训练已启动 (PID: $PID)"

# 5. 显示监控命令
echo -e "\n[5/5] 监控命令:"
echo "  查看日志：tail -f train_${NAME}.log"
echo "  查看 GPU：watch -n 1 nvidia-smi"
echo "  TensorBoard：tensorboard --logdir runs/detect --port 6006"
echo ""
echo "=============================================="
echo "训练已启动！实验保存于：runs/detect/$NAME/"
echo "=============================================="
