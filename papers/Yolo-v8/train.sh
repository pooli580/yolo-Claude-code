#!/bin/bash
# YOLOv8 云端训练一键脚本
# 用法：./train.sh [配置文件] [模型] [epochs] [batch]

set -e

# 默认配置
DATA_CFG="${1:-fabric-defect.yaml}"
MODEL="${2:-yolov8s-strip.yaml}"
EPOCHS="${3:-100}"
BATCH="${4:-16}"
DEVICE="${5:-0}"
NAME="${6:-fabric-exp-$(date +%Y%m%d-%H%M%S)}"

echo "============================================================"
echo "  YOLOv8 织物缺陷检测 - 训练脚本"
echo "============================================================"
echo ""
echo "配置:"
echo "  数据集：$DATA_CFG"
echo "  模型：$MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  Device: $DEVICE"
echo "  实验名：$NAME"
echo ""

# 1. 环境检查
echo ">>> 检查环境..."
python -c "from ultralytics import YOLO; print('✅ YOLO 已安装')" || exit 1
python -c "import torch; print('✅ CUDA 可用' if torch.cuda.is_available() else '⚠️  CPU 模式')"

# 2. 检查文件
echo ""
echo ">>> 检查文件..."
[ -f "$DATA_CFG" ] && echo "✅ 数据集配置：$DATA_CFG" || { echo "❌ 数据集配置不存在：$DATA_CFG"; exit 1; }
[ -f "$MODEL" ] && echo "✅ 模型配置：$MODEL" || echo "⚠️  模型配置不存在，使用默认模型"

# 3. 开始训练
echo ""
echo ">>> 开始训练..."
python train_fabric.py \
    --model "$MODEL" \
    --data "$DATA_CFG" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --name "$NAME" \
    --verbose

echo ""
echo "============================================================"
echo "  训练完成!"
echo "============================================================"
echo ""
echo "结果路径：runs/detect/$NAME/"
echo "  最佳模型：runs/detect/$NAME/weights/best.pt"
echo "  训练曲线：runs/detect/$NAME/results.png"
echo ""
echo "验证命令:"
echo "  python val_fabric.py --weights runs/detect/$NAME/weights/best.pt --data $DATA_CFG"
echo ""
echo "推理命令:"
echo "  python predict_fabric.py --weights runs/detect/$NAME/weights/best.pt --source path/to/image.jpg"
echo ""
