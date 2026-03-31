#!/bin/bash
# YOLOv8 验证脚本
# 用法：./val.sh [权重文件] [数据集配置]

set -e

WEIGHTS="${1:-runs/detect/fabric-exp-001/weights/best.pt}"
DATA="${2:-fabric-defect.yaml}"
BATCH="${3:-16}"
DEVICE="${4:-0}"

echo "============================================================"
echo "  YOLOv8 模型验证"
echo "============================================================"
echo ""
echo "配置:"
echo "  权重：$WEIGHTS"
echo "  数据集：$DATA"
echo "  Batch: $BATCH"
echo "  Device: $DEVICE"
echo ""

# 检查文件
[ -f "$WEIGHTS" ] && echo "✅ 权重文件：$WEIGHTS" || { echo "❌ 权重文件不存在：$WEIGHTS"; exit 1; }
[ -f "$DATA" ] && echo "✅ 数据集配置：$DATA" || { echo "❌ 数据集配置不存在：$DATA"; exit 1; }

echo ""
echo ">>> 开始验证..."
python val_fabric.py val \
    --weights "$WEIGHTS" \
    --data "$DATA" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --verbose

echo ""
echo "============================================================"
echo "  验证完成!"
echo "============================================================"
