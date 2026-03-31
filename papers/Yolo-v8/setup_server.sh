#!/bin/bash
# 服务器端初始化脚本
# 用法：在服务器上运行 ./setup_server.sh

set -e

echo "============================================================"
echo "  YOLOv8 服务器环境初始化"
echo "============================================================"
echo ""

# 检测项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "当前目录：$SCRIPT_DIR"
echo ""

# Step 1: 检查 Python
echo ">>> 检查 Python..."
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ 未找到 Python"
    exit 1
fi

$PYTHON_CMD --version
echo "✅ Python: $PYTHON_CMD"

# Step 2: 检查依赖
echo ""
echo ">>> 检查 Python 依赖..."
$PYTHON_CMD -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "⚠️  PyTorch 未安装"
$PYTHON_CMD -c "import ultralytics; print(f'✅ Ultralytics: {ultralytics.__version__}')" || echo "⚠️  Ultralytics 未安装"

# Step 3: 安装依赖（如果需要）
echo ""
read -p "是否安装/更新依赖？(y/n): " install_deps
if [ "$install_deps" = "y" ]; then
    echo ">>> 安装依赖..."
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install torch torchvision ultralytics opencv-python pillow matplotlib
fi

# Step 4: 检查 CUDA
echo ""
echo ">>> 检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ GPU 可用"
else
    echo "⚠️  未检测到 GPU，将使用 CPU 训练"
fi

# Step 5: 设置执行权限
echo ""
echo ">>> 设置脚本执行权限..."
chmod +x train.sh val.sh sync_to_cloud.sh 2>/dev/null || true
echo "✅ 脚本权限已设置"

# Step 6: 创建训练输出目录
echo ""
echo ">>> 创建目录结构..."
mkdir -p runs/detect 2>/dev/null || true
mkdir -p weights 2>/dev/null || true
echo "✅ 目录已创建"

# Step 7: 验证数据集配置
echo ""
echo ">>> 检查数据集配置..."
if [ -f "fabric-defect.yaml" ]; then
    echo "✅ 数据集配置：fabric-defect.yaml"
    $PYTHON_CMD -c "
import yaml
with open('fabric-defect.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'   类别数：{cfg.get(\"nc\", \"未知\")}')
print(f'   类别：{list(cfg.get(\"names\", {}).values())}')
"
else
    echo "⚠️  数据集配置不存在：fabric-defect.yaml"
fi

# Step 8: 显示使用指南
echo ""
echo "============================================================"
echo "  服务器设置完成!"
echo "============================================================"
echo ""
echo "快速开始:"
echo ""
echo "1. 环境检查:"
echo "   python check_environment.py"
echo ""
echo "2. 开始训练:"
echo "   ./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0"
echo ""
echo "3. 验证模型:"
echo "   ./val.sh runs/detect/fabric-exp-*/weights/best.pt fabric-defect.yaml"
echo ""
echo "4. 推理预测:"
echo "   python predict_fabric.py --weights best.pt --source image.jpg"
echo ""
echo "5. TensorBoard 监控:"
echo "   tensorboard --logdir runs/detect --host 0.0.0.0 --port 6006"
echo ""
echo "============================================================"
