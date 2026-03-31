#!/bin/bash
# 上传代码和数据集到云端服务器
# 用法：./upload_to_server.sh [server] [user]

set -e

# 配置（请根据实际情况修改）
SERVER="${1:-your-server-ip}"
USER="${2:-your-username}"
REMOTE_DIR="/path/to/project"

echo "============================================================"
echo "  上传 YOLOv8 项目到云端服务器"
echo "============================================================"
echo ""
echo "配置:"
echo "  服务器：$USER@$SERVER"
echo "  远程目录：$REMOTE_DIR"
echo ""

# 1. 上传代码
echo ">>> 上传代码..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'runs/' \
    papers/Yolo-v8/ $USER@$SERVER:$REMOTE_DIR/Yolo-v8/

echo ""
echo ">>> 上传数据集（如果有的话）..."
# 如果数据集较大，可以考虑单独上传
if [ -d "datasets/fabric-defect" ]; then
    rsync -avz --progress \
        datasets/fabric-defect/ $USER@$SERVER:$REMOTE_DIR/datasets/fabric-defect/
    echo "✅ 数据集上传完成"
else
    echo "⚠️  数据集目录不存在，跳过"
fi

echo ""
echo "============================================================"
echo "  上传完成!"
echo "============================================================"
echo ""
echo "SSH 登录命令:"
echo "  ssh $USER@$SERVER"
echo ""
echo "开始训练:"
echo "  cd $REMOTE_DIR/Yolo-v8"
echo "  python check_environment.py"
echo "  ./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0"
echo ""
