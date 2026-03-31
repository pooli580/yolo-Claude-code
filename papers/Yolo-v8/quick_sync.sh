#!/bin/bash
# 快速同步到云端服务器
# 用法：./quick_sync.sh

set -e

SERVER="10.20.26.171"
USER="root"
REMOTE_DIR="/root/yolo-project"

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

print_header "同步代码到云端服务器"

echo "服务器：$USER@$SERVER"
echo "远程目录：$REMOTE_DIR"
echo ""

# 检查 SSH 连接
echo ">>> 检查 SSH 连接..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$USER@$SERVER" "echo 连接成功" 2>/dev/null; then
    echo "✅ SSH 连接正常"
else
    echo "❌ SSH 连接失败"
    echo ""
    echo "请先配置 SSH 密钥："
    echo "  1. 生成密钥：ssh-keygen -t ed25519"
    echo "  2. 复制公钥：ssh-copy-id root@10.20.26.171"
    echo "  3. 或手动复制："
    echo "     type \$HOME/.ssh/id_ed25519.pub | ssh root@10.20.26.171 \"mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys\""
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建排除文件
EXCLUDE_FILE=$(mktemp)
cat > "$EXCLUDE_FILE" << 'EOF'
.git/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.vscode/
.idea/
*.log
runs/
weights/
*.zip
*.tar.gz
EOF

echo ""
echo ">>> 开始同步..."
echo ""

# rsync 同步
rsync -avz --progress \
    --exclude-from="$EXCLUDE_FILE" \
    --delete \
    "$SCRIPT_DIR/" \
    "$USER@$SERVER:$REMOTE_DIR/"

rm -f "$EXCLUDE_FILE"

echo ""
print_header "同步完成!"

echo "在服务器上执行:"
echo "  ssh root@10.20.26.171"
echo "  cd $REMOTE_DIR"
echo "  ./setup_server.sh  # 第一次需要"
echo "  ./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0"
echo ""
