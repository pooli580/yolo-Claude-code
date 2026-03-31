#!/bin/bash
# 同步代码到云端服务器
# 用法：./sync_to_cloud.sh [mode]
#   mode: push (默认) 或 pull

set -e

# ==================== 配置区域 ====================
# 请根据实际情况修改以下配置

# 服务器配置
SERVER="10.20.26.171"            # 服务器 IP 或域名
USER="root"                      # 服务器用户名
REMOTE_DIR="/gemini/code/Yolo-v8"  # 远程项目目录

# Git 配置（如果使用 Git 方案）
GIT_REMOTE_NAME="server"         # 远程仓库名称

# 同步模式：git 或 rsync
SYNC_MODE="rsync"                # 可选：git, rsync

# ==================== 不要修改以下内容 ====================

MODE="${1:-push}"

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

check_connection() {
    echo ">>> 检查服务器连接..."
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$USER@$SERVER" "echo 连接成功" 2>/dev/null; then
        echo "✅ 服务器连接成功：$USER@$SERVER"
    else
        echo "❌ 无法连接到服务器：$USER@$SERVER"
        echo ""
        echo "请检查:"
        echo "  1. 服务器 IP 是否正确"
        echo "  2. SSH 是否配置了密钥认证"
        echo "  3. 防火墙是否开放 SSH 端口"
        echo ""
        echo "配置 SSH 密钥："
        echo "  ssh-keygen -t ed25519"
        echo "  ssh-copy-id $USER@$SERVER"
        exit 1
    fi
}

sync_with_rsync() {
    print_header "使用 rsync 同步代码 ($MODE)"

    # 获取脚本所在目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"

    if [ "$MODE" = "push" ]; then
        echo ">>> 上传本地修改到服务器..."

        # 排除不需要同步的文件
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

        rsync -avz --progress \
            --exclude-from="$EXCLUDE_FILE" \
            --delete \
            "$SCRIPT_DIR/" \
            "$USER@$SERVER:$REMOTE_DIR/"

        rm -f "$EXCLUDE_FILE"

        echo ""
        echo "✅ 上传完成!"
        echo ""
        echo "在服务器上运行:"
        echo "  cd $REMOTE_DIR"
        echo "  python check_environment.py"
        echo "  ./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0"

    elif [ "$MODE" = "pull" ]; then
        echo ">>> 从服务器下载修改..."

        rsync -avz --progress \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude 'runs/' \
            "$USER@$SERVER:$REMOTE_DIR/" \
            "$SCRIPT_DIR/"

        echo ""
        echo "✅ 下载完成!"
    fi
}

sync_with_git() {
    print_header "使用 Git 同步代码 ($MODE)"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"

    # 检查是否在 git 仓库中
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo "❌ 当前目录不是 Git 仓库"
        echo ""
        echo "初始化 Git 仓库:"
        echo "  git init"
        echo "  git add ."
        echo "  git commit -m 'Initial commit'"
        exit 1
    fi

    if [ "$MODE" = "push" ]; then
        # 检查远程是否存在
        if git remote | grep -q "^$GIT_REMOTE_NAME$"; then
            echo "✅ Git 远程仓库已配置：$GIT_REMOTE_NAME"
        else
            echo ">>> 配置 Git 远程仓库..."
            git remote add "$GIT_REMOTE_NAME" "ssh://$USER@$SERVER$REMOTE_DIR/.git"
        fi

        # 在服务器上初始化 bare 仓库（第一次需要）
        echo ">>> 在服务器上初始化 Git 仓库..."
        ssh "$USER@$SERVER" "
            if [ ! -d '$REMOTE_DIR/.git' ]; then
                mkdir -p '$REMOTE_DIR'
                cd '$REMOTE_DIR'
                git init --bare
            fi
        "

        # 推送代码
        echo ">>> 推送到服务器..."
        git push -f "$GIT_REMOTE_NAME" master

        # 在服务器上检出代码
        echo ">>> 在服务器上检出代码..."
        ssh "$USER@$SERVER" "
            cd '$REMOTE_DIR'
            git --work-tree='$REMOTE_DIR' --git-dir='$REMOTE_DIR/.git' checkout -f master
        "

        echo ""
        echo "✅ 推送完成!"

    elif [ "$MODE" = "pull" ]; then
        echo ">>> 从服务器拉取修改..."
        git pull "$GIT_REMOTE_NAME" master
        echo ""
        echo "✅ 拉取完成!"
    fi
}

setup_ssh() {
    print_header "SSH 配置向导"

    echo "Step 1: 检查 SSH 密钥"
    if [ -f ~/.ssh/id_ed25519.pub ]; then
        echo "✅ SSH 密钥已存在：~/.ssh/id_ed25519.pub"
    else
        echo ">>> 生成新的 SSH 密钥..."
        ssh-keygen -t ed25519 -C "your-email@example.com"
    fi

    echo ""
    echo "Step 2: 复制公钥到服务器"
    echo "运行以下命令:"
    echo "  ssh-copy-id $USER@$SERVER"
    echo ""
    echo "或者手动复制:"
    echo "  cat ~/.ssh/id_ed25519.pub | ssh $USER@$SERVER 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"

    echo ""
    echo "Step 3: 测试连接"
    echo "  ssh $USER@$SERVER 'echo 连接成功'"
}

# ==================== 主程序 ====================

print_header "云端同步工具"

echo "配置信息:"
echo "  服务器：$USER@$SERVER"
echo "  远程目录：$REMOTE_DIR"
echo "  同步模式：$SYNC_MODE"
echo "  操作：$MODE"
echo ""

case "${2:-}" in
    setup-ssh)
        setup_ssh
        ;;
    *)
        check_connection
        if [ "$SYNC_MODE" = "git" ]; then
            sync_with_git
        else
            sync_with_rsync
        fi
        ;;
esac
