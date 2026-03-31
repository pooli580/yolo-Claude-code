# Git 远程同步配置指南

本指南说明如何配置本地和云端服务器之间的 Git 同步。

---

## 📋 前提条件

1. 云端服务器已安装 Git
2. 已配置 SSH 密钥认证

---

## 🔑 Step 1: 配置 SSH 密钥（如果没有）

### Windows (Git Bash)

```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your-email@example.com"

# 复制公钥到服务器
ssh-copy-id user@your-server

# 或手动复制
cat ~/.ssh/id_ed25519.pub | ssh user@your-server 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
```

### 测试连接

```bash
ssh user@your-server
```

---

## 📂 方案 A: 使用 rsync 同步（简单快捷）

### 配置 sync_to_cloud.sh

编辑 `sync_to_cloud.sh`，修改以下配置：

```bash
# 服务器配置
SERVER="your-server-ip"          # 改为你的服务器 IP
USER="your-username"             # 改为你的用户名
REMOTE_DIR="/home/$USER/yolo-project"  # 改为远程目录

# 同步模式
SYNC_MODE="rsync"
```

### 使用方法

```bash
# 上传本地修改到服务器
./sync_to_cloud.sh push

# 从服务器下载修改
./sync_to_cloud.sh pull
```

### 排除文件

以下文件会自动排除，不上传到服务器：

```
.git/
__pycache__/
*.pyc
runs/           # 训练输出
weights/        # 模型权重
*.log
```

---

## 📂 方案 B: 使用 Git 仓库（推荐用于正式项目）

### 1. 本地初始化 Git

```bash
cd D:\Claude-prj\papers\Yolo-v8

# 初始化 Git（如果还没有）
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: YOLOv8 fabric defect detection"
```

### 2. 配置服务器 Git 远程

编辑 `sync_to_cloud.sh`，修改配置：

```bash
SERVER="your-server-ip"
USER="your-username"
REMOTE_DIR="/home/$USER/yolo-project"
SYNC_MODE="git"
GIT_REMOTE_NAME="server"
```

### 3. 第一次同步

```bash
# 推送到服务器
./sync_to_cloud.sh push
```

### 4. 在服务器上配置（首次）

SSH 登录服务器后执行：

```bash
# 服务器端执行
cd /home/user/yolo-project

# 如果还没有 Git 仓库
git init

# 添加本地文件
git add .

# 创建 bare 仓库用于接收推送
cd /home/user/yolo-project.git
git init --bare
```

### 5. 日常使用

```bash
# 本地修改后推送到服务器
./sync_to_cloud.sh push

# 或在服务器上拉取
ssh user@your-server
cd /home/user/yolo-project
git pull origin master
```

---

## 🔄 方案 C: Syncthing 实时同步

### 安装 Syncthing

**本地（Windows）:**
1. 下载 https://syncthing.net/
2. 安装并启动

**服务器（Linux）:**
```bash
# Ubuntu/Debian
curl -s https://syncthing.net/keys/key.txt | sudo apt-key add -
echo "deb https://syncthing.net/syncthing/ stable main" | sudo tee /etc/apt/sources.list.d/syncthing.list
sudo apt-get update
sudo apt-get install syncthing

# 启动
syncthing
```

### 配置同步

1. 本地浏览器打开 `http://127.0.0.1:8384`
2. 服务器浏览器打开 `http://server-ip:8384`
3. 互相添加设备 ID
4. 添加同步文件夹 `D:\Claude-prj\papers\Yolo-v8` ↔ `/home/user/yolo-project`

---

## 💻 方案 D: VSCode Remote SSH（最方便）

### 安装扩展

在 VSCode 中安装 "Remote - SSH" 扩展

### 配置连接

1. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host..."
2. 输入 `user@your-server-ip`
3. 选择配置文件（通常是 `~/.ssh/config`）

### 直接编辑

连接后，可以直接打开服务器上的文件夹：
```
/home/user/yolo-project
```

所有修改实时保存在服务器上！

---

## 📝 推荐工作流

### 开发调试阶段（使用 rsync）

```bash
# 本地修改代码
# ...

# 快速同步到服务器
./sync_to_cloud.sh push

# SSH 登录并运行
ssh user@your-server
cd /home/user/yolo-project
python train_fabric.py --model yolov8s-strip.yaml --data fabric-defect.yaml
```

### 正式项目（使用 Git）

```bash
# 本地修改
git add .
git commit -m "改进 CASM 模块"

# 推送到服务器
./sync_to_cloud.sh push

# 服务器自动部署（可选）
# 配置 git hooks 自动拉取
```

---

## ⚙️ 自动化脚本

### 创建一个完整的训练脚本

在服务器上创建 `run_training.sh`:

```bash
#!/bin/bash
# 服务器端运行：./run_training.sh

set -e

cd /home/user/yolo-project

echo ">>> 检查环境..."
python check_environment.py

echo ">>> 开始训练..."
./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0

echo ">>> 验证模型..."
./val.sh runs/detect/fabric-exp-*/weights/best.pt fabric-defect.yaml
```

---

## 🚀 一键工作流

### 本地快速同步并训练

创建 `quick_train.sh`:

```bash
#!/bin/bash
# 用法：./quick_train.sh [epochs] [batch]

EPOCHS="${1:-100}"
BATCH="${2:-16}"

# 同步代码
./sync_to_cloud.sh push

# SSH 执行训练
ssh user@your-server "
    cd /home/user/yolo-project
    ./train.sh fabric-defect.yaml yolov8s-strip.yaml $EPOCHS $BATCH 0
"
```

---

## 📊 文件对比

| 操作 | rsync | Git | Syncthing | VSCode Remote |
|------|-------|-----|-----------|---------------|
| 初始配置 | 简单 | 中等 | 复杂 | 简单 |
| 同步速度 | 快 | 中等 | 快 | 实时 |
| 版本控制 | ❌ | ✅ | ❌ | ❌ |
| 冲突处理 | 覆盖 | 合并 | 双向 | N/A |
| 推荐场景 | 开发调试 | 正式项目 | 持续开发 | 远程编辑 |

---

**推荐**: 开发阶段用 rsync，正式发布用 Git
