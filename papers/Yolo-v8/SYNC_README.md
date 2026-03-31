# 同步到云端服务器

## 🚀 快速开始

### Step 1: 配置服务器信息

编辑 `sync_to_cloud.sh`，修改以下配置：

```bash
# 服务器配置
SERVER="your-server-ip"          # 改为你的服务器 IP
USER="your-username"             # 改为你的用户名
REMOTE_DIR="/home/$USER/yolo-project"  # 改为远程目录
```

### Step 2: 配置 SSH 密钥（第一次需要）

```bash
# 在 Git Bash 中执行
./sync_to_cloud.sh setup-ssh
```

按照提示配置 SSH 密钥。

### Step 3: 同步代码

```bash
# 上传本地修改到服务器
./sync_to_cloud.sh push

# 从服务器下载修改
./sync_to_cloud.sh pull
```

---

## 📁 同步内容

### 会同步的文件

- ✅ Python 脚本（`*.py`）
- ✅ YAML 配置文件
- ✅ Shell 脚本（`*.sh`）
- ✅ 文档（`*.md`）
- ✅ 数据集（如果配置了）

### 不会同步的文件

- ❌ `.git/` - Git 目录
- ❌ `__pycache__/` - Python 缓存
- ❌ `*.pyc` - 编译的 Python 文件
- ❌ `runs/` - 训练输出
- ❌ `weights/` - 模型权重文件
- ❌ `*.log` - 日志文件

---

## 💡 推荐工作流

### 开发模式

```bash
# 1. 本地修改代码
# 编辑 train_fabric.py, add new feature...

# 2. 同步到服务器
./sync_to_cloud.sh push

# 3. SSH 登录并运行
ssh user@server
cd /home/user/yolo-project
python train_fabric.py --data fabric-defect.yaml
```

### 快速训练

```bash
# 一键同步并训练
./sync_to_cloud.sh push && ssh user@server "
    cd /home/user/yolo-project
    ./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0
"
```

---

## ⚙️ 高级配置

### 自定义排除文件

编辑 `sync_to_cloud.sh` 中的排除列表：

```bash
cat > "$EXCLUDE_FILE" << 'EOF'
.git/
__pycache__/
*.pyc
你的排除文件
EOF
```

### 使用 Git 模式

编辑 `sync_to_cloud.sh`:

```bash
SYNC_MODE="git"  # 改为 git 模式
```

---

## 🔧 故障排除

### 连接失败

```bash
# 测试 SSH 连接
ssh user@server

# 如果失败，重新配置 SSH
./sync_to_cloud.sh setup-ssh
```

### 权限错误

```bash
# 在服务器上执行
chmod +x /home/user/yolo-project/*.sh
```

### 文件不同步

```bash
# 强制同步（删除服务器端多余文件）
./sync_to_cloud.sh push  # rsync 模式会自动处理
```
