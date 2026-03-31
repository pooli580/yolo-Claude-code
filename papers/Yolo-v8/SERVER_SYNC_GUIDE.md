# 云端服务器同步指南

## 📋 服务器信息

- **服务器 IP**: `10.20.26.171`
- **用户名**: `root`
- **远程目录**: `/root/yolo-project`

---

## 🔑 Step 1: 配置 SSH 密钥（第一次需要）

### 在 Git Bash 中执行

```bash
# 1. 生成 SSH 密钥（如果没有）
ssh-keygen -t ed25519 -C "your-email@example.com"
# 按回车使用默认路径

# 2. 复制公钥到服务器
ssh-copy-id root@10.20.26.171
# 输入服务器密码

# 3. 测试连接
ssh root@10.20.26.171 "echo 连接成功"
```

### 如果 ssh-copy-id 不可用（Windows）

```bash
# 手动复制公钥
type $HOME/.ssh/id_ed25519.pub | ssh root@10.20.26.171 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

---

## 🚀 Step 2: 同步代码

### 方法 A: 使用快速同步脚本（推荐）

```bash
cd D:\Claude-prj\papers\Yolo-v8

# 一键同步
./quick_sync.sh
```

### 方法 B: 使用完整同步脚本

```bash
cd D:\Claude-prj\papers\Yolo-v8

# 上传到服务器
./sync_to_cloud.sh push

# 从服务器下载
./sync_to_cloud.sh pull
```

### 方法 C: 手动 rsync

```bash
rsync -avz --progress \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'runs/' \
    --exclude 'weights/' \
    D:/Claude-prj/papers/Yolo-v8/ \
    root@10.20.26.171:/root/yolo-project/
```

---

## 💻 Step 3: 服务器端设置

### SSH 登录服务器

```bash
ssh root@10.20.26.171
```

### 初始化环境

```bash
cd /root/yolo-project

# 运行初始化脚本
./setup_server.sh
```

### 开始训练

```bash
# 环境检查
python check_environment.py

# 开始训练
./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0

# 验证模型
./val.sh runs/detect/fabric-exp-*/weights/best.pt fabric-defect.yaml
```

---

## 📁 同步说明

### 会同步的文件 ✅

| 类型 | 示例 |
|------|------|
| Python 脚本 | `*.py` |
| 配置文件 | `*.yaml`, `*.yml` |
| Shell 脚本 | `*.sh` |
| 文档 | `*.md` |

### 不会同步的文件 ❌

| 类型 | 原因 |
|------|------|
| `.git/` | Git 目录 |
| `__pycache__/` | Python 缓存 |
| `*.pyc` | 编译文件 |
| `runs/` | 训练输出（很大） |
| `weights/` | 模型权重（很大） |
| `*.log` | 日志文件 |

---

## 🔄 日常工作流

### 开发调试

```bash
# 1. 本地修改代码
# 编辑 train_fabric.py, add new feature...

# 2. 同步到服务器
./quick_sync.sh

# 3. SSH 登录并运行
ssh root@10.20.26.171
cd /root/yolo-project
python train_fabric.py --data fabric-defect.yaml --epochs 100
```

### 快速训练

```bash
# 一键同步并启动训练
./quick_sync.sh && ssh root@10.20.26.1771 "
    cd /root/yolo-project
    ./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0
"
```

### 查看训练进度

```bash
# SSH 登录后查看
ssh root@10.20.26.171

# 查看最新训练
cd /root/yolo-project
ls -lh runs/detect/

# 查看训练日志
tail -f runs/detect/fabric-exp-*/results.csv
```

---

## 📊 TensorBoard 监控

### 在服务器上启动

```bash
ssh root@10.20.26.171
cd /root/yolo-project
tensorboard --logdir runs/detect --host 0.0.0.0 --port 6006
```

### 本地浏览器访问

```
http://10.20.26.171:6006
```

---

## 🔧 常见问题

### Q: SSH 连接失败

```bash
# 检查 SSH 密钥
ls -la ~/.ssh/id_ed25519.pub

# 重新配置
ssh-keygen -t ed25519
type ~/.ssh/id_ed25519.pub | ssh root@10.20.26.171 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### Q: rsync 命令不存在（Windows）

安装 Git for Windows，它会提供 rsync：
https://git-scm.com/download/win

或者使用 scp：
```bash
scp -r * root@10.20.26.171:/root/yolo-project/
```

### Q: 训练中断后继续

```bash
# 恢复训练
python train_fabric.py --resume runs/detect/fabric-exp-*/weights/last.pt
```

---

## 📝 文件清单

已创建的文件：

| 文件 | 用途 |
|------|------|
| `quick_sync.sh` | 快速同步脚本 ⭐ |
| `sync_to_cloud.sh` | 完整同步脚本 |
| `setup_server.sh` | 服务器初始化 |
| `train.sh` | 训练脚本 |
| `val.sh` | 验证脚本 |
| `check_environment.py` | 环境检查 |

---

**开始使用**: `./quick_sync.sh` 🚀
