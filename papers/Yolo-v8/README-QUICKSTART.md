# YOLOv8-Strip 快速开始指南

## 项目说明

这是基于 **Strip R-CNN** 论文实现的 YOLOv8 改进版本，专门用于**高长宽比物体检测**（如遥感图像、织物缺陷）。

**核心创新**：使用 19×1 正交条带卷积 (H-Strip + V-Strip) 替代传统方形卷积。

---

## 一、环境准备

### 快速安装（推荐）

**Windows 用户**：双击运行 `setup_cpu_env.bat` 一键安装 CPU 环境

```bash
# 或手动安装
```

### 方案 A: CPU 版本（无独立显卡）

```bash
# 1. 安装 CPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 安装 Ultralytics
pip install ultralytics
```

### 方案 B: GPU 版本（有 NVIDIA 显卡）

```bash
# Windows + CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 方案 C: 使用项目虚拟环境

项目已预配置虚拟环境 (`.venv`)，激活后安装依赖：

```bash
# Windows PowerShell
cd papers\Yolo-v8
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

### 验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
```

### 完整测试

运行测试脚本验证环境：
```bash
python test_cpu_env.py
```

---

## 二、项目结构（精简后）

```
Yolo-v8/
├── __init__.py           # 模块入口，导出所有组件
├── strip_conv.py         # Strip 卷积模块
├── strip_net.py          # StripNet 骨干网络
├── strip_head.py         # Strip 检测头
├── strip_rcnn_head.py    # Strip R-CNN 头
├── yolov8_strip.py       # 完整模型定义
│
├── yolov8s-strip.yaml    # 模型配置（用于 YOLOv8 CLI）
├── fabric-defect.yaml    # 织物数据集配置
│
├── train_fabric.py       # 训练脚本
├── predict_fabric.py     # 预测脚本
└── README-QUICKSTART.md  # 本文档
```

---

## 三、如何使用

### 方式 1：使用 Python API（推荐）

```python
import torch
from yolov8_strip import yolov8s_strip

# 1. 创建模型
model = yolov8s_strip(num_classes=4)  # 4 类织物缺陷
model = model.cuda() if torch.cuda.is_available() else model

# 2. 加载权重（如果有）
# model.load_state_dict(torch.load('runs/detect/fabric-exp/weights/best.pt'))

# 3. 推理
model.eval()
img = torch.randn(1, 3, 640, 640).cuda()
with torch.no_grad():
    outputs = model(img)

# 4. 解析输出
print(outputs['cls'])   # 分类得分 [P3, P4, P5]
print(outputs['loc'])   # 边界框 [P3, P4, P5]
print(outputs['angle']) # 角度 [P3, P4, P5]
```

### 方式 2：使用 Ultralytics CLI

```bash
# 训练
yolo train model=yolov8s-strip.yaml data=fabric-defect.yaml epochs=100 batch=16

# 验证
yolo val model=runs/detect/fabric-exp/weights/best.pt data=fabric-defect.yaml

# 预测
yolo predict model=runs/detect/fabric-exp/weights/best.pt source=path/to/image.jpg
```

### 方式 3：使用训练脚本

```bash
# 基础训练
python train_fabric.py --data fabric-defect.yaml --epochs 100 --batch 16

# 指定模型和参数
python train_fabric.py \
    --model yolov8s-strip.yaml \
    --data fabric-defect.yaml \
    --epochs 200 \
    --batch 32 \
    --imgsz 1024 \
    --device 0

# 多 GPU 训练
python train_fabric.py --device 0,1 --batch 64

# 恢复训练
python train_fabric.py --resume runs/detect/fabric-exp/weights/last.pt
```

---

## 四、数据集配置

### 织物缺陷数据集 (fabric-defect.yaml)

```yaml
path: data/fabric
train: images/train
val: images/val

nc: 4  # 类别数
names:
  0: weaver_break      # 断经
  1: weft_stop         # 断纬
  2: hole              # 破洞
  3: oil_stain         # 油污
```

### 目录结构

```
data/fabric/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       ├── img001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── ...
    └── val/
        ├── img001.txt
        └── ...
```

**标注格式**（YOLO 格式）：
```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 五、核心模块说明

### Strip 卷积 (`strip_conv.py`)

```python
from __init__ import StripModule, StripC2f

# StripModule: 基本构建块
module = StripModule(
    in_channels=256,
    out_channels=256,
    strip_kernel_size=19,  # 19×1 条带
    shortcut=True,
)

# StripC2f: YOLOv8 C2f 的 Strip 版本
c2f = StripC2f(
    in_channels=512,
    out_channels=512,
    num_blocks=3,
    strip_kernel_size=19,
)
```

### 完整模型 (`yolov8_strip.py`)

```python
from yolov8_strip import YOLOv8Strip, yolov8s_strip

# 快速创建模型
model = yolov8s_strip(num_classes=80)

# 或自定义创建
from strip_net import StripNet
from yolov8_strip import YOLOv8Strip

backbone = StripNet(
    channels=[64, 128, 256, 512],
    depths=[2, 2, 3, 2],
    strip_kernel_size=19,
)

model = YOLOv8Strip(
    backbone=backbone,
    num_classes=15,
    strip_kernel_size=19,
)
```

---

## 六、模型变体

| 模型 | 函数 | 参数量 | 推荐用途 |
|------|------|--------|----------|
| Nano | `yolov8n_strip()` | ~4M | 移动端/实时检测 |
| Small | `yolov8s_strip()` | ~13M | 通用（推荐） |
| Medium | `yolov8m_strip()` | ~25M | 高精度检测 |
| Large | `yolov8l_strip()` | ~45M | 最高精度 |

---

## 七、训练技巧

### 1. 学习率调整

| 现象 | 调整 |
|------|------|
| Loss 不下降 | lr × 10 |
| Loss 震荡 | lr ÷ 10 |
| 收敛慢 | 使用 warmup + cosine decay |

### 2. Batch Size

| GPU 显存 | 推荐 |
|----------|------|
| 8GB | 8-16 |
| 16GB | 16-32 |
| 24GB | 32-64 |

### 3. 输入尺寸

| 场景 | 尺寸 |
|------|------|
| 通用检测 | 640×640 |
| 遥感图像 | 1024×1024 |
| 小目标 | 1280×1280 |

---

## 八、常见问题

### Q1: CUDA Out of Memory
```bash
# 减小 batch size
python train_fabric.py --batch 8

# 或启用混合精度
# 在代码中添加：amp=True
```

### Q2: ModuleNotFoundError
```bash
# 确保在项目根目录运行
cd papers/Yolo-v8
python train_fabric.py
```

### Q3: 如何使用自定义数据集
1. 修改 `fabric-defect.yaml` 中的路径和类别
2. 准备 YOLO 格式的标注
3. 运行训练

---

## 九、下一步

1. **开始训练**: `python train_fabric.py --data fabric-defect.yaml --epochs 100`
2. **查看训练曲线**: TensorBoard 查看 `runs/detect/fabric-exp/`
3. **推理测试**: `python predict_fabric.py --weights runs/detect/fabric-exp/weights/best.pt`

---

## 参考资料

- Strip R-CNN 论文：https://arxiv.org/abs/2501.03775
- Ultralytics 文档：https://docs.ultralytics.com
