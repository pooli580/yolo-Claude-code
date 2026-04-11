# YOLOv8 CPU -only 环境配置指南

> 适用于**没有独立显卡**的电脑（仅使用 CPU 训练/推理）

---

## 快速开始

### 1. 使用现有虚拟环境

本项目已预配置虚拟环境，位于 `.venv/` 目录。

```bash
# Windows (PowerShell)
cd papers\Yolo-v8
.venv\Scripts\Activate.ps1

# Windows (CMD)
cd papers\Yolo-v8
.venv\Scripts\activate.bat

# Git Bash
cd papers/Yolo-v8
source .venv/Scripts/activate
```

### 2. 安装 CPU 版 PyTorch

```bash
# 激活虚拟环境后
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装 Ultralytics (YOLOv8)

```bash
pip install ultralytics
```

---

## 完整安装步骤

### 方案 A: 从零开始（推荐）

```bash
# 1. 创建新的虚拟环境
python -m venv yolo-cpu-env

# 2. 激活虚拟环境
# Windows PowerShell
yolo-cpu-env\Scripts\Activate.ps1
# Windows CMD
yolo-cpu-env\Scripts\activate.bat
# Git Bash
source yolo-cpu-env/Scripts/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装 CPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. 安装 Ultralytics
pip install ultralytics

# 6. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CPU only: {not torch.cuda.is_available()}')"
```

### 方案 B: 使用项目现有虚拟环境

```bash
# 1. 进入项目目录
cd papers\Yolo-v8

# 2. 激活虚拟环境
.venv\Scripts\activate

# 3. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

---

## CPU 训练建议

由于 CPU 训练速度较慢，建议使用以下配置：

### 1. 使用小模型

| 模型 | 参数量 | CPU 推理速度 | 适用场景 |
|------|--------|-------------|----------|
| YOLOv8n | ~3M | 最快 | 移动端/快速迭代 |
| YOLOv8s | ~11M | 快 | 通用场景（推荐） |
| YOLOv8m | ~26M | 中等 | 高精度需求 |

### 2. 调整 batch size

CPU 训练时使用较小的 batch size：

```bash
# 推荐设置
python train.py --batch 4 --imgsz 640
```

### 3. 使用预训练权重

```python
from ultralytics import YOLO

# 使用预训练模型（大幅减少训练时间）
model = YOLO('yolov8n.pt')  # 或 yolov8s.pt

# 训练
results = model.train(
    data='your_dataset.yaml',
    epochs=50,
    batch=4,
    imgsz=640,
    device='cpu'  # 强制使用 CPU
)
```

### 4. 减少数据加载线程

```yaml
# 在训练配置中
workers: 0  # CPU 训练时使用 0 或 2
```

---

## 训练示例

### 使用项目中的 Fabric 缺陷检测数据集

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 训练（CPU 模式）
python train_fabric.py --batch 4 --workers 0 --device cpu
```

### 直接训练

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 训练
model.train(
    data='papers/Yolo-v8/fabric-defect.yaml',
    epochs=100,
    batch=4,
    imgsz=640,
    device='cpu',
    workers=0
)
```

---

## 推理示例

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/train/weights/best.pt')

# 推理
results = model('path/to/image.jpg', device='cpu')

# 显示结果
results[0].show()
```

---

## 性能优化建议

### 1. 使用 OpenVINO 加速（Intel CPU）

```bash
pip install openvino-dev
```

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 导出为 OpenVINO 格式
model.export(format='openvino')

# 使用 OpenVINO 推理
ov_model = YOLO('yolov8n_openvino_model')
results = ov_model('image.jpg')
```

### 2. 使用 ONNX Runtime

```bash
pip install onnx onnxruntime
```

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx')

# ONNX 推理
import onnxruntime as ort
session = ort.InferenceSession('yolov8n.onnx')
```

### 3. 启用 Intel AMX（13 代+ Intel CPU）

```bash
# 安装针对 Intel 优化的 PyTorch
pip install torch torchvision intel_extension_for_pytorch
```

---

## 预期性能

### Intel Core i5-12400 (6 核)

| 任务 | 设置 | 速度 |
|------|------|------|
| YOLOv8n 推理 | 640x640 | ~15-20 FPS |
| YOLOv8s 推理 | 640x640 | ~8-12 FPS |
| YOLOv8n 训练 | batch=4 | ~0.5-1 epoch/hour |

### Intel Core i7-13700K (16 核)

| 任务 | 设置 | 速度 |
|------|------|------|
| YOLOv8n 推理 | 640x640 | ~25-35 FPS |
| YOLOv8s 推理 | 640x640 | ~15-20 FPS |
| YOLOv8n 训练 | batch=4 | ~1-2 epochs/hour |

---

## 常见问题

### Q: 训练速度太慢怎么办？
A: 
- 使用更小的模型（YOLOv8n）
- 减少 batch size 到 2 或 4
- 使用预训练权重减少训练轮数
- 考虑使用云 GPU 服务（如 Colab、Kaggle）

### Q: 内存不足怎么办？
A:
- 减小 imgsz（如 320x320）
- 减小 batch size
- 关闭数据缓存（cache=False）

### Q: 如何查看 CPU 利用率？
A:
```python
import psutil
print(f"CPU 使用率：{psutil.cpu_percent(interval=1)}%")
```

---

## 云 GPU 替代方案

如果 CPU 训练太慢，可使用免费云 GPU：

1. **Google Colab** (免费 T4 GPU)
2. **Kaggle Notebooks** (免费 P100 GPU)
3. **阿里云 PAI** (免费额度)

---

## 验证安装

```python
# test_cpu.py
import torch
from ultralytics import YOLO

print("=" * 50)
print(f"PyTorch 版本：{torch.__version__}")
print(f"CUDA 可用：{torch.cuda.is_available()}")
print(f"CPU 核心数：{torch.get_num_threads()}")
print("=" * 50)

# 测试模型加载
model = YOLO('yolov8n.pt')
print("模型加载成功！")

# 测试推理
import numpy as np
from PIL import Image

# 创建测试图像
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
results = model(img, device='cpu')
print("推理成功！")
print(f"检测到的物体数量：{len(results[0].boxes)}")
```

运行：
```bash
python test_cpu.py
```

---

## 总结

| 步骤 | 命令 |
|------|------|
| 1. 创建虚拟环境 | `python -m venv yolo-cpu-env` |
| 2. 激活虚拟环境 | `yolo-cpu-env\Scripts\activate` |
| 3. 安装 PyTorch CPU | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| 4. 安装 Ultralytics | `pip install ultralytics` |
| 5. 验证安装 | `python -c "import torch; print(torch.__version__)"` |

祝使用愉快！🚀
