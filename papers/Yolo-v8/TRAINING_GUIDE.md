# YOLOv8-Strip 训练与使用指南

## 一、项目结构（已精简）

```
Yolo-v8/
├── core/                   # 核心模块（必需）
│   ├── __init__.py         # 导出：StripModule, StripC2f, yolov8s_strip 等
│   ├── strip_conv.py       # Strip 卷积模块
│   ├── strip_net.py        # StripNet 骨干网络
│   ├── strip_head.py       # Strip 检测头
│   └── yolov8_strip.py     # 完整 YOLOv8-Strip 模型
│
├── configs/                # 配置文件
│   ├── yolov8s-strip.yaml  # 模型配置
│   └── fabric-defect.yaml  # 织物数据集配置
│
├── examples/               # 示例代码
│   ├── 01_create_model.py  # 创建模型
│   ├── 02_train.py         # 训练
│   ├── 03_inference.py     # 推理
│   └── 04_custom_dataset.py # 自定义数据集
│
├── train_fabric.py         # 训练脚本（织物）
├── predict_fabric.py       # 预测脚本（织物）
└── README-QUICKSTART.md    # 快速开始
```

---

## 二、快速开始（3 步训练）

### 步骤 1: 安装依赖

```bash
pip install torch torchvision ultralytics
```

### 步骤 2: 准备数据集

创建 `fabric-defect.yaml`:

```yaml
path: data/fabric
train: images/train
val: images/val

nc: 4
names:
  0: weaver_break    # 断经
  1: weft_stop       # 断纬
  2: hole            # 破洞
  3: oil_stain       # 油污
```

目录结构:

```
data/fabric/
├── images/
│   ├── train/      # 训练图像
│   └── val/        # 验证图像
└── labels/
    ├── train/      # YOLO 格式标注
    └── val/
```

### 步骤 3: 开始训练

```bash
# 方法 1: 使用训练脚本（推荐）
python train_fabric.py --data fabric-defect.yaml --epochs 100 --batch 16

# 方法 2: 使用 Ultralytics CLI
yolo train model=yolov8s-strip.yaml data=fabric-defect.yaml epochs=100 batch=16
```

---

## 三、如何使用模块

### 1. 创建模型

```python
from __init__ import yolov8s_strip

# 创建 Small 版本（80 类）
model = yolov8s_strip(num_classes=80)

# 创建织物缺陷检测模型（4 类）
model_fabric = yolov8s_strip(num_classes=4)
```

### 2. 训练模型

**使用 Python API:**

```python
from ultralytics import YOLO

# 加载模型配置
model = YOLO('yolov8s-strip.yaml')

# 训练
model.train(
    data='fabric-defect.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='0'  # GPU 0
)
```

**使用命令行:**

```bash
yolo train model=yolov8s-strip.yaml data=fabric-defect.yaml epochs=100
```

### 3. 推理预测

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/fabric-exp/weights/best.pt')

# 预测单张图片
results = model('image.jpg')

# 预测整个目录
results = model(source='data/images/', save=True)
```

### 4. 使用核心模块构建自定义模型

```python
from __init__ import StripModule, StripC2f, StripNet, yolov8s_strip

# 方法 1: 使用预定义模型
model = yolov8s_strip(num_classes=80)

# 方法 2: 自定义骨干网络
backbone = StripNet(
    channels=[64, 128, 256, 512],
    depths=[2, 2, 3, 2],
    strip_kernel_size=19,
)

# 方法 3: 在现有模型中使用 Strip 模块
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 StripModule 替代普通卷积
        self.strip_block = StripModule(
            in_channels=256,
            out_channels=256,
            strip_kernel_size=19,
            shortcut=True,
        )

        # 使用 StripC2f 替代 C2f
        self.strip_c2f = StripC2f(
            in_channels=512,
            out_channels=512,
            num_blocks=3,
            strip_kernel_size=19,
        )

    def forward(self, x):
        x = self.strip_block(x)
        x = self.strip_c2f(x)
        return x
```

---

## 四、运行示例代码

每个示例都可以直接运行:

```bash
# 示例 1: 创建模型
cd papers/Yolo-v8
python examples/01_create_model.py

# 示例 2: 训练
python examples/02_train.py

# 示例 3: 推理
python examples/03_inference.py

# 示例 4: 自定义数据集
python examples/04_custom_dataset.py
```

---

## 五、训练技巧

### 学习率调整

| 现象 | 调整方案 |
|------|----------|
| Loss 不下降 | `lr0` × 10 |
| Loss 震荡 | `lr0` ÷ 10 |
| 收敛慢 | 使用 `lr0=0.01` + `warmup_epochs=5` |

### Batch Size

| GPU 显存 | 推荐 batch |
|----------|-----------|
| 8GB | 8-16 |
| 16GB | 16-32 |
| 24GB | 32-64 |

### 输入尺寸

| 场景 | imgsz |
|------|-------|
| 通用检测 | 640 |
| 遥感图像 | 1024 |
| 小目标 | 1280 |

---

## 六、输出说明

### 训练输出

```
runs/detect/fabric-exp/
├── weights/
│   ├── best.pt       # 最佳模型
│   └── last.pt       # 最后一轮模型
├── args.yaml         # 训练参数
├── results.csv       # 训练指标
└── confusion_matrix.png
```

### 模型输出格式

```python
outputs = model(img)

# 输出是字典，包含 3 个列表（P3, P4, P5 三个尺度）
outputs['cls']    # 分类得分列表
outputs['loc']    # 边界框列表
outputs['angle']  # 角度列表（旋转检测）
```

---

## 七、常见问题

### Q: CUDA Out of Memory
```bash
# 减小 batch size
python train_fabric.py --batch 8

# 减小图像尺寸
python train_fabric.py --imgsz 320
```

### Q: ModuleNotFoundError
```bash
# 确保在项目根目录运行
cd D:/Claude-prj/papers/Yolo-v8
python train_fabric.py
```

### Q: 如何加载预训练权重
```python
# 方法 1: Ultralytics
model = YOLO('yolov8s.pt')  # COCO 预训练

# 方法 2: 自定义权重
model = yolov8s_strip(num_classes=80)
model.load_state_dict(torch.load('best.pt'))
```

---

## 八、下一步

1. **阅读示例代码**: `examples/` 目录下的 4 个示例
2. **开始训练**: `python train_fabric.py`
3. **查看结果**: TensorBoard 或 `results.csv`
4. **部署模型**: 导出为 ONNX 或 TorchScript

---

## 参考资料

- Strip R-CNN 论文：https://arxiv.org/abs/2501.03775
- Ultralytics 文档：https://docs.ultralytics.com
