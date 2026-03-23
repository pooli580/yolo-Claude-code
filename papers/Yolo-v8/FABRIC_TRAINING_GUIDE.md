# 织物缺陷检测 - 训练指南

## 快速开始

### 1. 数据集准备

#### 数据集目录结构
```
datasets/
└── fabric-defect/          # 织物缺陷数据集根目录
    ├── images/
    │   ├── train/          # 训练图像
    │   │   ├── defect_001.jpg
    │   │   ├── defect_002.jpg
    │   │   └── ...
    │   └── val/            # 验证图像
    │       ├── defect_101.jpg
    │       └── ...
    └── labels/
        ├── train/          # 训练标注 (YOLO 格式.txt)
        │   ├── defect_001.txt
        │   ├── defect_002.txt
        │   └── ...
        └── val/            # 验证标注
            ├── defect_101.txt
            └── ...
```

#### YOLO 标注格式说明
每个 `.txt` 文件与对应图像同名，格式为：
```
<class-id> <x_center> <y_center> <width> <height>
```

示例 (`defect_001.txt`):
```
0 0.45 0.52 0.08 0.06
1 0.72 0.38 0.12 0.15
3 0.25 0.67 0.05 0.20
```

**注意**:
- `class-id`: 类别 ID (从 0 开始)
- 坐标均为归一化值 (0-1 之间)
- 每行一个目标

---

### 2. 配置文件

#### 创建数据集配置文件 `fabric-defect.yaml`

```yaml
# datasets/fabric-defect.yaml

# 数据集路径 (相对于当前工作目录)
path: datasets/fabric-defect
train: images/train
val: images/val

# 类别信息
nc: 6  # 缺陷类别数

# 缺陷类别名称 (根据实际情况修改)
names:
  0: hole           # 破洞
  1: stain          # 污渍
  2: weft_skew      # 纬斜
  3: missing_yarn   # 缺纱
  4: oil_spot       # 油斑
  5: broken_warp    # 断经
```

---

### 3. 开始训练

#### 方法 1: 使用训练脚本 (推荐)

```bash
# 进入项目目录
cd D:\Claude-prj\papers\Yolo-v8

# 开始训练
python train.py \
    --cfg yolov8s-strip.yaml \
    --data datasets/fabric-defect.yaml \
    --batch 16 \
    --img 640 \
    --epochs 100 \
    --device 0

# 参数说明:
#   --cfg       模型配置文件
#   --data      数据集配置文件
#   --batch     批次大小 (根据显存调整)
#   --img       输入图像尺寸
#   --epochs    训练轮数
#   --device    GPU 设备 (0 或 cpu，多卡用 0,1,2,3)
```

#### 方法 2: 使用 Ultralytics YOLO

```bash
# 安装 ultralytics (如果还没安装)
pip install ultralytics

# 训练
yolo detect train \
    model=yolov8s-strip.yaml \
    data=datasets/fabric-defect.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=runs/detect \
    name=fabric-defect-exp1
```

#### 方法 3: Python 代码训练

```python
from ultralytics import YOLO

# 加载模型配置
model = YOLO('yolov8s-strip.yaml')

# 开始训练
results = model.train(
    data='datasets/fabric-defect.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,           # GPU 设备
    workers=4,          # 数据加载线程数
    optimizer='AdamW',  # 优化器
    lr0=0.001,          # 初始学习率
    patience=30,        # 早停耐心值
    save_period=10,     # 每多少轮保存一次
)
```

---

### 4. 训练参数配置

#### 推荐配置 (织物缺陷检测)

```yaml
# 超参数配置
epochs: 100             # 训练轮数
batch_size: 16          # 批次大小 (根据显存调整)
imgsz: 640              # 输入尺寸
patience: 30            # 早停耐心值

# 优化器
optimizer: AdamW
lr0: 0.001              # 初始学习率
lrf: 0.01               # 最终学习率 (lr0 * lrf)
momentum: 0.937         # SGD 动量/Adam beta1
weight_decay: 0.05      # 权重衰减

# 数据增强
hsv_h: 0.015            # 色调增强
hsv_s: 0.7              # 饱和度增强
hsv_v: 0.4              # 亮度增强
degrees: 2.0            # 旋转角度
translate: 0.1          # 平移
scale: 0.5              # 缩放
shear: 0.0              # 剪切
perspective: 0.0        # 透视
flipud: 0.0             # 上下翻转概率
fliplr: 0.5             # 左右翻转概率
mosaic: 1.0             # Mosaic 增强概率
mixup: 0.0              # MixUp 增强概率
```

---

### 5. 训练监控

#### 查看训练日志
训练输出目录 (`runs/detect/fabric-defect-exp1/`):
```
runs/detect/fabric-defect-exp1/
├── weights/
│   ├── best.pt           # 最佳模型
│   └── last.pt           # 最后一轮模型
├── results.csv           # 训练结果 CSV
├── results.png           # 训练曲线
├── confusion_matrix.png  # 混淆矩阵
├── labels.jpg            # 标注分布可视化
└── train_batch*.jpg      # 训练批次可视化
```

#### TensorBoard 监控
```bash
# 启动 TensorBoard
tensorboard --logdir runs/detect

# 浏览器访问 http://localhost:6006
```

---

### 6. 验证和测试

#### 验证模型
```bash
# 在验证集上评估
python val.py \
    --weights runs/detect/fabric-defect-exp1/weights/best.pt \
    --data datasets/fabric-defect.yaml \
    --batch 16 \
    --img 640
```

#### 测试单张图片
```bash
# 预测单张图片
python predict.py \
    --weights runs/detect/fabric-defect-exp1/weights/best.pt \
    --source datasets/fabric-defect/images/val/defect_101.jpg \
    --conf 0.5  # 置信度阈值
```

#### Python 代码推理
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/fabric-defect-exp1/weights/best.pt')

# 推理单张图片
results = model('path/to/image.jpg')

# 推理整个目录
results = model('path/to/images/folder', save=True, save_txt=True)

# 处理结果
for result in results:
    boxes = result.boxes          # 边界框
    probs = result.probs          # 类别概率
    names = result.names          # 类别名称

    for box in boxes:
        print(f'Class: {names[int(box.cls)]}, Conf: {box.conf:.3f}')
        print(f'BBox: {box.xyxy[0]}')
```

---

### 7. 模型导出

```bash
# 导出为 ONNX 格式
yolo export model=runs/detect/fabric-defect-exp1/weights/best.pt format=onnx

# 导出为 TorchScript
yolo export model=runs/detect/fabric-defect-exp1/weights/best.pt format=torchscript

# 导出为 OpenVINO
yolo export model=runs/detect/fabric-defect-exp1/weights/best.pt format=openvino
```

---

## 常见问题

### Q1: CUDA Out of Memory
**解决方案**:
- 减小 `batch_size` (如 16 -> 8 -> 4)
- 减小 `imgsz` (如 640 -> 512)
- 使用梯度累积：`accumulate=4`

### Q2: 训练不收敛
**解决方案**:
- 检查标注格式是否正确
- 降低学习率：`lr0=0.0001`
- 增加 warmup: `warmup_epochs=5`

### Q3: 检测效果不好
**解决方案**:
- 增加训练轮数：`epochs=200`
- 增加数据增强强度
- 使用更大的模型：`yolov8m-strip`
- 检查类别是否平衡

### Q4: 数据集划分的建议
- **小规模数据集** (< 1000 张): 8:2 划分
- **中等数据集** (1000-10000 张): 8:1:1 划分 (train:val:test)
- **大规模数据集** (> 10000 张): 98:1:1 划分

---

## 性能优化建议

### 针对织物缺陷的优化

1. **细长缺陷 (纬斜、断经)**:
   - 使用 CASM 模块增强
   - 增加条带卷积核大小：`strip_kernel_size=25`

2. **小缺陷 (小破洞、油点)**:
   - 使用更高分辨率：`imgsz=1024`
   - 增加小目标检测层

3. **实时检测需求**:
   - 使用 YOLOv8n-Strip
   - 减小输入尺寸：`imgsz=416`
   - 导出为 TensorRT

---

## 参考资源

- YOLOv8 文档：https://docs.ultralytics.com/
- 本项目文档：`README.md`, `IMPLEMENTATION.md`
- Strip R-CNN 论文：https://arxiv.org/abs/2501.03775

---

**最后更新**: 2026-03-22
