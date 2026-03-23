# 织物缺陷检测 - 快速开始指南

## 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install pillow
```

---

## 2. 数据集准备

### 2.1 目录结构

将你的数据集整理成以下结构：

```
datasets/
└── fabric-defect/
    ├── images/              # 所有原始图片
    │   ├── defect_001.jpg
    │   ├── defect_002.jpg
    │   └── ...
    ├── labels/              # YOLO 格式标注 (.txt)
    │   ├── defect_001.txt
    │   ├── defect_002.txt
    │   └── ...
    └── dataset.yaml         # (可选) 如果已有划分
```

### 2.2 YOLO 标注格式

每个 `.txt` 文件格式（每行一个目标）：
```
<class-id> <x_center> <y_center> <width> <height>
```

示例：
```
0 0.45 0.52 0.08 0.06
1 0.72 0.38 0.12 0.15
```

**坐标说明**：
- 所有值归一化到 [0, 1]
- `x_center`, `y_center`: 边界框中心点
- `width`, `height`: 边界框宽高

### 2.3 划分数据集

如果还没有划分训练集和验证集：

```bash
# 进入项目目录
cd D:\Claude-prj\papers\Yolo-v8

# 8:2 划分训练集和验证集
python split_dataset.py --source datasets/fabric-defect --split 0.8

# 划分后的结构:
# datasets/fabric-defect/
# ├── images/
# │   ├── train/    # 训练图片
# │   └── val/      # 验证图片
# └── labels/
#     ├── train/    # 训练标注
#     └── val/      # 验证标注
```

---

## 3. 配置文件

编辑 `fabric-defect.yaml`，修改类别名称：

```yaml
# fabric-defect.yaml

path: datasets/fabric-defect
train: images/train
val: images/val

nc: 6  # 修改为你的缺陷类别数

names:
  0: hole           # 修改为你的缺陷类别
  1: stain
  2: weft_skew
  3: missing_yarn
  4: oil_spot
  5: broken_warp
```

---

## 4. 开始训练

### 方法 1: 使用专用训练脚本（推荐）

```bash
# 基础训练
python train_fabric.py --data fabric-defect.yaml --epochs 100 --batch 16

# 指定模型和参数
python train_fabric.py \
    --model yolov8s-strip.yaml \
    --data fabric-defect.yaml \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --name fabric-defect-exp1
```

### 方法 2: 使用 Ultralytics YOLO

```bash
# 训练
yolo detect train \
    model=yolov8s-strip.yaml \
    data=fabric-defect.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=runs/detect \
    name=fabric-defect-exp1
```

### 方法 3: 使用原始训练脚本

```bash
python train.py \
    --cfg yolov8s-strip.yaml \
    --data fabric-defect.yaml \
    --batch 16 \
    --epochs 100 \
    --img 640 \
    --device 0
```

---

## 5. 训练监控

### 查看训练输出

训练结果保存在 `runs/detect/fabric-defect-exp1/`:

```
runs/detect/fabric-defect-exp1/
├── weights/
│   ├── best.pt           # 最佳模型 ⭐
│   └── last.pt           # 最后一轮模型
├── results.csv           # 训练数据
├── results.png           # 训练曲线
├── confusion_matrix.png  # 混淆矩阵
└── ...
```

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir runs/detect

# 浏览器访问 http://localhost:6006
```

---

## 6. 验证模型

```bash
# 在验证集上评估
python val_fabric.py val \
    --weights runs/detect/fabric-defect-exp1/weights/best.pt \
    --data fabric-defect.yaml \
    --batch 16 \
    --imgsz 640
```

---

## 7. 推理预测

### 预测单张图片

```bash
# 预测单张图片
python predict_fabric.py \
    --weights runs/detect/fabric-defect-exp1/weights/best.pt \
    --source datasets/fabric-defect/images/val/defect_101.jpg \
    --conf 0.5

# 预测并保存结果
python predict_fabric.py \
    --weights best.pt \
    --source image.jpg \
    --save-txt \
    --save-conf
```

### 批量预测

```bash
# 预测整个目录
python val_fabric.py batch \
    --weights runs/detect/fabric-defect-exp1/weights/best.pt \
    --source-dir datasets/fabric-defect/images/val \
    --output-dir runs/detect/results \
    --conf 0.5
```

### Python 代码推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/fabric-defect-exp1/weights/best.pt')

# 推理
results = model('path/to/image.jpg', conf=0.5)

# 处理结果
for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            bbox = box.xyxy[0]
            print(f'{result.names[cls]}: {conf:.3f} - {bbox}')
```

---

## 8. 模型导出

```bash
# 导出为 ONNX (用于部署)
yolo export model=runs/detect/fabric-defect-exp1/weights/best.pt format=onnx

# 导出为 TorchScript
yolo export model=runs/detect/fabric-defect-exp1/weights/best.pt format=torchscript

# 导出为 OpenVINO
yolo export model=runs/detect/fabric-defect-exp1/weights/best.pt format=openvino
```

---

## 9. 常见问题

### Q1: 显存不足
```bash
# 减小 batch size
python train_fabric.py --batch 8  # 或更小

# 减小图像尺寸
python train_fabric.py --imgsz 512
```

### Q2: 训练太慢
```bash
# 使用多 GPU
python train_fabric.py --device 0,1

# 增加 workers 数量
python train_fabric.py --workers 8
```

### Q3: 检测效果不好
```bash
# 增加训练轮数
python train_fabric.py --epochs 200

# 使用更大的模型
python train_fabric.py --model yolov8m-strip.yaml

# 调整学习率
python train_fabric.py --lr0 0.0001
```

### Q4: 类别不平衡
- 对少数类进行过采样
- 在 loss 中使用 class_weight
- 使用 Focal Loss

---

## 10. 推荐配置

| 场景 | 模型 | epochs | batch | imgsz | 预计 mAP |
|------|------|--------|-------|-------|----------|
| 快速测试 | YOLOv8n-Strip | 50 | 16 | 640 | ~75% |
| 标准训练 | YOLOv8s-Strip | 100 | 16 | 640 | ~85% |
| 高精度 | YOLOv8m-Strip | 150 | 8 | 1024 | ~90% |
| 实时检测 | YOLOv8n-Strip | 100 | 32 | 512 | ~80% |

---

## 11. 训练完成后的检查清单

- [ ] 检查 `results.png` 确认损失下降
- [ ] 检查 `confusion_matrix.png` 查看分类效果
- [ ] 在验证集上运行 `val_fabric.py` 获取 mAP
- [ ] 随机抽取测试图片进行推理验证
- [ ] 导出模型用于部署

---

**最后更新**: 2026-03-22

**祝训练顺利！🎉**
