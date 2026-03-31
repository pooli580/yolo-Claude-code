# YOLOv8 云端训练快速开始指南

## 📋 前提条件

- 云端服务器已安装 Python 3.8+
- 已安装 CUDA 和 PyTorch（GPU 训练）
- 已上传代码和数据集到服务器

---

## 🚀 快速验证（5 分钟）

### Step 1: 上传文件到服务器

```bash
# 本地执行（在你的 Windows 机器）
# 使用 scp 或 rsync 上传

# 方法 1: scp 上传整个项目
scp -r papers/Yolo-v8 user@your-server:/path/to/project/

# 方法 2: rsync（推荐，支持断点续传）
rsync -avz --progress papers/Yolo-v8/ user@your-server:/path/to/project/

# 方法 3: 如果有数据集
scp -r datasets/fabric-defect user@your-server:/path/to/datasets/
```

### Step 2: 登录服务器并检查环境

```bash
# SSH 登录
ssh user@your-server

# 进入项目目录
cd /path/to/project/Yolo-v8

# 运行环境检查
python check_environment.py
```

### Step 3: 快速测试（CPU/GPU）

```bash
# 测试模型加载和推理
python -c "
from ultralytics import YOLO
import torch

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# 加载模型
model = YOLO('yolov8n.pt')
print('✅ 模型加载成功')

# 快速推理测试
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)
print('✅ 推理测试成功')
print(f'检测结果：runs/detect/predict/bus.jpg')
"
```

---

## 🏋️ 开始训练

### 方法 1: 一键训练脚本（推荐）

```bash
# 基础训练
./train.sh fabric-defect.yaml yolov8s-strip.yaml 100 16 0

# 参数说明:
#   fabric-defect.yaml    - 数据集配置
#   yolov8s-strip.yaml    - 模型配置
#   100                   - epochs
#   16                    - batch size
#   0                     - GPU 设备 (0 表示 GPU 0)
```

### 方法 2: Python 命令

```bash
python train_fabric.py \
    --model yolov8s-strip.yaml \
    --data fabric-defect.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0 \
    --name fabric-exp-001
```

### 方法 3: YOLO 官方命令

```bash
yolo detect train \
    model=yolov8s-strip.yaml \
    data=fabric-defect.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=runs/detect \
    name=fabric-exp-001
```

---

## 📊 监控训练

### 查看训练输出

```bash
# 实时查看训练日志
tail -f runs/detect/fabric-exp-001/results.csv

# 查看保存的权重
ls -lh runs/detect/fabric-exp-001/weights/
```

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir runs/detect --host 0.0.0.0 --port 6006

# 本地浏览器访问
# http://your-server-ip:6006
```

### 训练曲线图片

训练完成后查看：
```bash
# 训练结果图片
open runs/detect/fabric-exp-001/results.png  # macOS
xdg-open runs/detect/fabric-exp-001/results.png  # Linux
```

---

## ✅ 验证模型

### 在验证集上评估

```bash
# 使用验证脚本
./val.sh runs/detect/fabric-exp-001/weights/best.pt fabric-defect.yaml

# 或直接执行
python val_fabric.py val \
    --weights runs/detect/fabric-exp-001/weights/best.pt \
    --data fabric-defect.yaml \
    --batch 16 \
    --device 0
```

### 预期输出

```
Validating saves/detect/fabric-exp-001/weights/best.pt...
Ultralytics 8.4.26  Python 3.10.0-0000-00-00

Dataset: fabric-defect (6 classes)
                 Class     Images  Instances      P      R   mAP50  mAP50-95
                   all          0        0     0.72    0.68     0.74      0.52
                  hole          0        0     0.75    0.70     0.78      0.55
                 stain          0        0     0.70    0.65     0.72      0.50
              weft_skew          0        0     0.68    0.62     0.69      0.48
           missing_yarn          0        0     0.74    0.71     0.76      0.54
              oil_spot          0        0     0.73    0.69     0.75      0.53
            broken_warp          0        0     0.71    0.67     0.73      0.51
Speed: 0.3ms preprocess, 2.1ms inference, 0.0ms loss, 0.8ms postprocess per image
Results saved to runs/detect/fabric-exp-001
```

---

## 🔍 推理预测

### 预测单张图片

```bash
python predict_fabric.py \
    --weights runs/detect/fabric-exp-001/weights/best.pt \
    --source datasets/fabric-defect/images/val/defect_001.jpg \
    --conf 0.5 \
    --save-txt
```

### 批量预测整个目录

```bash
python predict_fabric.py \
    --weights runs/detect/fabric-exp-001/weights/best.pt \
    --source datasets/fabric-defect/images/val \
    --conf 0.5 \
    --save-txt \
    --save-conf
```

### Python 代码推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/fabric-exp-001/weights/best.pt')

# 推理
results = model.predict(
    source='path/to/image.jpg',
    conf=0.5,
    save=True,
    save_txt=True
)

# 处理结果
for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            bbox = box.xyxy[0].tolist()
            print(f'{result.names[cls]}: {conf:.3f} - bbox: {bbox}')
```

---

## 📦 模型导出

```bash
# 导出为 ONNX（用于部署）
yolo export model=runs/detect/fabric-exp-001/weights/best.pt format=onnx

# 导出为 TorchScript
yolo export model=runs/detect/fabric-exp-001/weights/best.pt format=torchscript

# 导出为 OpenVINO
yolo export model=runs/detect/fabric-exp-001/weights/best.pt format=openvino
```

---

## 🔧 常见问题

### Q1: 显存不足 (CUDA out of memory)

```bash
# 减小 batch size
python train_fabric.py --batch 8

# 减小图像尺寸
python train_fabric.py --imgsz 512

# 使用更小的模型
python train_fabric.py --model yolov8n-strip.yaml
```

### Q2: 训练太慢

```bash
# 使用多 GPU
python train_fabric.py --device 0,1,2,3

# 增加数据加载线程
python train_fabric.py --workers 8

# 使用混合精度训练
python train_fabric.py --amp
```

### Q3: 检测效果不好

```bash
# 增加训练轮数
python train_fabric.py --epochs 200

# 使用更大的模型
python train_fabric.py --model yolov8m-strip.yaml

# 调整学习率
python train_fabric.py --lr0 0.0001

# 增加数据增强
python train_fabric.py --augment
```

### Q4: 数据集路径问题

确保 `fabric-defect.yaml` 中的路径正确：

```yaml
# 使用绝对路径
path: /absolute/path/to/datasets/fabric-defect

# 或使用相对路径（相对于训练时的当前目录）
path: datasets/fabric-defect
```

---

## 📈 训练配置推荐

| 场景 | 模型 | Epochs | Batch | ImgSz | 预计时间 | 预计 mAP |
|------|------|--------|-------|-------|----------|----------|
| 快速测试 | YOLOv8n-Strip | 50 | 16 | 640 | ~30min | ~75% |
| 标准训练 | YOLOv8s-Strip | 100 | 16 | 640 | ~1h | ~85% |
| 高精度 | YOLOv8m-Strip | 150 | 8 | 1024 | ~3h | ~90% |
| 实时检测 | YOLOv8n-Strip | 100 | 32 | 512 | ~45min | ~80% |

---

## 📁 文件结构

训练完成后的目录结构：

```
runs/detect/fabric-exp-001/
├── weights/
│   ├── best.pt           # 最佳模型 ⭐
│   └── last.pt           # 最后一轮模型
├── results.csv           # 训练数据
├── results.png           # 训练曲线
├── confusion_matrix.png  # 混淆矩阵
├── labels.jpg            # 标注分布
├── train_batch*.jpg      # 训练批次可视化
├── val_batch*.jpg        # 验证批次可视化
└── args.yaml             # 训练参数
```

---

## 🎯 下一步

1. **训练完成后**: 运行验证脚本获取 mAP 指标
2. **效果不理想**: 分析混淆矩阵，调整超参数
3. **准备部署**: 导出模型为 ONNX/TorchScript
4. **持续改进**: 记录实验结果，迭代优化

---

**祝训练顺利！** 🎉

如需帮助，运行：
```bash
python train_fabric.py --help
```
