# YOLOv8-Strip: Large Strip Convolution for YOLOv8

基于 **Strip R-CNN** 论文实现的 YOLOv8 改进版本，专门针对**高长宽比物体检测**优化。

## 论文信息

**Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection**

- **arXiv**: https://arxiv.org/abs/2501.03775
- **GitHub**: https://github.com/HVision-NKU/Strip-R-CNN
- **作者**: Xinbin Yuan, Zhaohui Zheng, Yuxuan Li, et al. (南开大学 VCIP)

### 核心创新

1. **Large Strip Convolution**: 使用正交条带卷积 (H-Strip + V-Strip) 替代传统方形卷积
2. **StripNet Backbone**: 简单高效的主干网络，仅 13.3M 参数
3. **Strip Head**: 解耦检测头，定位分支使用 strip 卷积增强空间依赖

### 主要结果

| 数据集 | mAP | 参数量 | FLOPs |
|--------|-----|--------|-------|
| DOTA-v1.0 | 82.75% | 13.3M | 52.3G |
| FAIR1M-v1.0 | 48.26% | 13.3M | 52.3G |
| HRSC2016 | 98.70% | 13.3M | 52.3G |
| DIOR-R | 68.70% | 13.3M | 52.3G |

---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision
```

### 基本使用

```python
import torch
from yolov8_strip import yolov8s_strip

# 创建模型
model = yolov8s_strip(num_classes=15)  # DOTA 数据集 15 类
model.eval()

# 前向传播
img = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    outputs = model(img)

# 输出格式
print(outputs['cls'])   # 分类得分列表 [P3, P4, P5]
print(outputs['loc'])   # 边界框列表 [P3, P4, P5]
print(outputs['angle']) # 角度列表 [P3, P4, P5]
```

### 模型变体

| 模型 | 函数 | 参数量 | 推荐用途 |
|------|------|--------|----------|
| Nano | `yolov8n_strip()` | ~4M | 移动端/实时 |
| Small | `yolov8s_strip()` | ~26M | 通用 (推荐) |
| Medium | `yolov8m_strip()` | ~50M | 高精度 |
| Large | `yolov8l_strip()` | ~90M | 最高精度 |

---

## 架构设计

### 整体架构

```
输入图像 (3, H, W)
    |
    v
+------------------+
|  StripNet        |
|  Backbone        |
|  - Stem          |
|  - Stage1-4      |
|  输出：P3, P4, P5|
+------------------+
    |
    v
+------------------+
|  FPN+PAN Neck    |
|  (StripC2f)      |
+------------------+
    |
    v
+------------------+
|  Strip Detect    |
|  Head            |
|  - 分类分支      |
|  - 定位分支      |
|  - 角度分支      |
+------------------+
    |
    v
输出：cls, loc, angle
```

### StripModule 结构

```
输入 X
  |
  +---> [5x5 DW-Conv] -> BN -> GELU ---+
  |                                     |
  |                  +---> [1x19 DW-Conv] --+
  |                  |                      |
  |                  +---> [19x1 DW-Conv] --+--> [拼接]
  |                                         |
  |                  [1x1 PW-Conv] -> BN -->|
  |                                         |
  +-----------------> x (注意力加权) --------+
                      |
                      v
                   输出 Y
```

---

## 模块说明

### 文件结构

```
papers/Yolo-v8/
|-- __init__.py           # 模块入口
|-- strip_conv.py         # Strip 卷积核心模块
|   |-- LargeStripConv    # 大条带卷积 (H-Strip + V-Strip)
|   |-- StripConv         # 完整 Strip 卷积模块
|   |-- StripModule       # Strip 基本构建块
|   +-- StripC2f          # YOLOv8 C2f 的 Strip 版本
|-- strip_net.py          # StripNet 骨干网络
|   |-- StripNet          # 骨干网络类
|   |-- strip_net_tiny    # Tiny 版本 (~4M)
|   |-- strip_net_small   # Small 版本 (~13M, 推荐)
|   +-- strip_net_base    # Base 版本
|-- strip_head.py         # Strip 检测头
|   |-- ClassificationBranch  # 分类分支
|   |-- LocalizationBranch  # 定位分支 (含 Strip Module)
|   |-- AngleBranch       # 角度预测分支
|   |-- StripHead         # 完整检测头
|   +-- StripDetectHead   # 多尺度检测头
|-- yolov8_strip.py       # 完整 YOLOv8-Strip 模型
|-- yolov8s-strip.yaml    # YOLOv8 配置文件
+-- README.md             # 本文档
```

### API 参考

#### LargeStripConv

```python
conv = LargeStripConv(
    channels=64,           # 输入通道
    kernel_size=19,        # 条带核大小（奇数）
    groups=64,             # 分组数 (depthwise)
)
h_feat, v_feat = conv(x)   # 返回水平和垂直特征
```

#### StripModule

```python
module = StripModule(
    in_channels=64,
    out_channels=64,
    strip_kernel_size=19,
    shortcut=True,
)
out = module(x)
```

#### StripC2f

```python
c2f = StripC2f(
    in_channels=256,
    out_channels=256,
    num_blocks=3,
    strip_kernel_size=19,
)
out = c2f(x)
```

---

## 配置说明

### 条带卷积核大小

根据论文 Table 8 消融实验：

| 配置 | mAP | 说明 |
|------|-----|------|
| (19,19,19,19) | 81.75 | 推荐（最优） |
| (15,15,15,15) | 81.64 | 稍差 |
| (11,11,11,11) | 81.22 | 较差 |

### 关键设计

1. **序列式 vs 并行**: 序列式 (先 H 后 V) 效果更好
2. **初始平方卷积**: 5x5 平方卷积是必要的，移除会导致性能下降
3. **膨胀卷积**: 可用但效果不如标准大核

---

## 训练指南

### 数据集配置 (DOTA)

```yaml
path: data/DOTA
train: train/images
val: val/images

nc: 15
names:
  0: plane
  1: baseball-diamond
  2: bridge
  3: ground-track-field
  4: small-vehicle
  5: large-vehicle
  6: ship
  7: tennis-court
  8: basketball-court
  9: storage-tank
  10: soccer-ball-field
  11: roundabout
  12: harbor
  13: swimming-pool
  14: helicopter
```

### 训练命令

```bash
# 单 GPU 训练
python train.py --cfg yolov8s-strip.yaml --data dota.yaml --batch 16 --img 1024

# 多 GPU 训练
python -m torch.distributed.launch --nproc_per_node 4 train.py \
    --cfg yolov8s-strip.yaml \
    --data dota.yaml \
    --batch 64
```

### 推荐超参数

```yaml
epochs: 100-300
batch_size: 16-32
lr0: 0.001       # AdamW
weight_decay: 0.05
warmup_epochs: 3
img_size: 1024   # DOTA, 640 用于通用检测
```

---

## 性能对比

### DOTA-v1.0 测试结果

| Method | Backbone | mAP | Params | FLOPs |
|--------|----------|-----|--------|-------|
| Oriented R-CNN | ResNet-50 | 75.87 | 41.1M | 199G |
| LSKNet-S | LSKNet-S | 77.49 | 31.0M | 161G |
| PKINet-S | PKINet-S | 78.39 | 30.8M | 190G |
| **Strip R-CNN-S** | **StripNet-S** | **82.75** | **30.5M** | **159G** |

---

## 参考资料

1. Strip R-CNN: https://arxiv.org/abs/2501.03775
2. LSKNet: https://arxiv.org/abs/2303.04418
3. PKINet: https://arxiv.org/abs/2401.06839
4. YOLOv8: https://github.com/ultralytics/ultralytics

---

## 许可证

本实现仅用于学术研究和教育目的。

---

## 引用

```bibtex
@article{yuan2025strip,
  title={Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection},
  author={Yuan, Xinbin and Zheng, Zhaohui and Li, Yuxuan and Liu, Xialei and Liu, Li and Li, Xiang and Hou, Qibin and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2501.03775},
  year={2025}
}
```
