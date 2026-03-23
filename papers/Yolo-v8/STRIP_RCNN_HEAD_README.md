# Strip R-CNN Detection Heads

## 概述

基于 **Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection** 论文实现的检测头模块。

**论文信息:**
- arXiv: https://arxiv.org/abs/2501.03775
- GitHub: https://github.com/HVision-NKU/Strip-R-CNN

## 框图结构

本模块严格实现论文中的两种检测头结构：

### Oriented R-CNN Head (基线方法)

```
┌─────────────────────────────────────────────────────────┐
│  RoI ──→ [FC] ──→ [FC] ──┬──→ [FC] ──→ Classification │
│                          └──→ [FC] ──→ Localization + Angle │
└─────────────────────────────────────────────────────────┘
```

### Strip Head (核心创新)

```
┌─────────────────────────────────────────────────────────┐
│                 ┌──→ [FC] → [FC] ────→ [FC] → Class  │
│                                    └──→ [FC] → Angle  │
│  RoI → [FC] → [FC] ─                                  │
│                     └──→ [Conv] → [Strip Module] → [FC] → Loc │
└─────────────────────────────────────────────────────────┘
```

## 核心差异

| 特性 | Oriented R-CNN Head | Strip Head |
|------|---------------------|------------|
| 结构 | 共享 FC 层 | 解耦三分支 |
| 分类路径 | FC→FC→FC | FC→FC→FC |
| 角度路径 | FC→FC→FC | 从 fc2 分叉→FC |
| 定位路径 | FC→FC→FC | Conv→Strip Module→FC |
| 参数量 | ~14M | ~15M |
| 适用场景 | 通用旋转检测 | 高长宽比物体 |

## 模块说明

### OrientedRCNNHead

基线旋转目标检测头。

```python
OrientedRCNNHead(
    in_channels=256,      # 输入通道数
    num_classes=80,       # 分类类别数
    hidden_dim=1024,      # FC 隐藏层维度
    roi_size=7,           # RoI Align 输出尺寸
)
```

**输入/输出:**
- Input: `[B, C, 7, 7]` RoI 特征
- Output:
  - cls_scores: `[B, num_classes]`
  - loc_angle: `[B, 5]` (x, y, w, h, angle)

### StripHead

核心 Strip 检测头，定位分支使用 Strip Module 捕获长距离空间依赖。

```python
StripHead(
    in_channels=256,        # 输入通道数
    num_classes=80,         # 分类类别数
    strip_kernel_size=19,   # 条带卷积核大小
    hidden_dim=1024,        # FC 隐藏层维度
    roi_size=7,             # RoI Align 输出尺寸
)
```

**输入/输出:**
- Input: `[B, C, 7, 7]` RoI 特征
- Output:
  - cls_scores: `[B, num_classes]`
  - bboxes: `[B, 4]` (cx, cy, w, h)
  - angles: `[B, 1]`

### StripHeadV2

改进版 Strip Head，优化特征流动。

### StripDetectHead

多尺度检测头，用于 FPN 特征金字塔。

```python
StripDetectHead(
    in_channels=[256, 512, 512],  # P3, P4, P5 通道数
    num_classes=80,
    strip_kernel_size=19,
    hidden_dim=1024,
)
```

## 使用示例

### 基础使用

```python
import torch
from strip_rcnn_head import StripHead, OrientedRCNNHead

# RoI 特征
x = torch.randn(4, 256, 7, 7)

# Oriented R-CNN Head
oriented_head = OrientedRCNNHead(256, num_classes=80)
cls_scores, loc_angle = oriented_head(x)

# Strip Head
strip_head = StripHead(256, num_classes=80, strip_kernel_size=19)
cls_scores, bboxes, angles = strip_head(x)
```

### 在旋转检测中使用

```python
from strip_rcnn_head import StripDetectHead

# FPN 多尺度特征
features = [
    torch.randn(4, 256, 28, 28),  # P3
    torch.randn(4, 512, 14, 14),  # P4
    torch.randn(4, 512, 7, 7),    # P5
]

# 多尺度检测头
detect_head = StripDetectHead(
    in_channels=[256, 512, 512],
    num_classes=80,
    strip_kernel_size=19,
)

cls_list, bboxes_list, angles_list = detect_head(features)
```

### 与 RoI Align 结合

```python
import torch
from torch import nn
from torchvision.ops import roi_align

# 假设 backbone 特征和 RoI
backbone_feat = torch.randn(1, 256, 64, 64)
rois = torch.tensor([
    [0, 10, 10, 50, 50],  # (batch_idx, x1, y1, x2, y2)
    [0, 20, 20, 60, 60],
])

# RoI Align
roi_features = roi_align(
    backbone_feat, rois,
    output_size=(7, 7),
    spatial_scale=1.0/16.0
)

# Strip Head 预测
strip_head = StripHead(256, num_classes=80)
cls_scores, bboxes, angles = strip_head(roi_features)
```

## 参数量对比

| 模块 | 参数量 | 特点 |
|------|--------|------|
| Oriented R-CNN Head | 13.98M | 基线方法 |
| Strip Head | 14.98M | 推荐，解耦设计 |
| Strip Head V2 | 16.77M | 改进版本 |

## Strip Module 作用

Strip Module 是定位分支的核心组件：

1. **大条带卷积**: 19x1 和 1x19 正交卷积
2. **长距离依赖**: 捕获水平和垂直方向的长距离上下文
3. **高长宽比物体**: 对船舶、飞机、桥梁等细长结构有更好的定位能力

```
Strip Module 内部结构:
输入 → [5x5 Conv] → [H-Strip 1x19] → [V-Strip 19x1] → [1x1 Conv] → FFN → 输出
       ↓                                                      ↑
       └────────────────── 残差连接 ──────────────────────────┘
```

## 适用场景

### 推荐使用 Strip Head
- 遥感图像目标检测
- 高长宽比物体（船舶、飞机、桥梁、道路）
- 旋转目标检测 (Oriented Object Detection)

### 推荐使用 Oriented R-CNN Head
- 通用旋转检测基线
- 参数量敏感场景

## 文件结构

```
Yolo-v8/
├── strip_rcnn_head.py      # Strip R-CNN 检测头实现
├── strip_conv.py           # Strip 卷积模块
├── strip_head.py           # 旧版 Strip Head (参考)
├── strip_net.py            # StripNet Backbone
└── STRIP_RCNN_HEAD_README.md  # 本文档
```

## 测试

运行测试:
```bash
cd papers/Yolo-v8
python strip_rcnn_head.py
```

预期输出:
```
============================================================
Testing Strip R-CNN Detection Heads
============================================================

[Oriented R-CNN Head]
  Input: torch.Size([4, 256, 7, 7])
  cls_scores: torch.Size([4, 80])
  loc_angle: torch.Size([4, 5])

[Strip Head]
  Input: torch.Size([4, 256, 7, 7])
  cls_scores: torch.Size([4, 80])
  bboxes: torch.Size([4, 4])
  angles: torch.Size([4, 1])
...
[All tests passed!]
```

## 参考文献

```bibtex
@article{strip_rcnn_2025,
  title={Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection},
  author={Unknown},
  journal={arXiv preprint},
  year={2025}
}
```

相关方法:
- Oriented R-CNN: 旋转目标检测基线方法
- CARAFE: Content-Aware Reassembly of Features
- LSKNet: Large Selective Kernel Network
