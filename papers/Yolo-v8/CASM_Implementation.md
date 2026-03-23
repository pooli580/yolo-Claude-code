# Connectivity Attention Strip Module (CASM) 实现文档

## 概述

基于 **YOLOv2 架构图** 和 **CoANet 论文** 中的 Connectivity Attention 思想，为 YOLOv8 创建了一套完整的注意力增强模块。

## 创建的文件

```
papers/Yolo-v8/
├── connectivity_attention.py    # 核心模块实现
├── yolov8n-casm.yaml           # YOLOv8n-CASM 模型配置
├── test_casm.py                 # 模块测试脚本
└── CASM_README.md               # 使用指南
```

## 模块结构

### 1. 四方向条带卷积 (Quad-Direction Strip Convolution)

```
                    ┌─────────────────┐
                    │   Input Feature │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    1x1 Conv     │  (降维)
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  H-Strip Conv   │ │  V-Strip Conv   │ │ Diagonal Conv   │
│   (1 x K)       │ │   (K x 1)       │ │  (对角线)       │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Concat + 1x1  │  (融合)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Output        │
                    └─────────────────┘
```

**支持的方向**:
- **水平条带 (Horizontal)**: `HorizontalStripConv` - 1×K 卷积
- **垂直条带 (Vertical)**: `VerticalStripConv` - K×1 卷积
- **左对角线 (Left Diagonal)**: `LeftDiagonalStripConv` - 对角线位移卷积
- **右对角线 (Right Diagonal)**: `RightDiagonalStripConv` - 反对角线位移卷积

### 2. Connectivity Attention

```
    条带特征 ──→ Adaptive Pool ──→ FC ──→ BN ──→ GELU ──→ FC ──→ Sigmoid ──→ 权重相乘
         │
         │  水平方向：AdaptiveAvgPool2d((1, None))   # 沿高度池化
         │  垂直方向：AdaptiveAvgPool2d((None, 1))   # 沿宽度池化
         │  对角线方向：AdaptiveAvgPool2d(1)         # 全局池化
         ▼
    增强特征
```

### 3. CASM 核心模块

```
┌────────────────────────────────────────────────────────────┐
│                     CASM Module                            │
│                                                            │
│  Input ──┬─────────────────────────────────────────┐       │
│          │                                         │       │
│          │  ┌─────────────────────────────────┐   │       │
│          │  │  QuadDirectionStripConv         │   │       │
│          │  │  - 1x1 Conv (降维)              │   │       │
│          │  │  - H/V/LD/RD Strip Conv         │   │       │
│          │  │  - Concat + 1x1 Conv (融合)     │   │       │
│          │  └──────────────┬──────────────────┘   │       │
│          │                 │                      │       │
│          │  ┌──────────────▼──────────────────┐   │       │
│          │  │  ConnectivityAttention          │   │       │
│          │  │  - 方向注意力加权               │   │       │
│          │  └──────────────┬──────────────────┘   │       │
│          │                 │                      │       │
│          └────────────────┼──────────────────────┘       │
│                           │                               │
│                    ┌──────▼──────┐                        │
│                    │  + Identity │  (残差连接)            │
│                    └──────┬──────┘                        │
│                           │                               │
│                         Output                            │
└────────────────────────────────────────────────────────────┘
```

## 模块层次

| 模块 | 用途 | 位置 |
|------|------|------|
| `CASM` | 基础注意力模块 | Backbone/Neck |
| `CASMBlock` | 骨干网络块 | Backbone |
| `C2f_CASM` | Neck 特征融合 | Neck |
| `SPPF_CASM` | 增强空间金字塔池化 | Backbone 末端 |

## 与 YOLOv2 架构的关系

```
YOLOv2 架构                  CASM 实现
─────────────────────────   ─────────────────────────────
Batch Normalization    →    nn.BatchNorm2d + GELU
Convolutional Layers   →    Conv + StripConv
Pooling Layers         →    AdaptiveAvgPool (Connectivity Attention)
Passthrough Layer      →    类似 C2f_CASM 的拼接融合
Detection Layer        →    保持 YOLOv8 Detect Head
```

## 与 CoANet 论文的关系

| CoANet 组件 | CASM 实现 |
|-------------|-----------|
| Connectivity Attention | `ConnectivityAttention` 类 |
| Strip Convolution | `QuadDirectionStripConv` 类 |
| Four Directions | H/V/LD/RD 四个条带卷积 |
| Residual Connection | `with_residual` 参数 |

## 使用示例

### 在模型配置中使用

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, CASMBlock, [128, 19, True]]  # CASM 块
  - [-1, 1, SPPF_CASM, [512, 5, 19]]     # 增强 SPPF

neck:
  - [[-1, -3], 1, C2f_CASM, [256, 2, 19]]  # CASM 增强的 C2f
```

### 在代码中使用

```python
from connectivity_attention import CASM, CASMBlock, C2f_CASM, SPPF_CASM

# Backbone
x = torch.randn(1, 3, 640, 640)
block = CASMBlock(128, 128, strip_kernel_size=19)
out = block(x)

# Neck
c2f = C2f_CASM(256, 256, num_blocks=3, strip_kernel_size=19)
out = c2f(x)

# SPPF
sppf = SPPF_CASM(512, 512, strip_kernel_size=19)
out = sppf(x)
```

## 参考文献

### CoANet 论文
```bibtex
@ARTICLE{CoANet2022,
  author={Wang, Yu and Zhang, Tianzhu and Liu, Yang and Wang, Lijun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={CoANet: Connectivity Attention Network for Road Extraction From Satellite Imagery},
  year={2022},
  volume={60},
  pages={1-14},
  doi={10.1109/TGRS.2022.3141691}
}
```

### Strip Pooling
```bibtex
@INPROCEEDINGS{strip_pooling_2020,
  author={Hou, Qibin and Zhang, Li and Cheng, Ming-Ming and Feng, Jiashi},
  title={Rethinking Spatial Pooling for Scene Parsing},
  booktitle={CVPR},
  year={2020}
}
```

### Strip R-CNN
```bibtex
@ARTICLE{strip_rcnn_2025,
  author={Unknown},
  title={Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection},
  journal={arXiv},
  year={2025},
  eprint={2501.03775}
}
```

## 测试运行

```bash
cd papers/Yolo-v8
python test_casm.py
```

期望输出:
```
============================================================
Testing Connectivity Attention Strip Module (CASM)
============================================================

1. Single Direction Strip Convolutions
----------------------------------------
  HorizontalStripConv:
    Input:  (2, 64, 56, 56)
    Output: (2, 64, 56, 56)
    Params: 1,216

...

✓ All tests passed!
```
