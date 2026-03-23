# Connectivity Attention Strip Module (CASM) 使用指南

## 模块结构

```
CASM - Connectivity Attention Strip Module
├── 四方向条带卷积 (Quad-Direction Strip Conv)
│   ├── 水平条带 (Horizontal Strip)
│   ├── 垂直条带 (Vertical Strip)
│   ├── 左对角线条带 (Left Diagonal)
│   └── 右对角线条带 (Right Diagonal)
├── Connectivity Attention
└── 残差连接
```

## 在 YOLOv8 中使用

### 1. 修改模型配置文件

```yaml
# yolov8n-casm.yaml
backbone:
  - [[-1, 1, Conv, [64, 3, 2]]]
  - [[-1, 1, Conv, [128, 3, 2]]]
  - [[-1, 3, CASMBlock, [128, 19, True]]]  # 使用 CASMBlock
  - [[-1, 1, Conv, [256, 3, 2]]]
  - [[-1, 3, CASMBlock, [256, 19, True]]]
  - [[-1, 1, Conv, [512, 3, 2]]]
  - [[-1, 3, CASMBlock, [512, 19, True]]]
  - [[-1, 1, Conv, [512, 3, 2]]]
  - [[-1, 3, CASMBlock, [512, 19, True]]]
  - [[-1, 1, SPPF_CASM, [512, 5, 19]]]  # 使用 SPPF_CASM

neck:
  - [[-1, 1, Conv, [512, 1, 1]]]
  - [[-2, 1, nn.Upsample, [None, 2, 'nearest']]]
  - [[[-1, -3], 1, C2f_CASM, [512, 3, 19]]]  # 使用 C2f_CASM
  - [[-1, 1, Conv, [256, 1, 1]]]
  - [[-2, 1, nn.Upsample, [None, 2, 'nearest']]]
  - [[[-1, -3], 1, C2f_CASM, [256, 2, 19]]]
  - [[-1, 1, Conv, [128, 3, 2]]]
  - [[[-1, -4], 1, C2f_CASM, [256, 2, 19]]]
  - [[-1, 1, Conv, [256, 3, 2]]]
  - [[[-1, -4], 1, C2f_CASM, [512, 2, 19]]]

head:
  - [[-1, 1, Detect, [nc]]]
```

### 2. 在代码中使用

```python
from connectivity_attention import CASM, CASMBlock, C2f_CASM, SPPF_CASM

# 单个 CASM 模块
x = torch.randn(4, 64, 56, 56)
casm = CASM(64, 64, strip_kernel_size=19)
out = casm(x)

# CASM Block (用于 Backbone)
block = CASMBlock(128, 128, strip_kernel_size=19, shortcut=True)
out = block(x)

# C2f_CASM (用于 Neck)
c2f = C2f_CASM(256, 256, num_blocks=3, strip_kernel_size=19)
out = c2f(x)

# SPPF_CASM (用于 Backbone 末端)
sppf = SPPF_CASM(512, 512, strip_kernel_size=19)
out = sppf(x)
```

### 3. 训练命令

```bash
# 使用 CASM 增强的 YOLOv8
yolo train model=yolov8n-casm.yaml data=coco.yaml epochs=100 imgsz=640

# 或者在代码中
from ultralytics import YOLO
model = YOLO('yolov8n-casm.yaml')
model.train(data='coco.yaml', epochs=100, imgsz=640)
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `strip_kernel_size` | 条带卷积核大小 (奇数) | 19 |
| `expansion` | 通道扩展/压缩率 | 0.25 (CASM), 0.5 (CASMBlock) |
| `shortcut` | 是否使用残差连接 | True |
| `num_blocks` | CASM 块数量 (C2f_CASM) | 3 |

## 适用场景

✅ **推荐使用**:
- 遥感图像目标检测（道路、河流、桥梁）
- 细长结构物体检测（电线、管道、栏杆）
- 高长宽比物体检测（船舶、飞机、车辆）

❌ **不推荐**:
- 小目标密集场景（条带卷积可能丢失局部细节）
- 实时性要求极高的场景（计算开销较大）

## 参考文献

```bibtex
@ARTICLE{CoANet2022,
  author={Wang, Yu and Zhang, Tianzhu and Liu, Yang and Wang, Lijun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={CoANet: Connectivity Attention Network for Road Extraction From Satellite Imagery},
  year={2022},
  volume={60},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2022.3141691}
}

@article{strip_pooling_2020,
  title={Rethinking Spatial Pooling for Scene Parsing},
  author={Hou, Qibin and Zhang, Li and Cheng, Ming-Ming and Feng, Jiashi},
  journal={CVPR},
  year={2020}
}
```
