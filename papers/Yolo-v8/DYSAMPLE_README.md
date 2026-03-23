# Dynamic Upsampling Module (DySample + DUPS)

## 概述

基于两篇论文实现的动态上采样模块：

1. **DySample**: Learning to Upsample by Learning to Sample (ICCV 2023)
   - 论文：https://arxiv.org/abs/2312.15669
   - GitHub: https://github.com/tiny-smart/dysample

2. **DUPS**: Dynamic Upsampling for Efficient Semantic Segmentation
   - 论文：DUPS: DYNAMIC UPSAMPLING FOR EFFICIENT SEMANTIC SEGMENTATION

## 核心思想

### DySample

DySample 是一种超轻量动态上采样器，核心创新点：

1. **从动态卷积转向点采样**
   - 传统动态上采样器（CARAFE, FADE, SAPA）使用动态卷积，计算开销大
   - DySample 使用 PyTorch 内置的 `grid_sample` 函数，无需自定义 CUDA

2. **关键设计**
   - **Initial Sampling Position**: 双线性初始化而非最近邻初始化
   - **Offset Scope**: 静态 (0.25) / 动态范围因子约束偏移量
   - **Grouping**: 通道分组减少参数量

3. **优势**
   - 参数量：仅 2K-4K (CARAFE 的 3%)
   - FLOPs: 极低
   - 推理延迟：接近双线性插值 (6.2ms vs 1.6ms)

### DUPS (框图所示结构)

DUPS 采用双路径动态上采样设计：

```
框图 (a) - 2D Conv 版本:
输入 → [2D Conv → UP] ─┐
     → [2D Conv → UP] ─┼→ Concat → 2D Conv → 输出
     → [2D Conv    ] ─┘

框图 (b) - 1D Conv 版本:
输入 → [1D Conv → UP] ─┐
     → [1D Conv → UP] ─┴→ Concat → 2D Conv → 输出
```

## 模块结构

### DySample

```python
DySample(
    in_channels=64,
    scale_factor=2,
    groups=4,
    mode='static',      # 'static' | 'dynamic'
    init_mode='bilinear' # 'bilinear' | 'nearest'
)
```

**参数说明**:
- `in_channels`: 输入通道数
- `scale_factor`: 上采样倍数
- `groups`: 特征分组数（减少参数量）
- `mode`: 偏移生成模式
  - `static`: 单层线性投影 + 固定 0.25 范围因子
  - `dynamic`: 双层投影 + sigmoid 动态范围

### DUPSBlock2D (对应框图 a)

```python
DUPSBlock2D(
    in_channels=64,
    out_channels=128,
    scale_factor=2,
    mid_channels=None  # 默认 in_channels // 3
)
```

结构:
- 三个并行 2D Conv 分支
- 两个分支后接 DySample 上采样
- 拼接后通过 2D Conv 融合

### DUPSBlock1D (对应框图 b)

```python
DUPSBlock1D(
    in_channels=64,
    out_channels=128,
    scale_factor=2,
    kernel_size=7
)
```

结构:
- 水平条带卷积 (1×K)
- 垂直条带卷积 (K×1)
- 两路都上采样后拼接融合

### DynamicUpsample (统一接口)

```python
DynamicUpsample(
    in_channels=64,
    out_channels=128,
    scale_factor=2,
    method='hybrid'  # 'dysample' | 'dups-2d' | 'dups-1d' | 'hybrid'
)
```

### YOLOUpsample (YOLOv8 兼容层)

```python
YOLOUpsample(
    in_channels=64,
    out_channels=128,
    scale_factor=2,
    use_dynamic=True,   # False 退化到双线性上采样
    method='hybrid'
)
```

## 使用示例

### 基础使用

```python
import torch
from dysample_upsample import DySample, DynamicUpsample, YOLOUpsample

# 创建输入
x = torch.randn(1, 64, 32, 32)

# 1. DySample (最轻量)
upsampler = DySample(64, scale_factor=2)
out = upsampler(x)  # (1, 64, 64, 64)

# 2. DynamicUpsample (多方法支持)
up = DynamicUpsample(64, 128, scale_factor=2, method='dysample')
out = up(x)  # (1, 128, 64, 64)

# 3. YOLOUpsample (替换 YOLOv8 的 nn.Upsample)
yolo_up = YOLOUpsample(64, 128, scale_factor=2)
out = yolo_up(x)  # (1, 128, 64, 64)
```

### 在 YOLOv8 中使用

```python
# 修改 YOLOv8 的 neck 部分
from dysample_upsample import YOLOUpsample

class ModifiedYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # 替换传统上采样
        self.upsample1 = YOLOUpsample(512, 256, scale_factor=2, method='hybrid')
        self.upsample2 = YOLOUpsample(256, 128, scale_factor=2, method='dysample')

    def forward(self, x):
        # ... backbone 特征提取
        # 上采样融合
        p4 = self.upsample1(p3)
        p4 = torch.cat([p4, backbone_feat4], dim=1)
        # ...
```

## 参数量对比

| 模块 | 参数量 | 特点 |
|------|--------|------|
| DySample (static) | 2,080 | 最轻量，推荐 |
| DySample (dynamic) | 4,128 | 略高但更灵活 |
| DUPS 1D | 104,896 | 条带卷积，适合细长结构 |
| DUPS 2D | 110,654 | 三路并行，表达能力强 |
| Traditional + Conv | 73,856 | 基线方法 |
| Bilinear (no conv) | 0 | 无参数，但效果有限 |

## 推荐配置

### 目标检测 (YOLOv8)

```python
# Neck 部分上采样
YOLOUpsample(channels, channels, scale_factor=2, method='dysample')
```

### 语义分割

```python
# Decoder 部分上采样
DynamicUpsample(in_ch, out_ch, scale_factor=2, method='hybrid')
```

### 遥感图像 (细长结构)

```python
# 使用条带卷积增强
DynamicUpsample(in_ch, out_ch, scale_factor=2, method='dups-1d')
```

## 参考文献

### DySample
```bibtex
@inproceedings{liu2023dysample,
  title={Learning to Upsample by Learning to Sample},
  author={Liu, Wenze and Lu, Hao and Fu, Hongtao and Cao, Zhiguo},
  booktitle={Proceedings of IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

### DUPS
```bibtex
@article{dups2023,
  title={DUPS: Dynamic Upsampling for Efficient Semantic Segmentation},
  author={Unknown},
  journal={Under Review},
  year={2023}
}
```

### 相关方法
- CARAFE: Context-Aware Reassembly of Features (ICCV 2019)
- FADE: Fusing Assets of Decoder and Encoder (ECCV 2022)
- SAPA: Similarity-Aware Point Affiliation (NeurIPS 2022)

## 测试

运行测试:
```bash
cd papers/Yolo-v8
python dysample_upsample.py
```

预期输出:
```
============================================================
Testing Dynamic Upsampling Modules
============================================================

[DySample]
  Static: torch.Size([2, 64, 32, 32]) -> torch.Size([2, 64, 64, 64])
  Dynamic: torch.Size([2, 64, 32, 32]) -> torch.Size([2, 64, 64, 64])

[DUPS Block 2D]
  torch.Size([2, 64, 32, 32]) -> torch.Size([2, 128, 64, 64])

[DUPS Block 1D]
  torch.Size([2, 64, 32, 32]) -> torch.Size([2, 128, 64, 64])
...
[All tests passed!]
```

## 文件结构

```
Yolo-v8/
├── dysample_upsample.py    # 动态上采样模块实现
├── DYSAMPLE_README.md      # 本文档
├── strip_conv.py           # Strip R-CNN 条带卷积
├── connectivity_attention.py # CoANet 连通性注意力
└── ...
```
