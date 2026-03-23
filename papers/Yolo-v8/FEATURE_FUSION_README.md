# Feature Fusion Modules (特征融合模块)

## 概述

基于两篇论文实现的特征融合模块：

### 1. Feature Calibration Fusion Module (FCF)

**论文**: Multi-Scale Direction-Aware Network for Infrared Small Target Detection

**Fig. 6**: The feature calibration fusion module.

**核心思想**:
- 融合低级特征（空间细节）和高级特征（语义信息）
- 通过上采样、拼接、卷积校准实现特征融合
- 类似 YOLOv8 的 FPN 结构，但添加了校准机制

### 2. Cross-Field Frequency Fusion Module (CFFF)

**论文**: Complementary Advantages Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising

**核心思想**:
- 基于频域的特征融合
- 使用 2D DFT/IDFT 进行频域交互
- CRM (Correlation Response Module) + DRM (Detail Response Module)

## 模块结构

### Feature Calibration Fusion (FCF)

```
┌─────────────────────────────────────────────────────────┐
│ Low-level ────────────────────────────────────────┐    │
│     H×W×C2                                        │    │
│                                                   ↓    │
│ High-level ─→ [Upsampling] ───────→ [Concat] ────────┼→ C
│     H/2×W/2×C1          2×            H×W×(C1+C2)      │
│                                                       ↓ │
│                                                [1×1 Conv]
│                                                       ↓ │
│                                                [BN+ReLU]
│                                                       ↓ │
│                                                [3×3 Conv]
│                                                       ↓ │
│                                                [BN+ReLU] ─→ [Add] ←── Low-level
│                                                    H×W×C1     │
│                                                       ↓      │
│                                                [1×1 Conv]    │
│                                                       ↓      │
│                                                [BN+ReLU]     │
│                                                       ↓      │
│                                                Output H×W×C2 │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Field Frequency Fusion (CFFF)

```
┌─────────────────────────────────────────────────────────┐
│ F_R (Reference)          F_N (NIR/Target)               │
│     │                         │    │                    │
│     ↓                         ↓    ↓                    │
│ [Conv]                     [Conv] [Conv]                │
│     │                         │    │                    │
│     ↓ Q                       ↓ K  ↓ V                  │
│ [2D DFT]                   [2D DFT]                     │
│     │ Dq                      │ Dk                      │
│     └───────────┬─────────────┘                         │
│                 ↓                                       │
│             [CRM] ← 频域相关性计算                       │
│                 ↓                                       │
│ ┌───────────────┴───────────────┐                       │
│ ↓                               ↓                       │
│ [2D IDFT] ← 逆变换回空间域    [DRM] ← 细节增强          │
│     │                           │                       │
│     ↓ F_R,N                     │                       │
│     └──────────── (+) ←─────────┘                       │
│                     ↓                                   │
│                 Output                                  │
└─────────────────────────────────────────────────────────┘
```

## 使用示例

### Feature Calibration Fusion

```python
import torch
from feature_fusion import FeatureCalibrationFusion

# 低级特征 (高空间分辨率)
low_feat = torch.randn(2, 256, 64, 64)

# 高级特征 (低空间分辨率)
high_feat = torch.randn(2, 512, 32, 32)

# 创建融合模块
fcf = FeatureCalibrationFusion(
    in_channels_low=256,   # C2
    in_channels_high=512,  # C1
)

# 融合
out = fcf(low_feat, high_feat)
print(out.shape)  # [2, 256, 64, 64]
```

### Cross-Field Frequency Fusion

```python
import torch
from feature_fusion import CrossFieldFrequencyFusion

# 参考特征和目标特征 (需要相同尺寸和通道)
feat_r = torch.randn(2, 256, 64, 64)
feat_n = torch.randn(2, 256, 64, 64)

# 创建融合模块
cfff = CrossFieldFrequencyFusion(
    in_channels=256,
    reduction=4,
)

# 融合
out = cfff(feat_r, feat_n)
print(out.shape)  # [2, 256, 64, 64]
```

### CFFF Simple (高效版本)

```python
from feature_fusion import CrossFieldFrequencyFusionSimple

# 简化版本，使用幅度谱近似，避免复数运算
cfff_simple = CrossFieldFrequencyFusionSimple(256)
out = cfff_simple(feat_r, feat_n)
```

### 组合使用 (FCF + CFFF)

```python
from feature_fusion import FCF_CFFF_Block

# 多尺度特征
p3 = torch.randn(2, 256, 64, 64)  # 高分辨率
p4 = torch.randn(2, 512, 32, 32)  # 中分辨率
p5 = torch.randn(2, 512, 16, 16)  # 低分辨率

# 创建组合模块
block = FCF_CFFF_Block(
    channels_p3=256,
    channels_p4=512,
    channels_p5=512,
    use_cfff=True,
)

# 融合
p3_out, p4_out, p5_out = block(p3, p4, p5)
```

## 在 YOLOv8 中使用

### 替换 YOLOv8 Neck 中的上采样

```python
# 原始 YOLOv8 Neck (简化)
class YOLOv8Neck(nn.Module):
    def __init__(self):
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(512, 256, 1)

    def forward(self, p3, p4, p5):
        p5_up = self.upsample(p5)
        p4 = torch.cat([p4, p5_up], dim=1)
        p4 = self.conv(p4)
        # ...
```

```python
# 使用 FCF 替换
from feature_fusion import FeatureCalibrationFusion

class ImprovedYOLOv8Neck(nn.Module):
    def __init__(self):
        # 使用 FCF 替代简单的上采样 + 拼接
        self.fcf = FeatureCalibrationFusion(
            in_channels_low=256,  # P3 通道
            in_channels_high=512, # P4 通道
        )

    def forward(self, p3, p4, p5):
        p4 = self.fcf(p4, p5)  # 更好的融合效果
        p3 = self.fcf(p3, p4)
        # ...
```

## 模块对比

| 模块 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| FCF | 2.89M | 简单高效，类似 FPN | 通用特征融合 |
| FCF V2 | 4.07M | 添加注意力机制 | 需要更强融合 |
| CFFF | 3.07M | 频域交互，理论更强 | 跨模态融合 |
| CFFF Simple | 1.88M | 高效近似 | 实时推理 |
| FCF+CFFF Block | 8.05M | 组合优势 | 高精度需求 |

## 核心组件

### CRM (Correlation Response Module)

在频域计算两个特征之间的相关性响应：

```python
crm = CorrelationResponseModule(channels=256, reduction=4)

# 频域特征 (复数)
freq_q = torch.fft.fft2(q, dim=(-2, -1))
freq_k = torch.fft.fft2(k, dim=(-2, -1))

# 计算相关性权重
weight = crm(freq_q, freq_k)  # [B, C, H, W]
```

### DRM (Detail Response Module)

增强高频细节信息的响应：

```python
drm = DetailResponseModule(channels=256)

# 增强细节
enhanced = drm(v)  # v * weight
```

## 参考文献

### FCF
```bibtex
@article{mda_net_2024,
  title={Multi-Scale Direction-Aware Network for Infrared Small Target Detection},
  author={Unknown},
  journal={Unknown},
  year={2024}
}
```

### CFFF
```bibtex
@article{cafe_2024,
  title={Complementary Advantages Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising},
  author={Unknown},
  journal={Unknown},
  year={2024}
}
```

## 文件结构

```
Yolo-v8/
├── feature_fusion.py       # 特征融合模块实现
├── FEATURE_FUSION_README.md # 本文档
├── strip_conv.py           # Strip 卷积模块
├── connectivity_attention.py # 连通性注意力
└── ...
```

## 测试

运行测试:
```bash
cd papers/Yolo-v8
python feature_fusion.py
```

预期输出:
```
============================================================
Testing Feature Fusion Modules
============================================================

[Feature Calibration Fusion]
  Low: torch.Size([2, 256, 64, 64]), High: torch.Size([2, 512, 32, 32])
  Output: torch.Size([2, 256, 64, 64])

[Feature Calibration Fusion V2]
  Output: torch.Size([2, 256, 64, 64])

[Cross-Field Frequency Fusion]
  Ref: torch.Size([2, 256, 64, 64]), Target: torch.Size([2, 256, 64, 64])
  Output: torch.Size([2, 256, 64, 64])
...
[All tests passed!]
```
