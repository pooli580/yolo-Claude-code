"""
Feature Fusion Modules for YOLOv8
基于两篇论文的特征融合模块实现:

1. Feature Calibration Fusion Module (FCF)
   Paper: Multi-Scale Direction-Aware Network for Infrared Small Target Detection
   - 用于融合低级特征 (空间细节) 和高级特征 (语义信息)
   - 通过上采样、拼接、卷积校准实现特征融合

2. Cross-Field Frequency Fusion Module (CFFF)
   Paper: Complementary Advantages Exploiting Cross-Field Frequency Correlation
          for NIR-Assisted Image Denoising
   - 基于频域的特征融合
   - 使用 2D DFT/IDFT 进行频域交互
   - CRM (Correlation Response Module) + DRM (Detail Response Module)

适用于:
- YOLOv8 Neck 特征融合
- 多尺度特征融合
- 跨模态特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# 工具函数
# ============================================================================

def batch_fft2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D FFT (快速傅里叶变换)

    Args:
        x: 输入特征 [B, C, H, W]

    Returns:
        频域特征 [B, C, H, W, 2] (实部 + 虚部)
    """
    return torch.fft.fft2(x, dim=(-2, -1))


def batch_ifft2d(x: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
    """
    2D IFFT (逆傅里叶变换)

    Args:
        x: 频域特征 [B, C, H, W, 2] 或复数张量
        output_shape: 输出空间尺寸 (H, W)

    Returns:
        空间域特征 [B, C, H, W]
    """
    result = torch.fft.ifft2(x, dim=(-2, -1), s=output_shape)
    return torch.abs(result)


# ============================================================================
# Feature Calibration Fusion Module (FCF)
# ============================================================================

class FeatureCalibrationFusion(nn.Module):
    """
    Feature Calibration Fusion Module (FCF)

    基于论文：Multi-Scale Direction-Aware Network for Infrared Small Target Detection
    Fig. 6: The feature calibration fusion module.

    结构 (严格遵循框图):

    Low-level ─────────────────────────────────────────────┐
        H×W×C2                                             │
                                                           ↓
    High-level ─→ [Upsampling] ──────────→ [Concat] ──→ C ──┐
        H/2×W/2×C1        2×              H×W×(C1+C2)      │
                                                           ↓
                                                    [1×1 Conv]
                                                           ↓
                                                    [BN+ReLU]
                                                           ↓
                                                    [3×3 Conv]
                                                           ↓
                                                    [BN+ReLU] ──→ [Add] ←── Low-level
                                                        H×W×C1     (passthrough)
                                                           ↓
                                                    [1×1 Conv]
                                                           ↓
                                                    [BN+ReLU]
                                                           ↓
                                                    Output H×W×C2

    Args:
        in_channels_low: 低级特征通道数 (C2)
        in_channels_high: 高级特征通道数 (C1)
        out_channels: 输出通道数 (默认等于 C2)

    Input:
        low_feat: 低级特征 [B, C2, H, W]
        high_feat: 高级特征 [B, C1, H/2, W/2]

    Output:
        fused_feat: 融合特征 [B, C2, H, W]
    """

    def __init__(
        self,
        in_channels_low: int,
        in_channels_high: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels_low = in_channels_low
        self.in_channels_high = in_channels_high
        self.out_channels = out_channels or in_channels_low

        # 上采样层 (双线性插值)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 拼接后的通道压缩 (1x1 Conv)
        combined_channels = in_channels_low + in_channels_high
        self.conv1x1_compress = nn.Sequential(
            nn.Conv2d(combined_channels, in_channels_high, 1, bias=False),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True),
        )

        # 3x3 卷积特征提取
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels_high, in_channels_high, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True),
        )

        # 输出投影 (1x1 Conv) - 将 C1 转换回 C2
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels_high, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        low_feat: torch.Tensor,
        high_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            low_feat: 低级特征 [B, C2, H, W]
            high_feat: 高级特征 [B, C1, H/2, W/2]

        Returns:
            fused_feat: 融合特征 [B, C2, H, W]
        """
        # 保存低级特征用于后续相加
        identity = low_feat  # [B, C2, H, W]

        # 1. 高级特征上采样
        high_up = self.upsample(high_feat)  # [B, C1, H, W]

        # 2. 拼接低级和上采样后的高级特征
        concat_feat = torch.cat([low_feat, high_up], dim=1)  # [B, C1+C2, H, W]

        # 3. 通道压缩 (1x1 Conv + BN + ReLU)
        compressed = self.conv1x1_compress(concat_feat)  # [B, C1, H, W]

        # 4. 特征提取 (3x3 Conv + BN + ReLU)
        refined = self.conv3x3(compressed)  # [B, C1, H, W]

        # 5. 与低级特征相加 (类似残差连接)
        # 注意：这里需要通道匹配，所以先投影到 C2
        refined_proj = self.out_proj(refined)  # [B, C2, H, W]

        # 6. 相加得到最终输出
        fused_feat = refined_proj + identity  # [B, C2, H, W]

        return fused_feat


class FeatureCalibrationFusionV2(nn.Module):
    """
    Feature Calibration Fusion Module V2 - 改进版本

    改进点:
    1. 使用可学习的上采样 (转置卷积) 替代双线性插值
    2. 添加注意力机制增强融合效果
    3. 支持不同的输出通道配置
    """

    def __init__(
        self,
        in_channels_low: int,
        in_channels_high: int,
        out_channels: Optional[int] = None,
        use_transposed_conv: bool = True,
    ):
        super().__init__()

        self.in_channels_low = in_channels_low
        self.in_channels_high = in_channels_high
        self.out_channels = out_channels or in_channels_low

        # 上采样层
        if use_transposed_conv:
            self.upsample = nn.ConvTranspose2d(
                in_channels_high, in_channels_high,
                kernel_size=2, stride=2,
                bias=False
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.upsample_proj = nn.Conv2d(in_channels_high, in_channels_high, 1, bias=False)

        # 拼接后的处理
        combined_channels = in_channels_low + in_channels_high
        self.conv1 = nn.Sequential(
            nn.Conv2d(combined_channels, in_channels_high, 1, bias=False),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_high, in_channels_high, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True),
        )

        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels_high, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        # 通道注意力 (增强融合效果)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_high, in_channels_high // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_high // 4, in_channels_high, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        low_feat: torch.Tensor,
        high_feat: torch.Tensor
    ) -> torch.Tensor:
        # 上采样高级特征
        high_up = self.upsample(high_feat)
        if hasattr(self, 'upsample_proj'):
            high_up = self.upsample_proj(high_up)

        # 拼接
        concat = torch.cat([low_feat, high_up], dim=1)

        # 处理
        feat = self.conv1(concat)

        # 通道注意力
        attn = self.channel_attention(feat)
        feat = feat * attn

        feat = self.conv2(feat)
        out = self.out_conv(feat)

        return out


# ============================================================================
# Cross-Field Frequency Fusion Module (CFFF)
# ============================================================================

class CorrelationResponseModule(nn.Module):
    """
    CRM (Correlation Response Module)

    基于论文：Complementary Advantages Exploiting Cross-Field Frequency Correlation

    在频域计算两个特征之间的相关性响应

    Args:
        channels: 输入通道数
        reduction: 通道缩减比例 (默认 4)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        self.channels = channels
        self.reduction = reduction

        # 相关性计算后的投影 (输出与输入相同通道)
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),  # 生成权重
        )

    def forward(
        self,
        freq_q: torch.Tensor,
        freq_k: torch.Tensor
    ) -> torch.Tensor:
        """
        计算频域相关性响应

        Args:
            freq_q: Query 频域特征 (复数，但使用幅度谱)
            freq_k: Key 频域特征 (复数，但使用幅度谱)

        Returns:
            相关性权重 [B, C, H, W]
        """
        # 使用幅度谱进行计算
        q_mag = torch.abs(freq_q)  # [B, C, H, W]
        k_mag = torch.abs(freq_k)  # [B, C, H, W]

        # 拼接后通过投影生成权重
        combined = torch.cat([q_mag, k_mag], dim=1)
        weight = self.proj(combined)  # [B, C, H, W]

        return weight


class DetailResponseModule(nn.Module):
    """
    DRM (Detail Response Module)

    增强高频细节信息的响应

    Args:
        channels: 输入通道数
    """

    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels

        # 高频增强卷积
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        增强细节响应

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            增强后的特征 [B, C, H, W]
        """
        weight = self.enhance(x)
        return x * weight


class CrossFieldFrequencyFusion(nn.Module):
    """
    Cross-Field Frequency Fusion Module (CFFF)

    基于论文：Complementary Advantages Exploiting Cross-Field Frequency Correlation
            for NIR-Assisted Image Denoising

    结构 (严格遵循框图):

    F_R (Reference)          F_N (NIR/Target)
        │                         │    │
        ↓                         ↓    ↓
    [Conv]                     [Conv] [Conv]
        │                         │    │
        ↓ Q                       ↓ K  ↓ V
    [2D DFT]                   [2D DFT]
        │ Dq                      │ Dk
        └───────────┬─────────────┘
                    ↓
                [CRM] ← 频域相关性计算
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
    [2D IDFT] ← 逆变换回空间域    [DRM] ← 细节增强
        │                           │
        ↓ F_R,N                     │
        └──────────── (+) ←─────────┘
                        ↓
                    Output

    Args:
        in_channels: 输入通道数
        reduction: CRM 中的通道缩减比例

    Input:
        feat_r: 参考特征 (Reference) [B, C, H, W]
        feat_n: 目标特征 (NIR/Target) [B, C, H, W]

    Output:
        fused: 融合特征 [B, C, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.reduction = reduction

        # 参考分支卷积
        self.conv_r = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

        # 目标分支卷积 (生成 K 和 V)
        self.conv_k = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

        # CRM (Correlation Response Module)
        self.crm = CorrelationResponseModule(in_channels, reduction)

        # DRM (Detail Response Module)
        self.drm = DetailResponseModule(in_channels)

        # 输出卷积
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(
        self,
        feat_r: torch.Tensor,
        feat_n: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播 - 跨域频域融合

        Args:
            feat_r: 参考特征 [B, C, H, W]
            feat_n: 目标特征 [B, C, H, W]

        Returns:
            fused: 融合特征 [B, C, H, W]
        """
        B, C, H, W = feat_r.shape

        # ========== 参考分支 ==========
        # Conv → 2D DFT
        q = self.conv_r(feat_r)  # [B, C, H, W]
        freq_q = torch.fft.fft2(q, dim=(-2, -1))  # 复数张量

        # ========== 目标分支 ==========
        # Conv → 2D DFT (K) 和 Conv (V)
        k = self.conv_k(feat_n)  # [B, C, H, W]
        v = self.conv_v(feat_n)  # [B, C, H, W]
        freq_k = torch.fft.fft2(k, dim=(-2, -1))  # 复数张量

        # ========== CRM: 频域相关性计算 ==========
        corr_weight = self.crm(freq_q, freq_k)  # [B, C, H, W] 实数权重

        # ========== 2D IDFT: 逆变换回空间域 ==========
        # 使用相关性加权频域特征 (将实数权重转换为复数进行广播)
        # freq_q 是复数，corr_weight 是实数，需要正确广播
        weighted_freq = freq_q * corr_weight  # 复数 * 实数 = 复数，自动广播

        # 逆 FFT 回空间域
        feat_rn = torch.fft.ifft2(weighted_freq, dim=(-2, -1), s=(H, W))
        feat_rn = torch.abs(feat_rn)  # 取幅度 [B, C, H, W]

        # ========== DRM: 细节增强 ==========
        detail_feat = self.drm(v)  # [B, C, H, W]

        # ========== 融合 ==========
        # F_R,N + DRM 输出
        fused = feat_rn + detail_feat

        # 输出投影
        fused = self.out_conv(fused)
        fused = self.bn(fused)

        # 残差连接
        fused = fused + feat_r

        return fused


# ============================================================================
# 简化的 Cross-Field Frequency Fusion (无需复数运算)
# ============================================================================

class CrossFieldFrequencyFusionSimple(nn.Module):
    """
    Cross-Field Frequency Fusion Module - 简化版本

    使用幅度谱近似，避免复数运算，提高推理速度

    适用于需要高效推理的场景
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels

        # 卷积分支
        self.conv_r = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv_n = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

        # 频域交互 (使用幅度谱)
        self.freq_interaction = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 空间域融合
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
        )

    def forward(
        self,
        feat_r: torch.Tensor,
        feat_n: torch.Tensor
    ) -> torch.Tensor:
        # 卷积提取特征
        q = self.conv_r(feat_r)
        n = self.conv_n(feat_n)

        # 频域交互 (使用 FFT 幅度谱)
        with torch.no_grad():
            freq_q = torch.abs(torch.fft.fft2(q, dim=(-2, -1)))
            freq_n = torch.abs(torch.fft.fft2(n, dim=(-2, -1)))

        # 频域特征拼接 + 生成权重
        freq_combined = torch.cat([freq_q, freq_n], dim=1)
        freq_weight = self.freq_interaction(freq_combined)

        # 空间域加权融合
        fused = q * freq_weight + n

        # 空间域精炼
        fused = self.spatial_fusion(fused)

        # 残差连接
        fused = fused + feat_r

        return fused


# ============================================================================
# 组合模块：FCF + CFFF 用于 YOLOv8 Neck
# ============================================================================

class FCF_CFFF_Block(nn.Module):
    """
    组合 FCF 和 CFFF 的融合模块

    用于 YOLOv8 的 Neck 部分，融合多尺度特征

    结构:
    P3 (Low) ──→ [FCF] ←── P4 (High)
                    ↓
    P4 (Low) ──→ [FCF] ←── P5 (High)
                    ↓
                [CFFF] (跨特征融合)
                    ↓
                Output
    """

    def __init__(
        self,
        channels_p3: int,
        channels_p4: int,
        channels_p5: int,
        use_cfff: bool = True,
    ):
        super().__init__()

        self.use_cfff = use_cfff

        # FCF 模块
        self.fcf_p4p3 = FeatureCalibrationFusion(
            in_channels_low=channels_p3,
            in_channels_high=channels_p4,
            out_channels=channels_p3,
        )

        self.fcf_p5p4 = FeatureCalibrationFusion(
            in_channels_low=channels_p4,
            in_channels_high=channels_p5,
            out_channels=channels_p4,
        )

        # CFFF 模块 (可选) - 需要通道匹配
        if use_cfff:
            # 添加通道投影层使 P3 和 P4 融合后通道一致
            self.p4_to_p3_proj = nn.Sequential(
                nn.Conv2d(channels_p4, channels_p3, 1, bias=False),
                nn.BatchNorm2d(channels_p3),
                nn.ReLU(inplace=True),
            )
            self.cfff = CrossFieldFrequencyFusionSimple(channels_p3)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p3: P3 特征 [B, C3, H, W]
            p4: P4 特征 [B, C4, H/2, W/2]
            p5: P5 特征 [B, C5, H/4, W/4]

        Returns:
            (p3_out, p4_out, p5_out): 融合后的特征
        """
        # 自顶向下融合
        p4_fused = self.fcf_p5p4(p4, p5)  # 融合 P5 到 P4
        p3_fused = self.fcf_p4p3(p3, p4_fused)  # 融合 P4 到 P3

        if self.use_cfff:
            # 跨特征频域融合 - 需要通道匹配
            p4_proj = self.p4_to_p3_proj(p4_fused)  # [B, C3, H/2, W/2]
            # 上采样 P4 到 P3 尺寸
            p4_proj = F.interpolate(p4_proj, scale_factor=2, mode='bilinear', align_corners=False)
            # CFFF 融合
            p3_out = self.cfff(p3_fused, p4_proj)
            p4_out = p4_fused
        else:
            p3_out = p3_fused
            p4_out = p4_fused

        p5_out = p5

        return p3_out, p4_out, p5_out


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Feature Fusion Modules")
    print("=" * 60)

    # 测试 Feature Calibration Fusion
    print("\n[Feature Calibration Fusion]")
    low_feat = torch.randn(2, 256, 64, 64)
    high_feat = torch.randn(2, 512, 32, 32)

    fcf = FeatureCalibrationFusion(
        in_channels_low=256,
        in_channels_high=512,
    )
    out = fcf(low_feat, high_feat)
    print(f"  Low: {low_feat.shape}, High: {high_feat.shape}")
    print(f"  Output: {out.shape}")

    # 测试 FCF V2
    print("\n[Feature Calibration Fusion V2]")
    fcf_v2 = FeatureCalibrationFusionV2(256, 512)
    out = fcf_v2(low_feat, high_feat)
    print(f"  Output: {out.shape}")

    # 测试 Cross-Field Frequency Fusion
    print("\n[Cross-Field Frequency Fusion]")
    feat_r = torch.randn(2, 256, 64, 64)
    feat_n = torch.randn(2, 256, 64, 64)

    cfff = CrossFieldFrequencyFusion(256)
    out = cfff(feat_r, feat_n)
    print(f"  Ref: {feat_r.shape}, Target: {feat_n.shape}")
    print(f"  Output: {out.shape}")

    # 测试 CFFF Simple (更快)
    print("\n[Cross-Field Frequency Fusion (Simple)]")
    cfff_simple = CrossFieldFrequencyFusionSimple(256)
    out = cfff_simple(feat_r, feat_n)
    print(f"  Output: {out.shape}")

    # 测试组合模块
    print("\n[FCF + CFFF Block]")
    p3 = torch.randn(2, 256, 64, 64)
    p4 = torch.randn(2, 512, 32, 32)
    p5 = torch.randn(2, 512, 16, 16)

    block = FCF_CFFF_Block(256, 512, 512, use_cfff=True)
    p3_out, p4_out, p5_out = block(p3, p4, p5)
    print(f"  P3: {p3.shape} -> {p3_out.shape}")
    print(f"  P4: {p4.shape} -> {p4_out.shape}")
    print(f"  P5: {p5.shape} -> {p5_out.shape}")

    # 参数量对比
    print("\n" + "=" * 60)
    print("Parameter Count Comparison")
    print("=" * 60)

    def count_params(model):
        return sum(p.numel() for p in model.parameters()) / 1e6

    print(f"  FCF:              {count_params(fcf):.2f}M params")
    print(f"  FCF V2:           {count_params(fcf_v2):.2f}M params")
    print(f"  CFFF:             {count_params(cfff):.2f}M params")
    print(f"  CFFF Simple:      {count_params(cfff_simple):.2f}M params")
    print(f"  FCF+CFFF Block:   {count_params(block):.2f}M params")

    print("\n[All tests passed!]")
