"""
Dynamic Upsampling Module for YOLOv8
Based on:
1. DySample: Learning to Upsample by Learning to Sample (ICCV 2023)
   - https://arxiv.org/abs/2312.15669
   - https://github.com/tiny-smart/dysample

2. DUPS: Dynamic Upsampling for Efficient Semantic Segmentation
   - Dual-path dynamic upsampling with 2D/1D convolution variants

核心思想:
1. DySample - 基于点采样的超轻量动态上采样器
   - 避开动态卷积的计算开销，从点采样角度重新公式化上采样
   - 生成内容感知的采样偏移量，用 grid_sample 重采样特征
   - 无需高维引导特征，单输入即可实现动态上采样
   - 超轻量：仅使用 PyTorch 内置函数，无需自定义 CUDA

2. DUPS - 双路径动态上采样
   - (a) 2D Conv 路径：三路 2D 卷积 + 上采样 + 融合
   - (b) 1D Conv 路径：使用 1D 卷积降低计算量

关键设计:
- Initial Sampling Position: 双线性初始化而非最近邻初始化
- Offset Scope: 静态/动态范围因子约束偏移量移动范围
- Grouping: 通道分组减少参数量
- Dynamic Scope Factor: 逐点动态调整偏移范围

适用于:
- 语义分割特征恢复
- 目标检测特征金字塔上采样
- 遥感图像密集预测任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ============================================================================
# DySample 核心组件
# ============================================================================

class DySample(nn.Module):
    """
    DySample: 超轻量动态上采样器

    基于点采样而非动态卷积，使用 grid_sample 实现内容感知上采样

    Args:
        in_channels: 输入通道数
        scale_factor: 上采样倍数 (默认 2)
        groups: 特征分组数 (默认 4, 减少参数量)
        mode: 偏移生成模式 ('static' | 'dynamic')
            - static: 静态范围因子 (0.25)
            - dynamic: 动态范围因子 (通过 sigmoid 生成)
        init_mode: 初始化模式 ('bilinear' | 'nearest')
            - bilinear: 双线性初始化 (推荐)
            - nearest: 最近邻初始化

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, s*H, s*W)

    Example:
        >>> upsampler = DySample(64, scale_factor=2)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = upsampler(x)  # (1, 64, 64, 64)
    """

    def __init__(
        self,
        in_channels: int,
        scale_factor: int = 2,
        groups: int = 4,
        mode: str = 'static',
        init_mode: str = 'bilinear',
    ):
        super().__init__()

        assert scale_factor >= 1 and isinstance(scale_factor, int)
        assert mode in ('static', 'dynamic')
        assert init_mode in ('bilinear', 'nearest')

        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.groups = groups
        self.mode = mode
        self.init_mode = init_mode

        # 计算每组通道数
        self.out_channels = in_channels

        # 偏移生成器
        # 输出通道：2 * groups * scale_factor^2 (x, y 坐标 + 分组 + 上采样点)
        offset_channels = 2 * groups * (scale_factor ** 2)

        if mode == 'static':
            # 静态模式：单层线性投影
            self.offset_proj = nn.Conv2d(
                in_channels, offset_channels,
                kernel_size=1,
                bias=True
            )
            # 静态范围因子 (0.25 为理论最优值，防止采样点重叠)
            self.scope_factor = 0.25
        else:
            # 动态模式：两层投影生成动态范围因子
            self.offset_proj1 = nn.Conv2d(
                in_channels, offset_channels,
                kernel_size=1,
                bias=False
            )
            self.offset_proj2 = nn.Conv2d(
                in_channels, offset_channels,
                kernel_size=1,
                bias=True
            )
            # 动态范围通过 sigmoid 约束在 [0, 0.5] 范围内
            self.scope_factor = 0.5

        # 初始化偏移为 0 (即初始为上采样网格)
        if mode == 'static':
            nn.init.constant_(self.offset_proj.bias, 0.0)
        else:
            nn.init.constant_(self.offset_proj2.bias, 0.0)

        # 双线性初始化网格
        self._init_sampling_grid()

    def _init_sampling_grid(self):
        """初始化标准采样网格"""
        # 该网格在 forward 中动态生成，与输入尺寸匹配
        pass

    def _make_sampling_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        生成初始采样网格

        Args:
            batch_size: 批次大小
            height: 输出高度
            width: 输出宽度

        Returns:
            采样网格 (2, H_out, W_out)
        """
        # 输出尺寸
        out_h = height * self.scale_factor
        out_w = width * self.scale_factor

        # 生成归一化坐标网格 [-1, 1]
        # grid_sample 需要 (x, y) 坐标，归一化到 [-1, 1]

        # 对于 bilinear 初始化，采样点均匀分布
        if self.init_mode == 'bilinear':
            # 在输入特征图上均匀采样
            y_coords = torch.linspace(0, height - 1, out_h, device=device, dtype=dtype)
            x_coords = torch.linspace(0, width - 1, out_w, device=device, dtype=dtype)
        else:
            # nearest 初始化：重复采样
            y_coords = torch.arange(0, height, device=device, dtype=dtype).repeat_interleave(self.scale_factor)
            x_coords = torch.arange(0, width, device=device, dtype=dtype).repeat_interleave(self.scale_factor)

        # 生成网格
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 合并为 (2, H_out, W_out)
        grid = torch.stack([grid_x, grid_y], dim=0)

        # 归一化到 [-1, 1] 范围 (grid_sample 要求)
        grid = grid / max(height - 1, 1) * 2 - 1

        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 简化实现

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            上采样特征 (B, C, s*H, s*W)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        out_h = H * self.scale_factor
        out_w = W * self.scale_factor

        # 1. 生成初始采样网格 (2, H_out, W_out)
        base_grid = self._make_sampling_grid(B, H, W, device, dtype)

        # 2. 生成偏移量
        if self.mode == 'static':
            # 静态模式
            offset = self.offset_proj(x)  # (B, 2*g*s^2, H, W)
        else:
            # 动态模式：两个投影相乘得到动态范围
            offset1 = self.offset_proj1(x)
            offset2 = self.offset_proj2(x)
            offset = self.scope_factor * torch.sigmoid(offset1) * offset2

        # 3. 重塑偏移量 (B, 2*g*s^2, H, W) -> (B, 2, s*H, s*W)
        # 简化处理：对每个空间位置生成 s^2 个采样点
        offset = offset.view(B, self.groups, 2, self.scale_factor, self.scale_factor, H, W)
        offset = offset.permute(0, 1, 5, 3, 6, 4, 2).contiguous()
        offset = offset.view(B, self.groups, out_h, out_w, 2)

        # 对分组取平均，得到 (B, out_h, out_w, 2)
        offset = offset.mean(dim=1)

        # 应用范围因子
        offset = offset * self.scope_factor

        # 4. 生成最终采样网格
        # base_grid: (2, out_h, out_w) -> (out_h, out_w, 2)
        base_grid = base_grid.permute(1, 2, 0)  # (out_h, out_w, 2)
        sampling_grid = base_grid + offset

        # 5. grid_sample 重采样
        output = F.grid_sample(
            x,
            sampling_grid,
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        )

        return output


# ============================================================================
# DUPS 风格的双路径上采样 (基于框图)
# ============================================================================

class DUPSBlock2D(nn.Module):
    """
    DUPS 2D 卷积版本上采样模块 (对应框图 a)

    结构:
    1. 输入分成三路，每路通过 2D Conv
    2. 两路进行上采样 (UP)
    3. 三路特征拼接
    4. 通过 2D Conv 融合输出

    参考框图 (a):
    - 三个并行的 2D Conv 分支
    - 两个分支后接 UP 操作
    - 拼接后通过最终 2D Conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        mid_channels = mid_channels or in_channels // 3

        # 三个并行的 2D 卷积分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )

        # 上采样层 (使用 DySample)
        self.upsample1 = DySample(mid_channels, scale_factor=scale_factor)
        self.upsample2 = DySample(mid_channels, scale_factor=scale_factor)

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 3, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 三个分支
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)

        # 上采样两个分支
        feat1 = self.upsample1(feat1)
        feat2 = self.upsample2(feat2)
        # feat3 不经过上采样，保持原分辨率

        # 为了拼接，需要将 feat3 也上采样到相同尺寸
        feat3 = F.interpolate(feat3, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # 拼接融合
        out = torch.cat([feat1, feat2, feat3], dim=1)
        out = self.fusion_conv(out)

        return out


class DUPSBlock1D(nn.Module):
    """
    DUPS 1D 卷积版本上采样模块 (对应框图 b)

    结构:
    1. 输入分成两路，每路通过 1D Conv (水平 + 垂直)
    2. 两路都进行上采样
    3. 拼接后通过 2D Conv 融合

    参考框图 (b):
    - 两个并行的 1D Conv 分支 (水平和垂直条带)
    - 都接 UP 操作
    - 拼接后通过 2D Conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size

        # 水平条带卷积 (1 x K)
        self.h_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels // 2,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                bias=False
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
        )

        # 垂直条带卷积 (K x 1)
        self.v_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels // 2,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
                bias=False
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
        )

        # 上采样层
        self.upsample_h = DySample(in_channels // 2, scale_factor=scale_factor)
        self.upsample_v = DySample(in_channels // 2, scale_factor=scale_factor)

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 水平条带
        h_feat = self.h_conv(x)
        h_feat = self.upsample_h(h_feat)

        # 垂直条带
        v_feat = self.v_conv(x)
        v_feat = self.upsample_v(v_feat)

        # 拼接融合
        out = torch.cat([h_feat, v_feat], dim=1)
        out = self.fusion_conv(out)

        return out


# ============================================================================
# 增强的动态上采样模块 (结合 DySample + DUPS)
# ============================================================================

class DynamicUpsample(nn.Module):
    """
    通用动态上采样模块

    整合 DySample 和 DUPS 的思想，提供多种上采样策略

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数 (默认等于 in_channels)
        scale_factor: 上采样倍数
        method: 上采样方法
            - 'dysample': 纯 DySample (最轻量)
            - 'dups-2d': DUPS 2D 卷积版本
            - 'dups-1d': DUPS 1D 卷积版本 (更高效)
            - 'hybrid': 混合模式 (DySample + 1D Conv)

    Example:
        >>> # 轻量级上采样
        >>> up = DynamicUpsample(64, 128, scale_factor=2, method='dysample')
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = up(x)  # (1, 128, 64, 64)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        scale_factor: int = 2,
        method: str = 'hybrid',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.scale_factor = scale_factor
        self.method = method

        if method == 'dysample':
            # 纯 DySample + 1x1 卷积调整通道
            self.upsample = nn.Sequential(
                DySample(in_channels, scale_factor=scale_factor),
                nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU(),
            )

        elif method == 'dups-2d':
            self.upsample = DUPSBlock2D(
                in_channels, self.out_channels,
                scale_factor=scale_factor
            )

        elif method == 'dups-1d':
            self.upsample = DUPSBlock1D(
                in_channels, self.out_channels,
                scale_factor=scale_factor
            )

        elif method == 'hybrid':
            # 混合模式：DySample 上采样 + 条带卷积增强
            self.dysample = DySample(in_channels, scale_factor=scale_factor)

            # 条带卷积增强
            self.strip_enhance = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, (1, 7), padding=(0, 3), bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.GELU(),
                nn.Conv2d(in_channels // 2, in_channels // 2, (7, 1), padding=(3, 0), bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.GELU(),
            )

            # 输出投影
            self.out_proj = nn.Sequential(
                nn.Conv2d(in_channels // 2, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU(),
            )

            self.upsample = self

        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == 'hybrid':
            # hybrid 模式特殊处理
            x = self.dysample(x)
            x = self.strip_enhance(x)
            return self.out_proj(x)
        return self.upsample(x)


# ============================================================================
# 用于 YOLOv8 的上采样层封装
# ============================================================================

class YOLOUpsample(nn.Module):
    """
    YOLOv8 兼容的上采样层

    可替换 YOLOv8 中的 nn.Upsample

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        scale_factor: 上采样倍数
        use_dynamic: 是否使用动态上采样 (默认 True)
        method: 动态上采样方法 ('dysample' | 'dups-1d' | 'dups-2d' | 'hybrid')

    Example:
        # 在 YOLOv8 模型中替换:
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 改为:
        # self.upsample = YOLOUpsample(channels, channels, scale_factor=2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_dynamic: bool = True,
        method: str = 'hybrid',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        if use_dynamic:
            self.conv = DynamicUpsample(
                in_channels, out_channels,
                scale_factor=scale_factor,
                method=method
            )
        else:
            # 退化到传统上采样 + 卷积
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Dynamic Upsampling Modules")
    print("=" * 60)

    # 测试 DySample
    print("\n[DySample]")
    x = torch.randn(2, 64, 32, 32)

    dysample_static = DySample(64, scale_factor=2, mode='static')
    out = dysample_static(x)
    print(f"  Static: {x.shape} -> {out.shape}")

    dysample_dynamic = DySample(64, scale_factor=2, mode='dynamic')
    out = dysample_dynamic(x)
    print(f"  Dynamic: {x.shape} -> {out.shape}")

    # 测试 DUPS 2D
    print("\n[DUPS Block 2D]")
    dups_2d = DUPSBlock2D(64, 128, scale_factor=2)
    out = dups_2d(x)
    print(f"  {x.shape} -> {out.shape}")

    # 测试 DUPS 1D
    print("\n[DUPS Block 1D]")
    dups_1d = DUPSBlock1D(64, 128, scale_factor=2)
    out = dups_1d(x)
    print(f"  {x.shape} -> {out.shape}")

    # 测试 DynamicUpsample
    print("\n[DynamicUpsample]")
    for method in ['dysample', 'dups-2d', 'dups-1d', 'hybrid']:
        up = DynamicUpsample(64, 128, scale_factor=2, method=method)
        out = up(x)
        print(f"  {method}: {x.shape} -> {out.shape}")

    # 测试 YOLOUpsample
    print("\n[YOLOUpsample]")
    yolo_up = YOLOUpsample(64, 128, scale_factor=2, use_dynamic=True, method='hybrid')
    out = yolo_up(x)
    print(f"  Dynamic: {x.shape} -> {out.shape}")

    yolo_up_static = YOLOUpsample(64, 128, scale_factor=2, use_dynamic=False)
    out = yolo_up_static(x)
    print(f"  Bilinear: {x.shape} -> {out.shape}")

    # 参数量对比
    print("\n" + "=" * 60)
    print("Parameter Count Comparison")
    print("=" * 60)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    print(f"  DySample (static):  {count_params(dysample_static):,} params")
    print(f"  DySample (dynamic): {count_params(dysample_dynamic):,} params")
    print(f"  DUPS 2D:            {count_params(dups_2d):,} params")
    print(f"  DUPS 1D:            {count_params(dups_1d):,} params")

    # 传统上采样 (几乎无参数)
    bilinear_up = nn.Upsample(scale_factor=2, mode='bilinear')
    print(f"  Bilinear (no conv): {count_params(bilinear_up):,} params")

    # 传统上采样 + 卷积
    traditional = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(64, 128, 3, padding=1)
    )
    print(f"  Traditional + Conv: {count_params(traditional):,} params")

    print("\n[All tests passed!]")
