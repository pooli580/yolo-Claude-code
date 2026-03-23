"""
Large Strip Convolution for YOLOv8
Based on Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
Paper: https://arxiv.org/abs/2501.03775
GitHub: https://github.com/HVision-NKU/Strip-R-CNN

核心思想:
- 传统方形卷积在高长宽比物体上表现受限
- 使用正交的大条带卷积（水平 + 垂直）捕获各向异性上下文
- 序列式组合而非并行，减少计算负担和特征冗余

关键配置:
- 默认条带卷积核大小：19x1 和 1x19（论文最优）
- 初始小方形卷积：5x5
- 适用于检测高长宽比物体（如船舶、道路、飞机等）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LargeStripConv(nn.Module):
    """
    大条带卷积模块

    使用正交的水平 (H-Strip) 和垂直 (V-Strip) 卷积分离捕获空间特征
    相比大方形卷积，能更有效地捕获高长宽比物体的特征

    Args:
        channels: 输入通道数
        kernel_size: 条带卷积核大小（奇数，默认 19）
        dilation: 膨胀率（默认 1）
        groups: 分组卷积数（默认 channels，即 depthwise）
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 19,
        dilation: int = 1,
        groups: Optional[int] = None,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups if groups is not None else channels

        # 水平条带卷积 (1 x kernel_size)
        self.conv_h = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size - 1) // 2 * dilation),
            dilation=(1, dilation),
            groups=self.groups,
            bias=False
        )

        # 垂直条带卷积 (kernel_size x 1)
        self.conv_v = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2 * dilation, 0),
            dilation=(dilation, 1),
            groups=self.groups,
            bias=False
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化卷积权重"""
        for m in [self.conv_h, self.conv_v]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            (h_feat, v_feat): 水平和垂直条带特征
        """
        # 先应用水平条带卷积
        h_feat = self.conv_h(x)

        # 再应用垂直条带卷积
        v_feat = self.conv_v(x)

        return h_feat, v_feat


class StripConv(nn.Module):
    """
    完整的 Strip 卷积模块（包含初始方形卷积和条带卷积）

    结构:
    1. 小方形卷积 (5x5) 提取局部特征
    2. 水平大条带卷积 (1x19) 捕获水平方向长距离依赖
    3. 垂直大条带卷积 (19x1) 捕获垂直方向长距离依赖
    4. 逐点卷积融合通道信息
    5. 作为注意力权重重新加权输入特征

    参考论文 Figure 4 和 Section 3.2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        strip_kernel_size: int = 19,
        expansion: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.strip_kernel_size = strip_kernel_size

        # 1. 初始小方形卷积 (5x5 depthwise)
        self.local_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU()

        # 2. 大条带卷积（水平 + 垂直）
        self.strip_conv = LargeStripConv(
            in_channels,
            kernel_size=strip_kernel_size,
            groups=in_channels
        )

        # 3. 逐点卷积融合通道
        pw_channels = int(in_channels * expansion)
        self.pw_conv = nn.Conv2d(
            in_channels * 2,  # h_feat + v_feat
            pw_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(pw_channels)
        self.act2 = nn.GELU()

        # 4. 输出投影（如果需要改变通道数）
        if pw_channels != out_channels:
            self.proj_conv = nn.Conv2d(
                pw_channels, out_channels,
                kernel_size=1,
                bias=False
            )
            self.proj_bn = nn.BatchNorm2d(out_channels)
        else:
            self.proj_conv = None
            self.proj_bn = None

        self.act3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            输出特征图 [B, out_channels, H, W]
        """
        # 保存恒等映射（用于注意力机制）
        identity = x

        # Step 1: 局部特征提取
        x = self.local_conv(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Step 2: 大条带卷积捕获长距离依赖
        h_feat, v_feat = self.strip_conv(x)

        # Step 3: 拼接水平和垂直特征
        strip_feat = torch.cat([h_feat, v_feat], dim=1)

        # Step 4: 逐点卷积融合
        strip_feat = self.pw_conv(strip_feat)
        strip_feat = self.bn2(strip_feat)
        strip_feat = self.act2(strip_feat)

        # Step 5: 作为注意力权重重新加权输入
        # 类似 SegNeXt 的注意力机制
        out = identity * strip_feat

        # Step 6: 输出投影
        if self.proj_conv is not None:
            out = self.proj_conv(out)
            out = self.proj_bn(out)

        out = self.act3(out)

        return out


class StripModule(nn.Module):
    """
    Strip 模块 - YOLOv8 的基本构建块

    基于论文 Figure 4 的 Strip Block 设计:
    - 包含 strip sub-block（空间特征提取）
    - 包含 FFN sub-block（通道混合）
    - 残差连接

    这类似于 YOLOv8 的 C2f 模块，但使用 strip 卷积替代标准卷积
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strip_kernel_size: int = 19,
        expansion: float = 1.0,
        shortcut: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strip_kernel_size = strip_kernel_size
        self.shortcut = shortcut

        # Strip sub-block
        self.strip_subblock = StripConv(
            in_channels,
            out_channels,
            strip_kernel_size=strip_kernel_size,
            expansion=expansion,
        )

        # FFN sub-block (跟随 LSKNet 设计)
        ffn_channels = int(out_channels * 2)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, ffn_channels, 1, bias=False),
            nn.BatchNorm2d(ffn_channels),
            nn.GELU(),
            nn.Conv2d(ffn_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # 如果通道不匹配，添加投影层
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            输出特征图 [B, out_channels, H, W]
        """
        identity = self.proj(x)

        # Strip sub-block
        out = self.strip_subblock(x)

        # FFN sub-block
        out = self.ffn(out)

        # 残差连接
        if self.shortcut:
            out = out + identity

        out = self.act(out)

        return out


class StripC2f(nn.Module):
    """
    Strip 版本的 C2f 模块 - 用于 YOLOv8 骨干网络

    将原始 C2f 中的瓶颈块替换为 StripModule
    提供更好的高长宽比物体检测能力
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        strip_kernel_size: int = 19,
        expansion: float = 0.5,
        shortcut: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.expansion = expansion

        # 初始投影
        hidden_channels = int(out_channels * expansion)
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # 初始投影用于跳跃连接（输出 hidden_channels 以保持统一）
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # Strip 模块堆叠
        self.strip_modules = nn.ModuleList([
            StripModule(
                hidden_channels,
                hidden_channels,
                strip_kernel_size=strip_kernel_size,
                expansion=1.0,
                shortcut=shortcut,
            )
            for _ in range(num_blocks)
        ])

        # 最终投影
        # 拼接后的通道数 = hidden_channels * (num_blocks + 2)
        # 包括：proj2 输出 + proj1 输出 + num_blocks 个 StripModule 输出
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden_channels * (num_blocks + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - C2f 风格的多路径融合

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            输出特征图 [B, out_channels, H, W]
        """
        # 初始投影
        x_proj = self.proj1(x)

        # 收集所有输出用于拼接
        outputs = [self.proj2(x)]  # 直接跳过连接 (out_channels)
        outputs.append(x_proj)  # 第一层输出 (hidden_channels)

        # 通过 Strip 模块堆叠
        for strip_module in self.strip_modules:
            x_proj = strip_module(x_proj)
            outputs.append(x_proj)  # (hidden_channels)

        # 拼接所有输出
        # 总通道数 = out_channels + hidden_channels * (num_blocks + 1)
        out = torch.cat(outputs, dim=1)

        # 最终投影
        out = self.proj_out(out)

        return out


# 注册到 YOLOv8
if __name__ == "__main__":
    # 测试代码
    print("Testing Strip Convolution Modules...")

    # 测试 LargeStripConv
    x = torch.randn(2, 64, 56, 56)
    strip_conv = LargeStripConv(64, kernel_size=19)
    h_feat, v_feat = strip_conv(x)
    print(f"LargeStripConv: {x.shape} -> H: {h_feat.shape}, V: {v_feat.shape}")

    # 测试 StripConv
    strip_conv_full = StripConv(64, 64, strip_kernel_size=19)
    out = strip_conv_full(x)
    print(f"StripConv: {x.shape} -> {out.shape}")

    # 测试 StripModule
    strip_module = StripModule(64, 64, strip_kernel_size=19)
    out = strip_module(x)
    print(f"StripModule: {x.shape} -> {out.shape}")

    # 测试 StripC2f
    strip_c2f = StripC2f(64, 128, num_blocks=3, strip_kernel_size=19)
    out = strip_c2f(x)
    print(f"StripC2f: {x.shape} -> {out.shape}")

    print("\nAll tests passed!")
