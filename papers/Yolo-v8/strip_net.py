"""
StripNet Backbone for YOLOv8
基于 Strip R-CNN 的 StripNet 骨干网络设计

Paper: https://arxiv.org/abs/2501.03775
GitHub: https://github.com/HVision-NKU/Strip-R-CNN

网络变体:
- StripNet-Tiny: 3.8M 参数，18.2G FLOPs
- StripNet-Small: 13.3M 参数，52.3G FLOPs
- StripNet-Base: 更大版本

Table 1 配置:
Model    {C1, C2, C3, C4}      {D1, D2, D3, D4}
Tiny     {32, 64, 160, 256}    {3, 3, 5, 2}
Small    {64, 128, 320, 512}   {2, 2, 4, 2}
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional

# 支持直接运行和模块导入两种模式
try:
    from .strip_conv import StripModule, StripC2f
except ImportError:
    from strip_conv import StripModule, StripC2f


class Stem(nn.Module):
    """
    Stem 层 - 输入下采样模块
    跟随 LSKNet 设计，使用多级下采样
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()

        mid_channels = out_channels // 2

        # 两级下采样
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownSample(nn.Module):
    """
    下采样模块 - 用于 stage 之间过渡
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StripNet(nn.Module):
    """
    StripNet 骨干网络 - 用于 YOLOv8

    基于论文 Table 1 的配置
    使用 StripModule 作为基本构建块

    Args:
        channels: 各阶段输出通道数 [C1, C2, C3, C4]
        depths: 各阶段 StripModule 数量 [D1, D2, D3, D4]
        strip_kernel_size: 条带卷积核大小（默认 19）
        in_channels: 输入通道数（默认 3）
        out_indices: 输出哪些阶段的特征（默认 [1,2,3] 对应 P3,P4,P5）
    """

    def __init__(
        self,
        channels: List[int] = [64, 128, 320, 512],  # Small 配置
        depths: List[int] = [2, 2, 4, 2],  # Small 配置
        strip_kernel_size: int = 19,
        in_channels: int = 3,
        out_indices: List[int] = [1, 2, 3],  # 输出 P3, P4, P5
    ):
        super().__init__()

        self.channels = channels
        self.depths = depths
        self.strip_kernel_size = strip_kernel_size
        self.out_indices = out_indices

        # Stem
        self.stem = Stem(in_channels, channels[0])

        # 构建各阶段
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, (out_ch, depth) in enumerate(zip(channels, depths)):
            # 确定当前阶段的输入通道
            if i == 0:
                # 第一阶段输入来自 Stem
                in_ch = channels[0]
            else:
                # 其他阶段输入来自下采样
                in_ch = channels[i]

            # 创建 Strip 模块
            stage_modules = nn.ModuleList()
            for j in range(depth):
                module_in_ch = in_ch if j == 0 else out_ch
                stage_modules.append(
                    StripModule(
                        module_in_ch,
                        out_ch,
                        strip_kernel_size=strip_kernel_size,
                        expansion=1.0,
                        shortcut=True,
                    )
                )
            self.stages.append(stage_modules)

            # 添加下采样（除了最后一个阶段）
            if i < len(channels) - 1:
                self.downsamples.append(
                    DownSample(out_ch, channels[i + 1])
                )
            else:
                self.downsamples.append(nn.Identity())

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            多尺度特征列表 [P3, P4, P5]
        """
        outputs = []

        # Stem
        x = self.stem(x)

        # Stages
        for i, (stage, downsample) in enumerate(zip(self.stages, self.downsamples)):
            # 应用 Strip 模块
            for module in stage:
                x = module(x)

            # 在指定阶段输出特征
            if i in self.out_indices:
                outputs.append(x)

            # 下采样到下一阶段
            if i < len(self.stages) - 1:
                x = downsample(x)

        return outputs


def strip_net_tiny(
    strip_kernel_size: int = 19,
    pretrained: bool = False,
    **kwargs
) -> StripNet:
    """
    StripNet-Tiny: 3.8M 参数，18.2G FLOPs

    Table 1: {C1, C2, C3, C4} = {32, 64, 160, 256}
             {D1, D2, D3, D4} = {3, 3, 5, 2}
    """
    model = StripNet(
        channels=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],
        strip_kernel_size=strip_kernel_size,
        **kwargs
    )
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


def strip_net_small(
    strip_kernel_size: int = 19,
    pretrained: bool = False,
    **kwargs
) -> StripNet:
    """
    StripNet-Small: 13.3M 参数，52.3G FLOPs

    Table 1: {C1, C2, C3, C4} = {64, 128, 320, 512}
             {D1, D2, D3, D4} = {2, 2, 4, 2}

    这是论文的主要配置，在 DOTA-v1.0 上达到 82.75% mAP
    """
    model = StripNet(
        channels=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        strip_kernel_size=strip_kernel_size,
        **kwargs
    )
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


def strip_net_base(
    strip_kernel_size: int = 19,
    pretrained: bool = False,
    **kwargs
) -> StripNet:
    """
    StripNet-Base: 更大版本

    增加通道数和深度以获得更好性能
    """
    model = StripNet(
        channels=[96, 192, 384, 768],
        depths=[3, 3, 6, 3],
        strip_kernel_size=strip_kernel_size,
        **kwargs
    )
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


# 可变核大小配置（论文 Table 8 消融实验）
def strip_net_small_kernel_15(**kwargs) -> StripNet:
    """StripNet-Small with kernel size 15"""
    return strip_net_small(strip_kernel_size=15, **kwargs)


def strip_net_small_kernel_21(**kwargs) -> StripNet:
    """StripNet-Small with kernel size 21"""
    return strip_net_small(strip_kernel_size=21, **kwargs)


def strip_net_small_progressive(**kwargs) -> StripNet:
    """
    StripNet-Small with progressive kernel sizes
    Table 8: (21, 19, 17, 15) - 从大到小
    """
    # 需要自定义每个 stage 的 kernel size
    model = StripNet(
        channels=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        strip_kernel_size=19,  # 默认值，实际使用需要修改 StripModule
        **kwargs
    )
    # TODO: 为每个 stage 设置不同的 kernel size
    return model


if __name__ == "__main__":
    # 测试代码
    print("Testing StripNet Backbones...")

    # 测试 StripNet-Small
    x = torch.randn(1, 3, 640, 640)

    model = strip_net_small()
    outputs = model(x)

    print(f"\nStripNet-Small:")
    print(f"Input: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"  P{i+3}: {out.shape}")

    # 测试 StripNet-Tiny
    model_tiny = strip_net_tiny()
    outputs_tiny = model_tiny(x)

    print(f"\nStripNet-Tiny:")
    print(f"Input: {x.shape}")
    for i, out in enumerate(outputs_tiny):
        print(f"  P{i+3}: {out.shape}")

    # 计算参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters()) / 1e6

    print(f"\nStripNet-Small Params: {count_params(model):.2f}M")
    print(f"StripNet-Tiny Params: {count_params(model_tiny):.2f}M")

    print("\nAll tests passed!")
