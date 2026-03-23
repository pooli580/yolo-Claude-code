"""
YOLOv8-Strip: YOLOv8 with Strip R-CNN Integration
完整的 YOLOv8-Strip 模型定义

使用方法:
    from yolov8_strip import YOLOv8Strip, yolov8s_strip

    # 创建模型
    model = yolov8s_strip(num_classes=80)

    # 前向传播
    img = torch.randn(1, 3, 640, 640)
    outputs = model(img)

模型变体:
- YOLOv8n-Strip: Nano 版本
- YOLOv8s-Strip: Small 版本 (推荐)
- YOLOv8m-Strip: Medium 版本
- YOLOv8l-Strip: Large 版本
- YOLOv8x-Strip: Extra Large 版本
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

# 支持直接运行和模块导入两种模式
try:
    from .strip_net import StripNet, strip_net_small, strip_net_tiny
    from .strip_head import StripDetectHead
    from .strip_conv import StripC2f
except ImportError:
    from strip_net import StripNet, strip_net_small, strip_net_tiny
    from strip_head import StripDetectHead
    from strip_conv import StripC2f


class SPPF(nn.Module):
    """
    SPPF 层 - Spatial Pyramid Pooling Fast
    YOLOv8 的标准配置，保持兼容
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        hidden_channels = in_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.mp = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLOv8Strip(nn.Module):
    """
    YOLOv8-Strip: 完整的 YOLOv8 架构集成 Strip R-CNN

    架构:
    - Backbone: StripNet (替换原始 CSPDarknet)
    - Neck: FPN+PAN 使用 StripC2f
    - Head: StripDetectHead (解耦旋转检测头)

    Args:
        backbone: StripNet 骨干网络
        num_classes: 类别数
        strip_kernel_size: 条带卷积核大小
    """

    def __init__(
        self,
        backbone: StripNet,
        num_classes: int = 80,
        strip_kernel_size: int = 19,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.strip_kernel_size = strip_kernel_size

        # 获取骨干网络输出通道
        backbone_channels = backbone.channels

        # Neck: FPN + PAN 结构
        # 使用 StripC2f 替代标准 C2f
        self.neck = self._build_neck(backbone_channels)

        # Detect Head
        self.head = StripDetectHead(
            in_channels=backbone_channels[1:],  # P3, P4, P5
            num_classes=num_classes,
            strip_kernel_size=strip_kernel_size,
        )

    def _build_neck(self, channels: List[int]) -> nn.Module:
        """
        构建 FPN+PAN Neck

        channels: [C1, C2, C3, C4] 对应 [P2, P3, P4, P5]
        YOLOv8 的 PAN 结构：
        - FPN: P5 -> P4 -> P3 (自上而下)
        - PAN: P3 -> P4 -> P5 (自下而上)
        """

        # 上采样模块
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # FPN 自上而下特征融合
        self.fpn_p5_to_p4 = StripC2f(
            channels[3] + channels[2],  # P5+P4(upsample)
            channels[2],
            num_blocks=3,
            strip_kernel_size=self.strip_kernel_size,
            shortcut=False,
        )

        self.fpn_p4_to_p3 = StripC2f(
            channels[2] + channels[1],  # P4(fusion)+P3(upsample)
            channels[1],
            num_blocks=3,
            strip_kernel_size=self.strip_kernel_size,
            shortcut=False,
        )

        # PAN 自下而上特征融合
        # P3 -> P4: p3_fpn(128) 下采样到 320，与 p4_fpn(320) 拼接
        self.pan_p3_to_p4 = StripC2f(
            channels[2] + channels[2],  # P4(320) + P3_down(320)
            channels[2],
            num_blocks=3,
            strip_kernel_size=self.strip_kernel_size,
            shortcut=False,
        )

        # P4 -> P5: p4_out(320) 下采样到 512，与 p5_sppf(512) 拼接
        self.pan_p4_to_p5 = StripC2f(
            channels[3] + channels[3],  # P5(512) + P4_down(512)
            channels[3],
            num_blocks=3,
            strip_kernel_size=self.strip_kernel_size,
            shortcut=False,
        )

        # SPPF (在 P5 上)
        self.sppf = SPPF(channels[3], channels[3])

        # 下采样层
        # P3: 128 -> 320 (为了与 P4 拼接)
        self.down_p3_to_p4 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True),
        )
        # P4: 320 -> 512 (为了与 P5 拼接)
        self.down_p4_to_p5 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.SiLU(inplace=True),
        )

        return nn.ModuleDict({
            'fpn_p5_to_p4': self.fpn_p5_to_p4,
            'fpn_p4_to_p3': self.fpn_p4_to_p3,
            'pan_p3_to_p4': self.pan_p3_to_p4,
            'pan_p4_to_p5': self.pan_p4_to_p5,
            'sppf': self.sppf,
            'down_p3_to_p4': self.down_p3_to_p4,
            'down_p4_to_p5': self.down_p4_to_p5,
            'upsample': self.upsample,
        })

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            字典包含:
            - cls: 分类输出列表 [P3, P4, P5]
            - loc: 定位输出列表 [P3, P4, P5]
            - angle: 角度输出列表 [P3, P4, P5]
        """
        # Backbone: 获取 P3, P4, P5 特征
        features = self.backbone(x)

        if len(features) == 4:
            p2, p3, p4, p5 = features
        else:
            p3, p4, p5 = features

        # FPN: 自上而下特征融合
        # P5 -> P4
        p5_up = self.upsample(p5)
        p4_fpn = torch.cat([p4, p5_up], dim=1)
        p4_fpn = self.fpn_p5_to_p4(p4_fpn)

        # P4 -> P3
        p4_fpn_up = self.upsample(p4_fpn)
        p3_fpn = torch.cat([p3, p4_fpn_up], dim=1)
        p3_fpn = self.fpn_p4_to_p3(p3_fpn)

        # SPPF
        p5_sppf = self.sppf(p5)

        # PAN: 自下而上特征融合
        # P3 -> P4: p3_fpn 下采样后与 p4_fpn 拼接
        p3_pan_down = self.down_p3_to_p4(p3_fpn)  # 128 -> 320
        p4_pan = torch.cat([p4_fpn, p3_pan_down], dim=1)  # 320 + 320
        p4_out = self.pan_p3_to_p4(p4_pan)  # -> 320

        # P4 -> P5: p4_out 下采样后与 p5_sppf 拼接
        p4_pan_down = self.down_p4_to_p5(p4_out)  # 320 -> 512
        p5_pan = torch.cat([p5_sppf, p4_pan_down], dim=1)  # 512 + 512
        p5_out = self.pan_p4_to_p5(p5_pan)  # -> 512

        # P3 输出
        p3_out = p3_fpn  # 128

        # 检测头
        cls_outputs, loc_outputs, angle_outputs = self.head([p3_out, p4_out, p5_out])

        return {
            'cls': cls_outputs,
            'loc': loc_outputs,
            'angle': angle_outputs,
        }


def yolov8n_strip(
    num_classes: int = 80,
    strip_kernel_size: int = 19,
    pretrained: bool = False,
) -> YOLOv8Strip:
    """
    YOLOv8n-Strip: Nano 版本

    使用修改的 StripNet-Tiny 骨干，通道配置适配 YOLOv8 Neck
    """
    # 使用适配 YOLOv8 Neck 的通道配置
    # 原始 Tiny: [32, 64, 160, 256]
    # 调整后：   [64, 128, 256, 512] 以匹配 YOLOv8s 的 Neck
    backbone = StripNet(
        channels=[64, 128, 256, 512],  # 修改通道以匹配 Neck
        depths=[2, 2, 3, 2],            # 减少深度保持轻量
        strip_kernel_size=strip_kernel_size,
    )

    model = YOLOv8Strip(
        backbone=backbone,
        num_classes=num_classes,
        strip_kernel_size=strip_kernel_size,
    )

    return model


def yolov8s_strip(
    num_classes: int = 80,
    strip_kernel_size: int = 19,
    pretrained: bool = False,
) -> YOLOv8Strip:
    """
    YOLOv8s-Strip: Small 版本（推荐）

    使用 StripNet-Small 骨干
    论文主要配置：13.3M 参数，在 DOTA-v1.0 上达到 82.75% mAP
    """
    backbone = strip_net_small(
        strip_kernel_size=strip_kernel_size,
        pretrained=pretrained,
    )

    model = YOLOv8Strip(
        backbone=backbone,
        num_classes=num_classes,
        strip_kernel_size=strip_kernel_size,
    )

    return model


def yolov8m_strip(
    num_classes: int = 80,
    strip_kernel_size: int = 19,
    pretrained: bool = False,
) -> YOLOv8Strip:
    """
    YOLOv8m-Strip: Medium 版本

    使用 StripNet-Base 骨干
    """
    backbone = strip_net_base(
        strip_kernel_size=strip_kernel_size,
        pretrained=pretrained,
    )

    model = YOLOv8Strip(
        backbone=backbone,
        num_classes=num_classes,
        strip_kernel_size=strip_kernel_size,
    )

    return model


def yolov8l_strip(
    num_classes: int = 80,
    strip_kernel_size: int = 19,
    pretrained: bool = False,
) -> YOLOv8Strip:
    """
    YOLOv8l-Strip: Large 版本

    更大的通道和深度
    """
    backbone = StripNet(
        channels=[128, 256, 512, 1024],
        depths=[4, 4, 8, 4],
        strip_kernel_size=strip_kernel_size,
    )

    model = YOLOv8Strip(
        backbone=backbone,
        num_classes=num_classes,
        strip_kernel_size=strip_kernel_size,
    )

    return model


def count_parameters(model: nn.Module) -> float:
    """计算模型参数量 (M)"""
    return sum(p.numel() for p in model.parameters()) / 1e6


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Testing YOLOv8-Strip Models")
    print("=" * 60)

    # 创建测试输入
    x = torch.randn(1, 3, 640, 640)

    # 测试不同变体
    models = {
        'YOLOv8n-Strip': yolov8n_strip,
        'YOLOv8s-Strip': yolov8s_strip,
        # 'YOLOv8m-Strip': yolov8m_strip,  # 需要 strip_net_base
        # 'YOLOv8l-Strip': yolov8l_strip,
    }

    for name, model_fn in models.items():
        print(f"\n{name}:")
        print("-" * 40)

        try:
            model = model_fn(num_classes=80)
            outputs = model(x)

            print(f"  参数量：{count_parameters(model):.2f}M")
            print(f"  输入：{x.shape}")

            for i, (cls, loc, angle) in enumerate(
                zip(outputs['cls'], outputs['loc'], outputs['angle'])
            ):
                print(f"  P{i+3} - cls: {cls.shape}, loc: {loc.shape}, angle: {angle.shape}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
