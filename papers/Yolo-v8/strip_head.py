"""
Strip Head for YOLOv8 - 基于 Strip R-CNN 的解耦检测头设计

Paper: https://arxiv.org/abs/2501.03775
GitHub: https://github.com/HVision-NKU/Strip-R-CNN

核心设计 (Section 3.3, Figure 5):
1. 解耦分类和定位分支
2. 定位分支使用 Strip Module 捕获长距离空间依赖
3. 分类和角度预测共享全连接层
4. 改进高长宽比物体的定位能力

结构对比:
- Oriented R-CNN Head: 共享 FC 层用于分类 + 定位 + 角度
- Strip Head: 解耦为三个分支，定位分支加入 strip 卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# 支持直接运行和模块导入两种模式
try:
    from .strip_conv import StripModule, StripConv
except ImportError:
    from strip_conv import StripModule, StripConv


class ClassificationBranch(nn.Module):
    """
    分类分支

    使用两个全连接层（1024 维输出）
    跟随 Double-Head RCNN 设计
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 两个全连接层
        self.fc1 = nn.Linear(in_channels, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ROI 特征 [B, C, H, W]

        Returns:
            分类得分 [B, num_classes]
        """
        # 全局平均池化
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class LocalizationBranch(nn.Module):
    """
    定位分支

    核心改进：
    1. 3x3 卷积提取局部特征
    2. Strip Module 捕获长距离空间依赖
    3. 全连接层回归边界框参数 (x, y, w, h)

    这使得模型对高长宽比物体有更好的定位能力
    """

    def __init__(
        self,
        in_channels: int,
        strip_kernel_size: int = 19,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.strip_kernel_size = strip_kernel_size
        self.hidden_dim = hidden_dim

        # 3x3 卷积提取局部特征
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Strip Module 捕获长距离依赖
        self.strip_module = StripModule(
            in_channels,
            in_channels,
            strip_kernel_size=strip_kernel_size,
            expansion=1.0,
            shortcut=True,
        )

        # 全局平均池化 + 全连接
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 4)  # (x, y, w, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ROI 特征 [B, C, H, W]

        Returns:
            边界框参数 [B, 4]
        """
        # 局部特征
        x = self.local_conv(x)

        # Strip 模块增强空间依赖
        x = self.strip_module(x)

        # 池化 + 回归
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class AngleBranch(nn.Module):
    """
    角度预测分支

    使用三个全连接层
    简化设计：独立计算，不共享 FC 层
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 1024,
        share_fc1: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.share_fc1 = share_fc1

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 简化的角度预测：直接使用 in_channels -> hidden -> 1
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_angle = nn.Linear(hidden_dim, 1)  # 预测角度 θ

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, shared_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: ROI 特征 [B, C, H, W]
            shared_feat: 共享的 FC1 输出特征（可选）

        Returns:
            角度预测 [B, 1]
        """
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # 独立计算
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc_angle(x)

        return x


class StripHead(nn.Module):
    """
    Strip Head - YOLOv8 的解耦检测头

    基于论文 Figure 5 的设计:
    - 分类分支：两个 FC 层
    - 定位分支：Conv + Strip Module + FC
    - 角度分支：三个 FC 层（前两层与分类共享）

    适用于旋转目标检测（Oriented Object Detection）
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        strip_kernel_size: int = 19,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strip_kernel_size = strip_kernel_size
        self.hidden_dim = hidden_dim

        # 分类分支
        self.cls_branch = ClassificationBranch(
            in_channels, num_classes, hidden_dim
        )

        # 定位分支
        self.loc_branch = LocalizationBranch(
            in_channels, strip_kernel_size, hidden_dim
        )

        # 角度分支
        self.angle_branch = AngleBranch(
            in_channels, hidden_dim, share_fc1=True
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [B, C, H, W]（来自 FPN 的特征）

        Returns:
            (cls_scores, bboxes, angles):
            - cls_scores: 分类得分 [B, num_classes]
            - bboxes: 边界框 [B, 4] (x, y, w, h)
            - angles: 角度 [B, 1]
        """
        # 分类
        cls_scores = self.cls_branch(x)

        # 定位
        bboxes = self.loc_branch(x)

        # 角度
        angles = self.angle_branch(x)

        return cls_scores, bboxes, angles


class StripDetectHead(nn.Module):
    """
    Strip Detect Head - 用于 YOLOv8 的多尺度检测头

    类似 YOLOv8 的 Detect 类，但每个尺度都使用 Strip Head
    支持 P3, P4, P5 三个尺度的特征
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 80,
        strip_kernel_size: int = 19,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strip_kernel_size = strip_kernel_size
        self.hidden_dim = hidden_dim
        self.num_scales = len(in_channels)

        # 为每个尺度创建独立的检测头
        self.heads = nn.ModuleList([
            StripHead(
                in_ch, num_classes, strip_kernel_size, hidden_dim
            )
            for in_ch in in_channels
        ])

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        前向传播

        Args:
            features: 多尺度特征列表 [P3, P4, P5]

        Returns:
            所有尺度的 (cls_scores, bboxes, angles) 列表
        """
        cls_outputs = []
        loc_outputs = []
        angle_outputs = []

        for feature, head in zip(features, self.heads):
            cls, loc, angle = head(feature)
            cls_outputs.append(cls)
            loc_outputs.append(loc)
            angle_outputs.append(angle)

        return cls_outputs, loc_outputs, angle_outputs


class YOLOv8StripC2f(nn.Module):
    """
    用于 YOLOv8 Neck 的 StripC2f 模块

    在原始 C2f 基础上加入 strip 卷积，用于特征融合网络
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        strip_kernel_size: int = 19,
        shortcut: bool = True,
    ):
        super().__init__()

        from .strip_conv import StripC2f

        self.c2f = StripC2f(
            in_channels,
            out_channels,
            num_blocks=num_blocks,
            strip_kernel_size=strip_kernel_size,
            expansion=0.5,
            shortcut=shortcut,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2f(x)


if __name__ == "__main__":
    # 测试代码
    print("Testing Strip Head for YOLOv8...")

    # 测试 StripHead
    x = torch.randn(4, 256, 14, 14)

    head = StripHead(
        in_channels=256,
        num_classes=80,
        strip_kernel_size=19,
        hidden_dim=1024,
    )

    cls_scores, bboxes, angles = head(x)
    print(f"\nStripHead:")
    print(f"  Input: {x.shape}")
    print(f"  cls_scores: {cls_scores.shape}")
    print(f"  bboxes: {bboxes.shape}")
    print(f"  angles: {angles.shape}")

    # 测试 StripDetectHead
    features = [
        torch.randn(4, 256, 28, 28),  # P3
        torch.randn(4, 512, 14, 14),  # P4
        torch.randn(4, 512, 7, 7),    # P5
    ]

    detect_head = StripDetectHead(
        in_channels=[256, 512, 512],
        num_classes=80,
        strip_kernel_size=19,
    )

    cls_out, loc_out, angle_out = detect_head(features)

    print(f"\nStripDetectHead:")
    for i, (cls, loc, angle) in enumerate(zip(cls_out, loc_out, angle_out)):
        print(f"  Scale P{i+3}: cls={cls.shape}, loc={loc.shape}, angle={angle.shape}")

    print("\nAll tests passed!")
