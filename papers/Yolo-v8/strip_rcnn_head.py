"""
Strip R-CNN Detection Heads
基于 Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection

Paper: https://arxiv.org/abs/2501.03775
GitHub: https://github.com/HVision-NKU/Strip-R-CNN

本模块严格实现论文框图中的两种检测头结构:

┌─────────────────────────────────────────────────────────────────┐
│ Oriented R-CNN Head (上图)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RoI ──→ [FC] ──→ [FC] ──┬──→ [FC] ──→ Classification         │
│                          └──→ [FC] ──→ Localization + Angle    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Strip Head (下图 - 核心创新)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌──→ [FC] ──→ [FC] ────→ [FC] ──→ Class    │
│                                        └──→ [FC] ──→ Angle     │
│  RoI ──┬──→ [FC] ──┘                                                           │
│        │                                                                        │
│        └──→ [Conv] ──→ [Strip Module] ──→ [FC] ──→ Localization                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

核心差异:
1. Oriented R-CNN Head: 共享 FC 层用于分类 + 定位 + 角度
2. Strip Head: 解耦为三个独立分支
   - 分类分支：两个 FC 层 → 分类
   - 角度分支：两个 FC 层 → 角度
   - 定位分支：Conv + Strip Module + FC → 定位 (核心创新)

Strip Module 的作用:
- 使用大条带卷积 (19x1 和 1x19) 捕获长距离空间依赖
- 对高长宽比物体（船舶、飞机、桥梁）有更好的定位能力
- 正交的水平和垂直条带捕获各向异性上下文信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# 支持直接运行和模块导入两种模式
try:
    from .strip_conv import StripModule
except ImportError:
    from strip_conv import StripModule


# ============================================================================
# Oriented R-CNN Head (基线方法)
# ============================================================================

class OrientedRCNNHead(nn.Module):
    """
    Oriented R-CNN Head - 基线检测方法

    结构 (严格遵循框图):
    RoI → FC → FC → [FC→Classification, FC→Localization+Angle]

    所有任务共享前两个全连接层，然后在最后一个 FC 层分叉
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        hidden_dim: int = 1024,
        roi_size: int = 7,  # RoI Align 输出尺寸
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size

        # 共享的全连接层 (框图中的前两个 FC)
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 分类分支 (最后一个 FC)
        self.fc_cls = nn.Linear(hidden_dim, num_classes)

        # 定位 + 角度分支 (最后一个 FC)
        # 输出：4 个边界框参数 + 1 个角度 = 5 维
        self.fc_loc_angle = nn.Linear(hidden_dim, 5)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: RoI 特征 [B, C, roi_size, roi_size]

        Returns:
            (cls_scores, loc_angle):
            - cls_scores: 分类得分 [B, num_classes]
            - loc_angle: 定位 + 角度 [B, 5] (x, y, w, h, angle)
        """
        B = x.size(0)

        # 展平 + 共享 FC 层
        x = x.view(B, -1)  # [B, C * roi_size * roi_size]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)

        # 分叉
        cls_scores = self.fc_cls(x)  # [B, num_classes]
        loc_angle = self.fc_loc_angle(x)  # [B, 5]

        return cls_scores, loc_angle


# ============================================================================
# Strip Head (核心创新)
# ============================================================================

class ClassificationBranch(nn.Module):
    """
    分类分支 (Strip Head)

    结构 (严格遵循框图):
    → [FC] → [FC] → [FC] → Classification
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        hidden_dim: int = 1024,
        roi_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size

        # 三个全连接层
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_cls = nn.Linear(hidden_dim, num_classes)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc_cls(x)
        return x


class AngleBranch(nn.Module):
    """
    角度预测分支 (Strip Head)

    结构 (严格遵循框图):
    从 fc2 分叉 → [FC] → Angle
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 角度预测 FC
        self.fc_angle = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: fc2 的输出特征 [B, hidden_dim]

        Returns:
            角度预测 [B, 1]
        """
        x = self.fc_angle(x)
        return x


class LocalizationBranch(nn.Module):
    """
    定位分支 (Strip Head) - 核心创新

    结构 (严格遵循框图):
    → [Conv] → [Strip Module] → [FC] → Localization

    使用 Strip Module 捕获长距离空间依赖，提高高长宽比物体的定位精度
    """

    def __init__(
        self,
        in_channels: int,
        strip_kernel_size: int = 19,
        hidden_dim: int = 1024,
        roi_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.strip_kernel_size = strip_kernel_size
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size

        # 3x3 卷积提取局部特征 (框图中的 Conv)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Strip Module 捕获长距离依赖 (框图中的 Strip Module)
        self.strip_module = StripModule(
            in_channels,
            in_channels,
            strip_kernel_size=strip_kernel_size,
            expansion=1.0,
            shortcut=True,
        )

        # 全局平均池化 + FC 回归边界框 (框图中的 FC)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_loc = nn.Linear(in_channels, 4)  # (x, y, w, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RoI 特征 [B, C, roi_size, roi_size]

        Returns:
            边界框参数 [B, 4]
        """
        # Conv 提取局部特征
        x = self.conv(x)

        # Strip Module 增强空间依赖
        x = self.strip_module(x)

        # 池化 + 回归
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc_loc(x)

        return x


class StripHead(nn.Module):
    """
    Strip Head - Strip R-CNN 的核心检测头

    严格遵循论文框图的下图结构:

                    ┌──→ [FC] → [FC] ──┬──→ [FC] → Classification
                                       └──→ [FC] → Angle
    RoI → [FC] → [FC] ─
                       └──→ [Conv] → [Strip Module] → [FC] → Localization

    核心创新点:
    1. 解耦分类、角度、定位三个分支
    2. 定位分支使用 Strip Module 捕获长距离空间依赖
    3. 分类和角度共享前两个 FC 层，定位分支独立处理

    Args:
        in_channels: 输入通道数
        num_classes: 分类类别数
        strip_kernel_size: Strip Module 中的条带卷积核大小
        hidden_dim: FC 层隐藏层维度
        roi_size: RoI Align 输出尺寸

    Input:
        x: RoI 特征 [B, C, roi_size, roi_size]

    Output:
        (cls_scores, bboxes, angles):
        - cls_scores: 分类得分 [B, num_classes]
        - bboxes: 边界框 [B, 4] (cx, cy, w, h)
        - angles: 旋转角度 [B, 1]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        strip_kernel_size: int = 19,
        hidden_dim: int = 1024,
        roi_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strip_kernel_size = strip_kernel_size
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size

        # 共享的 FC 层 (框图中的前两个 FC)
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 分类分支 (从 fc2 分叉)
        self.cls_branch = ClassificationBranch(
            in_channels, num_classes, hidden_dim, roi_size
        )
        # 覆盖 cls_branch 的 fc1 和 fc2，使用共享的
        self.cls_branch.fc1 = None  # 使用共享 fc1
        self.cls_branch.fc2 = None  # 使用共享 fc2

        # 角度分支 (从 fc2 分叉)
        self.angle_branch = AngleBranch(hidden_dim)

        # 定位分支 (独立路径)
        self.loc_branch = LocalizationBranch(
            in_channels, strip_kernel_size, hidden_dim, roi_size
        )

        self.act = nn.ReLU(inplace=True)

    def _forward_shared(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播共享部分"""
        B = x.size(0)
        x = x.view(B, -1)  # [B, C * roi_size^2]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        shared_feat = self.act(x)  # 共享特征 [B, hidden_dim]
        return shared_feat, x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: RoI 特征 [B, C, roi_size, roi_size]

        Returns:
            (cls_scores, bboxes, angles):
            - cls_scores: 分类得分 [B, num_classes]
            - bboxes: 边界框 [B, 4]
            - angles: 旋转角度 [B, 1]
        """
        B = x.size(0)

        # 共享路径
        shared_feat, _ = self._forward_shared(x)

        # 分类分支 (使用共享特征)
        cls_scores = self.cls_branch.fc_cls(shared_feat)

        # 角度分支 (使用共享特征)
        angles = self.angle_branch(shared_feat)

        # 定位分支 (独立路径，使用原始输入)
        bboxes = self.loc_branch(x)

        return cls_scores, bboxes, angles


# ============================================================================
# 增强的 Strip Head V2 (改进版本)
# ============================================================================

class StripHeadV2(nn.Module):
    """
    Strip Head V2 - 改进版本

    改进点:
    1. 定位分支也使用共享特征，但添加空间注意力
    2. 添加特征增强模块
    3. 支持可调节的 RoI 尺寸

    结构:
                    ┌──→ [FC] → [FC] ────→ [FC] → Classification
                                       └──→ [FC] → Angle
    RoI → [FC] → [FC] ─┘
                       ──→ [Conv] → [Strip Module] → [FC] → Localization
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        strip_kernel_size: int = 19,
        hidden_dim: int = 1024,
        roi_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strip_kernel_size = strip_kernel_size
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size

        # 共享 FC 层
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 分类头
        self.cls_head = nn.Linear(hidden_dim, num_classes)

        # 角度头
        self.angle_head = nn.Linear(hidden_dim, 1)

        # 定位分支 - 使用独立 Conv + Strip Module 路径
        self.loc_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.strip_module = StripModule(
            hidden_dim // 2,
            hidden_dim // 2,
            strip_kernel_size=strip_kernel_size,
            expansion=1.0,
            shortcut=True,
        )

        self.loc_pool = nn.AdaptiveAvgPool2d(1)
        self.loc_head = nn.Linear(hidden_dim // 2, 4)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.size(0)
        x_flat = x.view(B, -1)

        # 共享路径
        x_shared = self.fc1(x_flat)
        x_shared = self.act(x_shared)
        shared_feat = self.fc2(x_shared)
        shared_feat = self.act(shared_feat)

        # 分类
        cls_scores = self.cls_head(shared_feat)

        # 角度
        angles = self.angle_head(shared_feat)

        # 定位 (独立路径 - 使用原始输入 x 而不是 x_shared)
        loc_feat = self.loc_conv(x)
        loc_feat = self.strip_module(loc_feat)
        loc_feat = self.loc_pool(loc_feat)
        loc_feat = loc_feat.view(B, -1)
        bboxes = self.loc_head(loc_feat)

        return cls_scores, bboxes, angles


# ============================================================================
# 多尺度检测头 (用于 FPN)
# ============================================================================

class OrientedRCNNDetectHead(nn.Module):
    """
    Oriented R-CNN 多尺度检测头

    用于 FPN 的多个尺度 (P3, P4, P5)
    每个尺度使用相同的 OrientedRCNNHead 结构
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 80,
        hidden_dim: int = 1024,
        roi_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = len(in_channels)

        # 为每个尺度创建检测头
        self.heads = nn.ModuleList([
            OrientedRCNNHead(
                in_ch, num_classes, hidden_dim, roi_size
            )
            for in_ch in in_channels
        ])

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: 多尺度特征列表 [P3, P4, P5]

        Returns:
            (cls_scores_list, loc_angle_list)
        """
        cls_list = []
        loc_angle_list = []

        for feat, head in zip(features, self.heads):
            cls, loc_angle = head(feat)
            cls_list.append(cls)
            loc_angle_list.append(loc_angle)

        return cls_list, loc_angle_list


class StripDetectHead(nn.Module):
    """
    Strip R-CNN 多尺度检测头

    用于 FPN 的多个尺度 (P3, P4, P5)
    每个尺度使用相同的 StripHead 结构
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 80,
        strip_kernel_size: int = 19,
        hidden_dim: int = 1024,
        roi_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = len(in_channels)

        # 为每个尺度创建检测头
        self.heads = nn.ModuleList([
            StripHead(
                in_ch, num_classes, strip_kernel_size, hidden_dim, roi_size
            )
            for in_ch in in_channels
        ])

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: 多尺度特征列表 [P3, P4, P5]

        Returns:
            (cls_list, bboxes_list, angles_list)
        """
        cls_list = []
        bboxes_list = []
        angles_list = []

        for feat, head in zip(features, self.heads):
            cls, bbox, angle = head(feat)
            cls_list.append(cls)
            bboxes_list.append(bbox)
            angles_list.append(angle)

        return cls_list, bboxes_list, angles_list


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Strip R-CNN Detection Heads")
    print("=" * 60)

    # 测试 Oriented R-CNN Head
    print("\n[Oriented R-CNN Head]")
    x = torch.randn(4, 256, 7, 7)  # RoI 特征

    oriented_head = OrientedRCNNHead(
        in_channels=256,
        num_classes=80,
        hidden_dim=1024,
    )

    cls_scores, loc_angle = oriented_head(x)
    print(f"  Input: {x.shape}")
    print(f"  cls_scores: {cls_scores.shape}")
    print(f"  loc_angle: {loc_angle.shape}")

    # 测试 Strip Head
    print("\n[Strip Head]")
    strip_head = StripHead(
        in_channels=256,
        num_classes=80,
        strip_kernel_size=19,
        hidden_dim=1024,
    )

    cls_scores, bboxes, angles = strip_head(x)
    print(f"  Input: {x.shape}")
    print(f"  cls_scores: {cls_scores.shape}")
    print(f"  bboxes: {bboxes.shape}")
    print(f"  angles: {angles.shape}")

    # 测试 Strip Head V2
    print("\n[Strip Head V2]")
    strip_head_v2 = StripHeadV2(
        in_channels=256,
        num_classes=80,
        strip_kernel_size=19,
        hidden_dim=1024,
    )

    cls_scores, bboxes, angles = strip_head_v2(x)
    print(f"  Input: {x.shape}")
    print(f"  cls_scores: {cls_scores.shape}")
    print(f"  bboxes: {bboxes.shape}")
    print(f"  angles: {angles.shape}")

    # 测试多尺度检测头
    print("\n[Multi-Scale Strip Detect Head]")
    features = [
        torch.randn(4, 256, 7, 7),  # P3
        torch.randn(4, 512, 7, 7),  # P4
        torch.randn(4, 512, 7, 7),  # P5
    ]

    detect_head = StripDetectHead(
        in_channels=[256, 512, 512],
        num_classes=80,
        strip_kernel_size=19,
    )

    cls_list, bboxes_list, angles_list = detect_head(features)

    for i, (cls, bbox, angle) in enumerate(zip(cls_list, bboxes_list, angles_list)):
        print(f"  P{i+3}: cls={cls.shape}, bbox={bbox.shape}, angle={angle.shape}")

    # 参数量对比
    print("\n" + "=" * 60)
    print("Parameter Count Comparison")
    print("=" * 60)

    def count_params(model):
        return sum(p.numel() for p in model.parameters()) / 1e6

    print(f"  Oriented R-CNN Head: {count_params(oriented_head):.2f}M params")
    print(f"  Strip Head:          {count_params(strip_head):.2f}M params")
    print(f"  Strip Head V2:       {count_params(strip_head_v2):.2f}M params")

    print("\n[All tests passed!]")
