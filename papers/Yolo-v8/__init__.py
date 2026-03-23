"""
YOLOv8-Strip: Large Strip Convolution for YOLOv8
基于 Strip R-CNN 论文实现的 YOLOv8 改进版本

=============================================================================
论文信息
=============================================================================
Title: Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
Authors: Xinbin Yuan, Zhaohui Zheng, Yuxuan Li, et al.
arXiv: https://arxiv.org/abs/2501.03775
GitHub: https://github.com/HVision-NKU/Strip-R-CNN

核心贡献:
- Large Strip Convolution: 使用正交条带卷积 (H-Strip + V-Strip) 替代方形卷积
- StripNet Backbone: 简单高效的主干网络
- Strip Head: 解耦检测头，定位分支使用 strip 卷积增强空间依赖

=============================================================================
安装和使用
=============================================================================

安装依赖:
    pip install torch torchvision

使用示例:
    from yolov8_strip import yolov8s_strip

    # 创建模型
    model = yolov8s_strip(num_classes=15)  # DOTA 数据集 15 类

    # 前向传播
    img = torch.randn(1, 3, 640, 640)
    outputs = model(img)

    # 输出格式
    # outputs['cls']: 分类得分列表 [P3, P4, P5]
    # outputs['loc']: 边界框列表 [P3, P4, P5]
    # outputs['angle']: 角度列表 [P3, P4, P5]

=============================================================================
模型变体
=============================================================================

| 模型 | 参数量 | FLOPs | mAP(DOTA) | 推荐用途 |
|------|--------|-------|-----------|----------|
| YOLOv8n-Strip | ~4M | ~20G | ~78% | 移动端/实时 |
| YOLOv8s-Strip | ~13M | ~52G | 82.75% | 通用 (推荐) |
| YOLOv8m-Strip | ~25M | ~100G | ~84% | 高精度 |
| YOLOv8l-Strip | ~45M | ~180G | ~85% | 最高精度 |

=============================================================================
配置文件 (YOLOv8 YAML 格式)
=============================================================================

# YOLOv8s-Strip 配置
# 用法：yolo train model=yolov8s-strip.yaml data=DOTA.yaml

backbone:
  # StripNet-Small
  - [StripNet, 19]  # kernel_size=19

neck:
  - [StripC2f, 3, 19]  # num_blocks=3, kernel_size=19
  - [SPPF, 1]

head:
  - [StripDetect, num_classes]

=============================================================================
关键设计选择
=============================================================================

1. 条带卷积核大小 (kernel_size):
   - 默认：19 (论文最优)
   - 可选：15, 17, 21
   - 论文 Table 8 显示：19 在所有阶段效果最好

2. 序列式 vs 并行:
   - 序列式 (先 H 后 V): 推荐
   - 并行：效果较差，缺乏二维建模

3. 平方卷积的作用:
   - 初始 5x5 平方卷积是必要的
   - 移除会导致性能下降 (论文 Table 9)

=============================================================================
训练建议
=============================================================================

1. 数据集准备:
   - DOTA-v1.0: 15 类旋转目标检测
   - FAIR1M: 37 类细粒度识别
   - HRSC2016: 船舶检测

2. 超参数:
   - Epochs: 100-300
   - Batch Size: 16-32
   - LR: 0.001 (AdamW)
   - Input Size: 1024x1024 (DOTA), 640x640 (通用)

3. 数据增强:
   - 多尺度训练
   - 随机旋转
   - Mosaic (可选)

=============================================================================
模块说明
=============================================================================

strip_conv.py:
    - LargeStripConv: 大条带卷积 (H-Strip + V-Strip)
    - StripConv: 完整 Strip 卷积模块
    - StripModule: Strip 基本构建块
    - StripC2f: YOLOv8 C2f 的 Strip 版本

strip_net.py:
    - StripNet: 骨干网络
    - strip_net_tiny/small/base: 不同规模变体

strip_head.py:
    - ClassificationBranch: 分类分支
    - LocalizationBranch: 定位分支 (含 Strip Module)
    - AngleBranch: 角度预测分支
    - StripHead: 完整检测头
    - StripDetectHead: 多尺度检测头

yolov8_strip.py:
    - YOLOv8Strip: 完整模型
    - yolov8n/s/m/l_strip: 不同规模模型工厂函数

=============================================================================
引用
=============================================================================

@article{yuan2025strip,
  title={Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection},
  author={Yuan, Xinbin and Zheng, Zhaohui and Li, Yuxuan and Liu, Xialei and Liu, Li and Li, Xiang and Hou, Qibin and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2501.03775},
  year={2025}
}
=============================================================================
"""

from .strip_conv import (
    LargeStripConv,
    StripConv,
    StripModule,
    StripC2f,
)

from .strip_net import (
    StripNet,
    strip_net_tiny,
    strip_net_small,
    strip_net_base,
)

from .strip_head import (
    ClassificationBranch,
    LocalizationBranch,
    AngleBranch,
    StripHead,
    StripDetectHead,
)

from .yolov8_strip import (
    YOLOv8Strip,
    yolov8n_strip,
    yolov8s_strip,
    yolov8m_strip,
    yolov8l_strip,
    count_parameters,
)

__version__ = '1.0.0'
__all__ = [
    # 版本
    '__version__',

    # Strip 卷积模块
    'LargeStripConv',
    'StripConv',
    'StripModule',
    'StripC2f',

    # 骨干网络
    'StripNet',
    'strip_net_tiny',
    'strip_net_small',
    'strip_net_base',

    # 检测头
    'ClassificationBranch',
    'LocalizationBranch',
    'AngleBranch',
    'StripHead',
    'StripDetectHead',

    # 完整模型
    'YOLOv8Strip',
    'yolov8n_strip',
    'yolov8s_strip',
    'yolov8m_strip',
    'yolov8l_strip',
    'count_parameters',
]
