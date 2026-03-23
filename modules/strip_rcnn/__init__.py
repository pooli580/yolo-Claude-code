# Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
# Paper: https://github.com/HVision-NKU/Strip-R-CNN
#
# 核心创新:
# 1. Large Strip Convolution - 使用正交的大条带卷积捕获高长宽比物体特征
# 2. StripNet Backbone - 简单高效的主干网络
# 3. Strip Head - 解耦的检测头，定位分支使用 strip 卷积

from .strip_conv import StripConv, LargeStripConv
from .strip_module import StripModule
from .strip_net import StripNet, strip_net_tiny, strip_net_small, strip_net_base
from .strip_head import StripHead

__all__ = [
    'StripConv',
    'LargeStripConv',
    'StripModule',
    'StripNet',
    'strip_net_tiny',
    'strip_net_small',
    'strip_net_base',
    'StripHead',
]
