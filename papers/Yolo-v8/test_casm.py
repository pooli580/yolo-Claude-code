"""
测试 Connectivity Attention Strip Module (CASM)
"""

import torch
import torch.nn as nn
from connectivity_attention import (
    HorizontalStripConv,
    VerticalStripConv,
    LeftDiagonalStripConv,
    RightDiagonalStripConv,
    QuadDirectionStripConv,
    ConnectivityAttention,
    CASM,
    CASMBlock,
    C2f_CASM,
    SPPF_CASM,
)


def autopad(k, p=None, d=1):
    """Pad to 'same' shape"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_module(name, module, input_shape):
    """测试模块"""
    x = torch.randn(*input_shape)

    # 前向传播
    with torch.no_grad():
        out = module(x)

    # 计算参数量
    params = count_parameters(module)

    print(f"{name}:")
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape) if isinstance(out, torch.Tensor) else tuple(o.shape for o in out)}")
    print(f"  Params: {params:,}")
    print()

    return out


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Connectivity Attention Strip Module (CASM)")
    print("=" * 60)
    print()

    # 测试单方向条带卷积
    print("1. Single Direction Strip Convolutions")
    print("-" * 40)

    test_module("  HorizontalStripConv", HorizontalStripConv(64, kernel_size=19), (2, 64, 56, 56))
    test_module("  VerticalStripConv", VerticalStripConv(64, kernel_size=19), (2, 64, 56, 56))
    test_module("  LeftDiagonalStripConv", LeftDiagonalStripConv(64, kernel_size=19), (2, 64, 56, 56))
    test_module("  RightDiagonalStripConv", RightDiagonalStripConv(64, kernel_size=19), (2, 64, 56, 56))

    # 测试四方向条带卷积
    print("2. Quad-Direction Strip Convolution")
    print("-" * 40)
    test_module("  QuadDirectionStripConv", QuadDirectionStripConv(64, 64, strip_kernel_size=19), (2, 64, 56, 56))

    # 测试 Connectivity Attention
    print("3. Connectivity Attention")
    print("-" * 40)
    ca = ConnectivityAttention(64, strip_kernel_size=19)
    h_feat = torch.randn(2, 64, 28, 28)
    v_feat = torch.randn(2, 64, 28, 28)
    d_feat = torch.randn(2, 64, 28, 28)

    with torch.no_grad():
        h_out, v_out, d_out = ca(h_feat, v_feat, d_feat)

    print(f"  Input:  {tuple(h_feat.shape)} x3")
    print(f"  Output: {tuple(h_out.shape)} x3")
    print(f"  Params: {count_parameters(ca):,}")
    print()

    # 测试 CASM
    print("4. CASM (Connectivity Attention Strip Module)")
    print("-" * 40)
    test_module("  CASM", CASM(64, 64, strip_kernel_size=19), (2, 64, 56, 56))

    # 测试 CASM Block
    print("5. CASM Block (for Backbone)")
    print("-" * 40)
    test_module("  CASMBlock", CASMBlock(128, 128, strip_kernel_size=19, shortcut=True), (2, 128, 28, 28))

    # 测试 C2f_CASM
    print("6. C2f_CASM (for Neck)")
    print("-" * 40)
    test_module("  C2f_CASM", C2f_CASM(256, 256, num_blocks=3, strip_kernel_size=19), (2, 256, 14, 14))

    # 测试 SPPF_CASM
    print("7. SPPF_CASM (for Backbone End)")
    print("-" * 40)
    test_module("  SPPF_CASM", SPPF_CASM(512, 512, strip_kernel_size=19), (2, 512, 7, 7))

    # 完整的 forward pass 测试
    print("=" * 60)
    print("Complete Forward Pass Test")
    print("=" * 60)

    # 模拟 YOLOv8 backbone 的 forward
    print("\nSimulating YOLOv8-CASM Backbone Forward Pass:")
    print("-" * 40)

    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640)

    # P1/2
    conv1 = Conv(3, 64, 3, 2)
    x = conv1(x)
    print(f"  P1/2:  {tuple(x.shape)}")

    # P2/4
    conv2 = Conv(64, 128, 3, 2)
    x = conv2(x)
    block1 = CASMBlock(128, 128, strip_kernel_size=19)
    x = block1(x)
    print(f"  P2/4:  {tuple(x.shape)}")

    # P3/8
    conv3 = Conv(128, 256, 3, 2)
    x = conv3(x)
    block2 = CASMBlock(256, 256, strip_kernel_size=19)
    x = block2(x)
    print(f"  P3/8:  {tuple(x.shape)}")

    # P4/16
    conv4 = Conv(256, 512, 3, 2)
    x = conv4(x)
    block3 = CASMBlock(512, 512, strip_kernel_size=19)
    x = block3(x)
    print(f"  P4/16: {tuple(x.shape)}")

    # P5/32
    conv5 = Conv(512, 512, 3, 2)
    x = conv5(x)
    block4 = CASMBlock(512, 512, strip_kernel_size=19)
    x = block4(x)
    print(f"  P5/32: {tuple(x.shape)}")

    # SPPF_CASM
    sppf_casm = SPPF_CASM(512, 512, strip_kernel_size=19)
    x = sppf_casm(x)
    print(f"  SPPF:  {tuple(x.shape)}")

    print("\n[OK] All tests passed!")
    print()
