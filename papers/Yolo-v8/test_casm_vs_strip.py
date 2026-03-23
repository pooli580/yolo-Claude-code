"""
对比测试：CASM vs Strip 模块
测试 Connectivity Attention Strip Module 和原有 Strip Convolution 的性能
"""

import torch
import torch.nn as nn
import time

# 导入 CASM 模块
from connectivity_attention import (
    CASM,
    CASMBlock,
    C2f_CASM,
    SPPF_CASM,
    QuadDirectionStripConv,
    ConnectivityAttention,
)

# 导入原有 Strip 模块
from strip_conv import (
    StripConv,
    StripModule,
    StripC2f,
    LargeStripConv,
)


def count_parameters(model):
    """计算模型参数量 (百万)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def test_speed(model, input_tensor, warmup=10, iterations=100):
    """测试推理速度"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # 计时
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    end = time.time()

    avg_time = (end - start) / iterations * 1000  # ms
    return avg_time


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f" {title} ")
    print("=" * 70)


def print_separator():
    """打印分隔线"""
    print("-" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    print_header("CASM vs Strip 模块对比测试")

    # ============== 测试 1: 基础条带卷积 ==============
    print_header("测试 1: 基础条带卷积对比")

    x = torch.randn(1, 64, 56, 56)

    # StripConv (原有)
    strip_conv = StripConv(64, 64, strip_kernel_size=19)
    out_strip = strip_conv(x)
    params_strip = count_parameters(strip_conv)
    speed_strip = test_speed(strip_conv, x)

    # QuadDirectionStripConv (CASM)
    qd_conv = QuadDirectionStripConv(64, 64, strip_kernel_size=19)
    out_qd = qd_conv(x)
    params_qd = count_parameters(qd_conv)
    speed_qd = test_speed(qd_conv, x)

    print(f"\n输入：{tuple(x.shape)}\n")
    print(f"{'模块':<35} {'输出':<20} {'参数量 (M)':<12} {'速度 (ms)':<12}")
    print_separator()
    print(f"{'StripConv (原有)':<35} {str(tuple(out_strip.shape)):<20} {params_strip:<12.3f} {speed_strip:<12.2f}")
    print(f"{'QuadDirectionStripConv (CASM)':<35} {str(tuple(out_qd.shape)):<20} {params_qd:<12.3f} {speed_qd:<12.2f}")

    # ============== 测试 2: 基础模块对比 ==============
    print_header("测试 2: 基础模块对比")

    x2 = torch.randn(1, 128, 28, 28)

    # StripModule (原有)
    strip_module = StripModule(128, 128, strip_kernel_size=19)
    out_sm = strip_module(x2)
    params_sm = count_parameters(strip_module)
    speed_sm = test_speed(strip_module, x2)

    # CASM (新)
    casm = CASM(128, 128, strip_kernel_size=19)
    out_casm = casm(x2)
    params_casm = count_parameters(casm)
    speed_casm = test_speed(casm, x2)

    print(f"\n输入：{tuple(x2.shape)}\n")
    print(f"{'模块':<35} {'输出':<20} {'参数量 (M)':<12} {'速度 (ms)':<12}")
    print_separator()
    print(f"{'StripModule (原有)':<35} {str(tuple(out_sm.shape)):<20} {params_sm:<12.3f} {speed_sm:<12.2f}")
    print(f"{'CASM (新)':<35} {str(tuple(out_casm.shape)):<20} {params_casm:<12.3f} {speed_casm:<12.2f}")

    # ============== 测试 3: CASMBlock 测试 ==============
    print_header("测试 3: CASMBlock (新模块)")

    x3 = torch.randn(1, 256, 14, 14)

    casm_block = CASMBlock(256, 256, strip_kernel_size=19)
    out_block = casm_block(x3)
    params_block = count_parameters(casm_block)
    speed_block = test_speed(casm_block, x3)

    print(f"\n输入：{tuple(x3.shape)}\n")
    print(f"{'模块':<35} {'输出':<20} {'参数量 (M)':<12} {'速度 (ms)':<12}")
    print_separator()
    print(f"{'CASMBlock':<35} {str(tuple(out_block.shape)):<20} {params_block:<12.3f} {speed_block:<12.2f}")

    # ============== 测试 4: C2f 变体对比 ==============
    print_header("测试 4: C2f 变体对比")

    x4 = torch.randn(1, 256, 14, 14)

    # StripC2f (原有)
    strip_c2f = StripC2f(256, 256, num_blocks=3, strip_kernel_size=19)
    out_sc2f = strip_c2f(x4)
    params_sc2f = count_parameters(strip_c2f)
    speed_sc2f = test_speed(strip_c2f, x4)

    # C2f_CASM (新)
    c2f_casm = C2f_CASM(256, 256, num_blocks=3, strip_kernel_size=19)
    out_c2f = c2f_casm(x4)
    params_c2f = count_parameters(c2f_casm)
    speed_c2f = test_speed(c2f_casm, x4)

    print(f"\n输入：{tuple(x4.shape)}\n")
    print(f"{'模块':<35} {'输出':<20} {'参数量 (M)':<12} {'速度 (ms)':<12}")
    print_separator()
    print(f"{'StripC2f (原有)':<35} {str(tuple(out_sc2f.shape)):<20} {params_sc2f:<12.3f} {speed_sc2f:<12.2f}")
    print(f"{'C2f_CASM (新)':<35} {str(tuple(out_c2f.shape)):<20} {params_c2f:<12.3f} {speed_c2f:<12.2f}")

    # ============== 测试 5: SPPF 变体对比 ==============
    print_header("测试 5: SPPF 变体对比")

    x5 = torch.randn(1, 512, 7, 7)

    # 原有 SPPF (本地实现)
    class SPPF(nn.Module):
        def __init__(self, c1, c2, k=5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, 1, bias=False),
                                     nn.BatchNorm2d(c_), nn.GELU())
            self.cv2 = nn.Sequential(nn.Conv2d(c_ * 4, c2, 1, bias=False),
                                     nn.BatchNorm2d(c2), nn.GELU())
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

    sppf = SPPF(512, 512)
    out_sppf = sppf(x5)
    params_sppf = count_parameters(sppf)
    speed_sppf = test_speed(sppf, x5)

    # SPPF_CASM (新)
    sppf_casm = SPPF_CASM(512, 512, strip_kernel_size=19)
    out_sppf_casm = sppf_casm(x5)
    params_sppf_casm = count_parameters(sppf_casm)
    speed_sppf_casm = test_speed(sppf_casm, x5)

    print(f"\n输入：{tuple(x5.shape)}\n")
    print(f"{'模块':<35} {'输出':<20} {'参数量 (M)':<12} {'速度 (ms)':<12}")
    print_separator()
    print(f"{'SPPF (原有)':<35} {str(tuple(out_sppf.shape)):<20} {params_sppf:<12.3f} {speed_sppf:<12.2f}")
    print(f"{'SPPF_CASM (新)':<35} {str(tuple(out_sppf_casm.shape)):<20} {params_sppf_casm:<12.3f} {speed_sppf_casm:<12.2f}")

    # ============== 测试 6: Connectivity Attention 分析 ==============
    print_header("测试 6: Connectivity Attention 分析")

    # 使用较大的 batch size 避免 BatchNorm 问题
    ca = ConnectivityAttention(64, strip_kernel_size=19)
    ca.eval()  # 设置为评估模式

    # 测试四方向特征
    h_feat = torch.randn(2, 64, 28, 28)
    v_feat = torch.randn(2, 64, 28, 28)
    d_feat = torch.randn(2, 64, 28, 28)

    with torch.no_grad():
        h_out, v_out, d_out = ca(h_feat, v_feat, d_feat)

    params_ca = count_parameters(ca)

    print(f"\n输入：{tuple(h_feat.shape)} x 3\n")
    print(f"Connectivity Attention 输出:")
    print(f"  水平方向：{tuple(h_out.shape)}")
    print(f"  垂直方向：{tuple(v_out.shape)}")
    print(f"  对角线方向：{tuple(d_out.shape)}")
    print(f"\n参数量：{params_ca:.3f} M")

    # ============== 总结 ==============
    print_header("测试总结")

    print("\n参数量对比:")
    print(f"  StripConv         -> {params_strip:.3f} M")
    print(f"  QuadDirectionStrip-> {params_qd:.3f} M")
    print(f"  StripModule       -> {params_sm:.3f} M")
    print(f"  CASM              -> {params_casm:.3f} M")
    print(f"  CASMBlock         -> {params_block:.3f} M")
    print(f"  StripC2f          -> {params_sc2f:.3f} M")
    print(f"  C2f_CASM          -> {params_c2f:.3f} M")
    print(f"  SPPF              -> {params_sppf:.3f} M")
    print(f"  SPPF_CASM         -> {params_sppf_casm:.3f} M")

    print("\n速度对比 (ms):")
    print(f"  StripConv         -> {speed_strip:.2f} ms")
    print(f"  QuadDirectionStrip-> {speed_qd:.2f} ms")
    print(f"  StripModule       -> {speed_sm:.2f} ms")
    print(f"  CASM              -> {speed_casm:.2f} ms")
    print(f"  CASMBlock         -> {speed_block:.2f} ms")
    print(f"  StripC2f          -> {speed_sc2f:.2f} ms")
    print(f"  C2f_CASM          -> {speed_c2f:.2f} ms")
    print(f"  SPPF              -> {speed_sppf:.2f} ms")
    print(f"  SPPF_CASM         -> {speed_sppf_casm:.2f} ms")

    print("\n[OK] 所有测试完成!")
