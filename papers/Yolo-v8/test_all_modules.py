"""
YOLOv8-Strip 项目 - 综合测试脚本
测试所有已实现的模块

运行方式:
    python test_all_modules.py

预期输出:
    所有模块测试通过后显示 [✓] 所有测试通过!
"""

import torch
import sys
from typing import Dict, List, Tuple

# 结果记录
test_results = {
    'passed': [],
    'failed': [],
    'errors': [],
}

def record_result(module_name: str, success: bool, error_msg: str = None):
    """记录测试结果"""
    if success:
        test_results['passed'].append(module_name)
        print(f"  [OK] {module_name}")
    else:
        test_results['failed'].append(module_name)
        print(f"  [FAIL] {module_name}: {error_msg}")


def test_strip_conv():
    """测试 Strip 卷积模块"""
    print("\n" + "=" * 60)
    print("1. Testing Strip Convolution Modules")
    print("=" * 60)

    try:
        from strip_conv import LargeStripConv, StripConv, StripModule, StripC2f

        x = torch.randn(2, 64, 56, 56)

        # LargeStripConv
        conv = LargeStripConv(64, kernel_size=19)
        h_feat, v_feat = conv(x)
        record_result("LargeStripConv", h_feat.shape == v_feat.shape == x.shape)

        # StripConv
        conv_full = StripConv(64, 64, strip_kernel_size=19)
        out = conv_full(x)
        record_result("StripConv", out.shape == x.shape)

        # StripModule
        module = StripModule(64, 64, strip_kernel_size=19)
        out = module(x)
        record_result("StripModule", out.shape == x.shape)

        # StripC2f
        c2f = StripC2f(64, 128, num_blocks=3, strip_kernel_size=19)
        out = c2f(x)
        record_result("StripC2f", out.shape == (2, 128, 56, 56))

    except Exception as e:
        record_result("Strip Conv Modules", False, str(e))


def test_connectivity_attention():
    """测试 Connectivity Attention 模块"""
    print("\n" + "=" * 60)
    print("2. Testing Connectivity Attention Modules")
    print("=" * 60)

    try:
        from connectivity_attention import (
            HorizontalStripConv, VerticalStripConv,
            LeftDiagonalStripConv, RightDiagonalStripConv,
            QuadDirectionStripConv, ConnectivityAttention,
            CASM, CASMBlock, C2f_CASM, SPPF_CASM
        )

        x = torch.randn(2, 64, 56, 56)

        # 单方向条带卷积
        h_strip = HorizontalStripConv(64, kernel_size=19)
        out = h_strip(x)
        record_result("HorizontalStripConv", out.shape == x.shape)

        v_strip = VerticalStripConv(64, kernel_size=19)
        out = v_strip(x)
        record_result("VerticalStripConv", out.shape == x.shape)

        # 四方向条带卷积
        qd_conv = QuadDirectionStripConv(64, 64, strip_kernel_size=19)
        out = qd_conv(x)
        record_result("QuadDirectionStripConv", out.shape == x.shape)

        # Connectivity Attention
        ca = ConnectivityAttention(64, strip_kernel_size=19)
        h_feat = torch.randn(2, 64, 28, 28)
        v_feat = torch.randn(2, 64, 28, 28)
        d_feat = torch.randn(2, 64, 28, 28)
        h_out, v_out, d_out = ca(h_feat, v_feat, d_feat)
        record_result("ConnectivityAttention",
                      h_out.shape == v_out.shape == d_out.shape == h_feat.shape)

        # CASM
        casm = CASM(64, 64, strip_kernel_size=19)
        out = casm(x)
        record_result("CASM", out.shape == x.shape)

        # CASMBlock
        block = CASMBlock(128, 128, strip_kernel_size=19)
        x2 = torch.randn(2, 128, 28, 28)
        out = block(x2)
        record_result("CASMBlock", out.shape == x2.shape)

        # C2f_CASM
        c2f = C2f_CASM(256, 256, num_blocks=3, strip_kernel_size=19)
        x3 = torch.randn(2, 256, 14, 14)
        out = c2f(x3)
        record_result("C2f_CASM", out.shape == x3.shape)

        # SPPF_CASM
        sppf = SPPF_CASM(512, 512, strip_kernel_size=19)
        x4 = torch.randn(2, 512, 7, 7)
        out = sppf(x4)
        record_result("SPPF_CASM", out.shape == x4.shape)

    except Exception as e:
        record_result("Connectivity Attention Modules", False, str(e))


def test_dysample():
    """测试 DySample 动态上采样模块"""
    print("\n" + "=" * 60)
    print("3. Testing DySample Modules")
    print("=" * 60)

    try:
        from dysample_upsample import (
            DySample, DUPSBlock2D, DUPSBlock1D,
            DynamicUpsample, YOLOUpsample
        )

        x = torch.randn(2, 64, 32, 32)

        # DySample Static
        dysample_s = DySample(64, scale_factor=2, mode='static')
        out = dysample_s(x)
        record_result("DySample (static)", out.shape == (2, 64, 64, 64))

        # DySample Dynamic
        dysample_d = DySample(64, scale_factor=2, mode='dynamic')
        out = dysample_d(x)
        record_result("DySample (dynamic)", out.shape == (2, 64, 64, 64))

        # DUPSBlock2D
        dups2d = DUPSBlock2D(64, 128, scale_factor=2)
        out = dups2d(x)
        record_result("DUPSBlock2D", out.shape == (2, 128, 64, 64))

        # DUPSBlock1D
        dups1d = DUPSBlock1D(64, 128, scale_factor=2, kernel_size=7)
        out = dups1d(x)
        record_result("DUPSBlock1D", out.shape == (2, 128, 64, 64))

        # DynamicUpsample
        dyn_up = DynamicUpsample(64, 128, scale_factor=2, method='dysample')
        out = dyn_up(x)
        record_result("DynamicUpsample", out.shape == (2, 128, 64, 64))

        # YOLOUpsample
        yolo_up = YOLOUpsample(64, 128, scale_factor=2, use_dynamic=True)
        out = yolo_up(x)
        record_result("YOLOUpsample", out.shape == (2, 128, 64, 64))

    except Exception as e:
        record_result("DySample Modules", False, str(e))


def test_feature_fusion():
    """测试特征融合模块"""
    print("\n" + "=" * 60)
    print("4. Testing Feature Fusion Modules")
    print("=" * 60)

    try:
        from feature_fusion import (
            FeatureCalibrationFusion, FeatureCalibrationFusionV2,
            CrossFieldFrequencyFusion, CrossFieldFrequencyFusionSimple,
            FCF_CFFF_Block
        )

        # FCF
        fcf = FeatureCalibrationFusion(in_channels_low=256, in_channels_high=512)
        low_feat = torch.randn(2, 256, 64, 64)
        high_feat = torch.randn(2, 512, 32, 32)
        out = fcf(low_feat, high_feat)
        record_result("FeatureCalibrationFusion", out.shape == (2, 256, 64, 64))

        # FCF V2
        fcf_v2 = FeatureCalibrationFusionV2(in_channels_low=256, in_channels_high=512)
        out = fcf_v2(low_feat, high_feat)
        record_result("FeatureCalibrationFusionV2", out.shape == (2, 256, 64, 64))

        # CFFF
        cfff = CrossFieldFrequencyFusion(in_channels=256, reduction=4)
        feat_r = torch.randn(2, 256, 64, 64)
        feat_n = torch.randn(2, 256, 64, 64)
        out = cfff(feat_r, feat_n)
        record_result("CrossFieldFrequencyFusion", out.shape == (2, 256, 64, 64))

        # CFFF Simple
        cfff_simple = CrossFieldFrequencyFusionSimple(256)
        out = cfff_simple(feat_r, feat_n)
        record_result("CrossFieldFrequencyFusionSimple", out.shape == (2, 256, 64, 64))

        # FCF+CFFF Block
        block = FCF_CFFF_Block(
            channels_p3=256, channels_p4=512, channels_p5=512,
            use_cfff=True,
        )
        p3 = torch.randn(2, 256, 64, 64)
        p4 = torch.randn(2, 512, 32, 32)
        p5 = torch.randn(2, 512, 16, 16)
        p3_out, p4_out, p5_out = block(p3, p4, p5)
        record_result("FCF_CFFF_Block",
                      p3_out.shape == (2, 256, 64, 64) and
                      p4_out.shape == (2, 512, 32, 32) and
                      p5_out.shape == (2, 512, 16, 16))

    except Exception as e:
        record_result("Feature Fusion Modules", False, str(e))


def test_strip_rcnn_head():
    """测试 Strip R-CNN 检测头"""
    print("\n" + "=" * 60)
    print("5. Testing Strip R-CNN Detection Heads")
    print("=" * 60)

    try:
        from strip_rcnn_head import (
            OrientedRCNNHead, StripHead, StripHeadV2,
            StripDetectHead
        )

        roi_feat = torch.randn(4, 256, 7, 7)

        # Oriented R-CNN Head
        oriented_head = OrientedRCNNHead(in_channels=256, num_classes=80)
        cls_scores, loc_angle = oriented_head(roi_feat)
        record_result("OrientedRCNNHead",
                      cls_scores.shape == (4, 80) and loc_angle.shape == (4, 5))

        # Strip Head
        strip_head = StripHead(in_channels=256, num_classes=80, strip_kernel_size=19)
        cls_scores, bboxes, angles = strip_head(roi_feat)
        record_result("StripHead",
                      cls_scores.shape == (4, 80) and
                      bboxes.shape == (4, 4) and
                      angles.shape == (4, 1))

        # Strip Head V2
        strip_head_v2 = StripHeadV2(in_channels=256, num_classes=80, strip_kernel_size=19)
        cls_scores, bboxes, angles = strip_head_v2(roi_feat)
        record_result("StripHeadV2",
                      cls_scores.shape == (4, 80) and
                      bboxes.shape == (4, 4) and
                      angles.shape == (4, 1))

        # Multi-Scale Detect Head
        features = [
            torch.randn(4, 256, 7, 7),
            torch.randn(4, 512, 7, 7),
            torch.randn(4, 512, 7, 7),
        ]
        detect_head = StripDetectHead(
            in_channels=[256, 512, 512],
            num_classes=80,
            strip_kernel_size=19,
        )
        cls_list, bboxes_list, angles_list = detect_head(features)
        record_result("StripDetectHead",
                      len(cls_list) == 3 and
                      all(t.shape == (4, 80) for t in cls_list))

    except Exception as e:
        record_result("Strip R-CNN Heads", False, str(e))


def test_yolov8_strip():
    """测试完整 YOLOv8-Strip 模型"""
    print("\n" + "=" * 60)
    print("6. Testing Complete YOLOv8-Strip Models")
    print("=" * 60)

    try:
        from yolov8_strip import yolov8n_strip, yolov8s_strip, count_parameters

        x = torch.randn(1, 3, 640, 640)

        # YOLOv8n-Strip
        model_n = yolov8n_strip(num_classes=80)
        outputs = model_n(x)
        record_result("YOLOv8n-Strip",
                      len(outputs['cls']) == 3 and
                      all(t.shape[0] == 1 for t in outputs['cls']))

        # YOLOv8s-Strip
        model_s = yolov8s_strip(num_classes=80)
        outputs = model_s(x)
        record_result("YOLOv8s-Strip",
                      len(outputs['cls']) == 3 and
                      all(t.shape[0] == 1 for t in outputs['cls']))

    except Exception as e:
        record_result("YOLOv8-Strip Models", False, str(e))


def print_summary():
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = len(test_results['passed']) + len(test_results['failed'])
    passed = len(test_results['passed'])
    failed = len(test_results['failed'])

    print(f"\nTotal Tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if test_results['passed']:
        print(f"\n[OK] All modules tested successfully!" if failed == 0 else
              f"\nPartial success: {passed}/{total} tests passed")

    if failed > 0:
        print("\nFailed tests:")
        for name in test_results['failed']:
            print(f"  - {name}")

    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8-Strip Project - Comprehensive Module Tests")
    print("=" * 60)

    # 运行所有测试
    test_strip_conv()
    test_connectivity_attention()
    test_dysample()
    test_feature_fusion()
    test_strip_rcnn_head()
    test_yolov8_strip()

    # 打印总结
    all_passed = print_summary()

    sys.exit(0 if all_passed else 1)
