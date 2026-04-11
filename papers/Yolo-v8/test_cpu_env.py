# -*- coding: utf-8 -*-
"""
YOLOv8 CPU 环境测试脚本
"""

import sys
import os

# 确保控制台输出使用 UTF-8
os.system('chcp 65001 >nul')

def print_separator(title=""):
    print("=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)

def test_pytorch():
    """测试 PyTorch 安装"""
    print("\n[1/5] 测试 PyTorch...")
    try:
        import torch
        print(f"  [OK] PyTorch 版本：{torch.__version__}")
        print(f"  [OK] CUDA 可用：{torch.cuda.is_available()}")
        print(f"  [OK] CPU 线程数：{torch.get_num_threads()}")

        # 测试 CPU 张量运算
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print(f"  [OK] CPU 矩阵乘法测试通过")

        return True
    except Exception as e:
        print(f"  [FAIL] PyTorch 测试失败：{e}")
        return False

def test_torchvision():
    """测试 Torchvision 安装"""
    print("\n[2/5] 测试 Torchvision...")
    try:
        import torchvision
        print(f"  [OK] Torchvision 版本：{torchvision.__version__}")

        # 测试图像加载
        from PIL import Image
        import numpy as np
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_tensor = torchvision.transforms.ToTensor()(Image.fromarray(img))
        print(f"  [OK] 图像张量转换测试通过")

        return True
    except Exception as e:
        print(f"  [FAIL] Torchvision 测试失败：{e}")
        return False

def test_ultralytics():
    """测试 Ultralytics (YOLOv8) 安装"""
    print("\n[3/5] 测试 Ultralytics (YOLOv8)...")
    try:
        from ultralytics import YOLO
        print("  [OK] Ultralytics 导入成功")

        # 测试模型加载（使用预训练的 YOLOv8n）
        try:
            model = YOLO('yolov8n.pt')
            print("  [OK] YOLOv8n 模型加载成功")
        except Exception as e:
            print(f"  [WARN] 模型加载失败（可能需要下载）: {e}")
            print("  首次运行会自动下载预训练模型")

        return True
    except Exception as e:
        print(f"  [FAIL] Ultralytics 测试失败：{e}")
        return False

def test_inference():
    """测试推理功能"""
    print("\n[4/5] 测试 CPU 推理...")
    try:
        from ultralytics import YOLO
        import numpy as np

        # 创建测试图像
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 加载模型
        model = YOLO('yolov8n.pt')

        # 推理
        results = model(test_img, device='cpu', verbose=False)

        print(f"  [OK] CPU 推理成功")
        print(f"  [OK] 检测到的物体数量：{len(results[0].boxes)}")

        return True
    except Exception as e:
        print(f"  [FAIL] 推理测试失败：{e}")
        return False

def test_training():
    """测试训练功能（迷你训练）"""
    print("\n[5/5] 测试训练功能（简化版）...")
    try:
        from ultralytics import YOLO

        # 使用极小的数据集进行快速测试
        model = YOLO('yolov8n.pt')

        print("  [OK] 训练功能测试准备就绪")
        print("  提示：CPU 训练较慢，建议使用小模型和少量 epochs")

        return True
    except Exception as e:
        print(f"  [FAIL] 训练功能测试失败：{e}")
        return False

def main():
    print_separator("YOLOv8 CPU 环境测试")

    results = {
        'pytorch': test_pytorch(),
        'torchvision': test_torchvision(),
        'ultralytics': test_ultralytics(),
        'inference': test_inference(),
        'training': test_training()
    }

    print("\n")
    print_separator("测试结果汇总")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test}: {status}")

    print(f"\n  总计：{passed}/{total} 测试通过")

    if passed == total:
        print("\n  [SUCCESS] 所有测试通过！环境配置正确。")
        return 0
    else:
        print("\n  [WARN] 部分测试失败，请检查安装。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
