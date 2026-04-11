"""
示例 3: 使用训练好的模型进行推理

展示如何加载训练好的模型并进行预测
"""

import torch
import cv2
import numpy as np
from pathlib import Path

# ============================================================================
# 1. 加载训练好的模型
# ============================================================================
print("=" * 60)
print("加载模型")
print("=" * 60)

# 方法 1: 使用 Ultralytics YOLO
from ultralytics import YOLO

# 假设训练后的模型路径
model_path = 'runs/detect/fabric-exp-001/weights/best.pt'

# 检查模型是否存在
if Path(model_path).exists():
    model = YOLO(model_path)
    print(f"已加载模型：{model_path}")
else:
    print(f"模型不存在：{model_path}")
    print("请先运行训练脚本生成模型")
    print("=" * 60)

# ============================================================================
# 2. 单张图片推理
# ============================================================================
print("\n" + "=" * 60)
print("单张图片推理")
print("=" * 60)

# 创建测试图像（实际使用时替换为真实图片路径）
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 使用 YOLO 进行预测
# results = model(test_image)

# # 显示结果
# for result in results:
#     boxes = result.boxes      # 边界框
#     probs = result.probs      # 分类概率
#     masks = result.masks      # 分割掩码（如果是分割模型）
#
#     print(f"检测到 {len(boxes)} 个目标")
#
#     for box in boxes:
#         xyxy = box.xyxy[0].cpu().numpy()  # 边界框坐标
#         conf = box.conf[0].cpu().numpy()  # 置信度
#         cls = int(box.cls[0].cpu().numpy())  # 类别
#         print(f"  - 类别{cls}: 置信度{conf:.2f}, 框={xyxy}")

print("提示：取消注释开始实际推理")

# ============================================================================
# 3. 批量推理
# ============================================================================
print("\n" + "=" * 60)
print("批量推理")
print("=" * 60)

# image_folder = 'data/fabric/images/val'
# results = model(source=image_folder, save=True, save_dir='runs/predict')

# print(f"预测结果保存在：runs/predict")

print("提示：取消注释开始批量推理")

# ============================================================================
# 4. 使用自定义模型进行推理（高级）
# ============================================================================
print("\n" + "=" * 60)
print("使用自定义 YOLOv8-Strip 模型推理")
print("=" * 60)

# 从 yolov8_strip 模块导入
try:
    from yolov8_strip import yolov8s_strip

    # 创建模型
    model_custom = yolov8s_strip(num_classes=4)

    # 加载权重
    # model_custom.load_state_dict(torch.load(model_path, map_location='cpu'))

    model_custom.eval()

    # 准备输入
    img_tensor = torch.randn(1, 3, 640, 640)

    # 推理
    with torch.no_grad():
        outputs = model_custom(img_tensor)

    # 解析输出
    print("\n输出格式:")
    for i, (cls, loc, angle) in enumerate(zip(outputs['cls'], outputs['loc'], outputs['angle'])):
        print(f"  P{i+3}: cls={cls.shape}, loc={loc.shape}, angle={angle.shape}")

except ImportError as e:
    print(f"导入失败：{e}")
    print("请确保在项目根目录运行")

print("\n" + "=" * 60)
print("示例 3 完成!")
print("=" * 60)
