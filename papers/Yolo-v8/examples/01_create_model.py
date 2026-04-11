"""
示例 1: 创建 YOLOv8-Strip 模型

展示如何创建不同变体的 YOLOv8-Strip 模型
"""

import torch
from __init__ import yolov8n_strip, yolov8s_strip, yolov8m_strip, count_parameters

# ============================================================================
# 1. 创建 Small 版本（推荐）
# ============================================================================
print("=" * 60)
print("YOLOv8s-Strip (Small - 推荐)")
print("=" * 60)

model_s = yolov8s_strip(num_classes=80)
print(f"参数量：{count_parameters(model_s):.2f}M")
print(f"输入：(1, 3, 640, 640)")

# 测试前向传播
model_s.eval()
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    outputs = model_s(x)

for i, (cls, loc, angle) in enumerate(zip(outputs['cls'], outputs['loc'], outputs['angle'])):
    print(f"P{i+3} - cls: {cls.shape}, loc: {loc.shape}, angle: {angle.shape}")

# ============================================================================
# 2. 创建 Nano 版本（轻量）
# ============================================================================
print("\n" + "=" * 60)
print("YOLOv8n-Strip (Nano - 轻量)")
print("=" * 60)

model_n = yolov8n_strip(num_classes=80)
print(f"参数量：{count_parameters(model_n):.2f}M")

# ============================================================================
# 3. 创建自定义类别数的模型
# ============================================================================
print("\n" + "=" * 60)
print("YOLOv8s-Strip (自定义 4 类 - 织物缺陷)")
print("=" * 60)

model_fabric = yolov8s_strip(num_classes=4)
print(f"参数量：{count_parameters(model_fabric):.2f}M")
print(f"类别数：4 (weaver_break, weft_stop, hole, oil_stain)")

# ============================================================================
# 4. 保存模型
# ============================================================================
torch.save(model_fabric.state_dict(), 'yolov8s_fabric.pth')
print("\n模型已保存：yolov8s_fabric.pth")

# ============================================================================
# 5. 加载模型
# ============================================================================
model_loaded = yolov8s_strip(num_classes=4)
model_loaded.load_state_dict(torch.load('yolov8s_fabric.pth'))
print("模型已加载：yolov8s_fabric.pth")

# 清理测试文件
import os
os.remove('yolov8s_fabric.pth')
print("测试文件已清理")

print("\n" + "=" * 60)
print("示例 1 完成!")
print("=" * 60)
