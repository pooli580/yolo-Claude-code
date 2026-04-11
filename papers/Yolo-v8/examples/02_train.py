"""
示例 2: 训练 YOLOv8-Strip 模型

展示如何使用 Ultralytics 框架训练模型
"""

from ultralytics import YOLO

# ============================================================================
# 方法 1: 使用 YOLOv8 CLI 训练（推荐）
# ============================================================================
"""
命令行运行：

# 基础训练
yolo train model=yolov8s-strip.yaml data=fabric-defect.yaml epochs=100 batch=16

# 指定更多参数
yolo train \\
    model=yolov8s-strip.yaml \\
    data=fabric-defect.yaml \\
    epochs=200 \\
    batch=16 \\
    imgsz=1024 \\
    device=0 \\
    optimizer=AdamW \\
    lr0=0.001 \\
    project=runs/detect \\
    name=fabric-exp
"""

# ============================================================================
# 方法 2: 使用 Python API 训练
# ============================================================================
print("=" * 60)
print("使用 Python API 训练")
print("=" * 60)

# 创建模型（使用 Ultralytics 的 YOLO 类）
# 注意：需要 yolov8s-strip.yaml 文件定义模型架构
model = YOLO('yolov8s-strip.yaml')

# 训练参数
train_args = {
    'data': 'fabric-defect.yaml',    # 数据集配置
    'epochs': 100,                    # 训练轮数
    'batch': 16,                      # 批次大小
    'imgsz': 640,                     # 输入尺寸
    'device': '0',                    # GPU 设备
    'workers': 4,                     # 数据加载线程数
    'optimizer': 'AdamW',             # 优化器
    'lr0': 0.001,                     # 初始学习率
    'project': 'runs/detect',         # 保存目录
    'name': 'fabric-exp-001',         # 实验名称
    'patience': 30,                   # 早停耐心值
    'save_period': 10,                # 多少轮保存一次
    'verbose': True,                  # 详细输出
}

print("\n训练配置:")
for key, value in train_args.items():
    print(f"  {key}: {value}")

print("\n开始训练...")
print("=" * 60)

# 开始训练
# results = model.train(**train_args)

# print("\n训练完成!")
# print(f"最佳模型：runs/detect/fabric-exp-001/weights/best.pt")

print("\n提示：取消注释 model.train() 开始实际训练")
print("=" * 60)

# ============================================================================
# 方法 3: 从头训练 vs 迁移学习
# ============================================================================
"""
从头训练:
    model = YOLO('yolov8s-strip.yaml')
    model.train(data='fabric-defect.yaml', epochs=100)

迁移学习 (使用 COCO 预训练权重):
    model = YOLO('yolov8s.pt')  # 加载预训练权重
    model.train(data='fabric-defect.yaml', epochs=100)
"""

print("\n示例 2 完成!")
