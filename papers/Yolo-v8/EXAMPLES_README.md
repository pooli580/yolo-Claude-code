# YOLOv8-Strip 极简核心模块

## 目录结构

```
Yolo-v8/
├── core/                   # 核心模块（必需）
│   ├── __init__.py         # 导出所有组件
│   ├── strip_conv.py       # Strip 卷积
│   ├── strip_net.py        # StripNet 骨干
│   ├── strip_head.py       # Strip 检测头
│   └── yolov8_strip.py     # 完整模型
│
├── configs/                # 配置文件
│   ├── yolov8s-strip.yaml  # 模型配置
│   └── fabric-defect.yaml  # 数据集配置
│
├── scripts/                # 脚本
│   ├── train.py            # 训练脚本
│   └── predict.py          # 预测脚本
│
├── examples/               # 示例代码
│   ├── 01_create_model.py    # 创建模型
│   ├── 02_train.py           # 训练
│   ├── 03_inference.py       # 推理
│   └── 04_custom_dataset.py  # 自定义数据集
│
└── README.md               # 快速开始
```
