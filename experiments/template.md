# YOLO 实验记录模板
# 每次实验创建新文件：experiments/<exp_id>_<name>/README.md

---
experiment:
  id: exp_001
  name: yolov8_baseline
  date: 2026-03-18
  status: completed  # running, completed, failed

# 模型配置
model:
  type: yolov8n
  pretrained: true
  input_size: 640

# 数据集
data:
  name: coco
  classes: 80
  train: datasets/coco/train2017
  val: datasets/coco/val2017

# 训练配置
training:
  epochs: 100
  batch_size: 16
  optimizer: SGD
  lr0: 0.01
  weight_decay: 0.0005
  scheduler: cosine
  warmup_epochs: 3

# 硬件
hardware:
  gpu: RTX 4090
  memory: 24GB
  device: 0

# 实验结果
results:
  best_epoch: 87
  map50: 0.723
  map50_95: 0.542
  precision: 0.681
  recall: 0.654
  training_time: 4h 23m

# 训练曲线
curves:
  - name: loss
  - name: mAP
  - name: learning_rate

# 调整记录
changes:
  - epoch: 0
    change: 初始配置
    reason: 基线实验

# 笔记和观察
notes: |
  - 基线实验，使用默认配置
  - 训练过程稳定，无明显震荡
  - 第 87 epoch 达到最佳 mAP
  - 无明显过拟合迹象

# 下一步计划
next_steps:
  - 尝试增加学习率观察收敛速度
  - 尝试 Mosaic 数据增强
  - 调整 anchor boxes 适配自定义数据集

# 相关文件
files:
  config: configs/exp_001.yaml
  log: experiments/exp_001/train.log
  metrics: experiments/exp_001/metrics.json
  weights: runs/detect/exp_001/weights/best.pt
---

## 训练日志摘要

```
# 在此粘贴关键训练日志
```

## Loss 曲线

![loss](./plots/loss.png)

## mAP 曲线

![map](./plots/map.png)
