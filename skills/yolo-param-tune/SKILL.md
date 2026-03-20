---
name: yolo-param-tune
description: YOLO 超参数调优 - 分析训练日志并给出参数调整建议
argument-hint: <log-file|metrics.json> [--model=yolov8]
allowed-tools: [Read, Write, Glob, TodoWrite, AskUserQuestion, Bash]
---

# YOLO 参数调优技能

## 触发条件
当用户请求：
- 分析训练日志
- 怎么调参提升 mAP
- Loss 不下降怎么办
- 训练震荡如何调整
- 根据结果修改参数

## 参数解析
用户输入：$ARGUMENTS

- **log**: 训练日志或指标文件路径
- **model**: YOLO 模型版本 (默认：yolov8)

## 问题诊断知识库

### 学习率问题
| 现象 | 原因 | 解决方案 |
|------|------|----------|
| Loss 不下降 | lr 太小 | lr × 10 |
| Loss 震荡 | lr 太大 | lr ÷ 10 |
| 收敛慢 | 无 warmup | 添加 warmup |
| 后期抖动 | 无衰减 | 使用 cosine decay |

### 过拟合/欠拟合
| 现象 | 原因 | 解决方案 |
|------|------|----------|
| train Loss↓ val Loss↑ | 过拟合 | 增加 weight_decay, 数据增强 |
| train Loss 高 | 欠拟合 | 减小正则化，增大模型 |

### 数据相关问题
| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 小目标检测差 | Anchor 不匹配 | 重新聚类 anchor |
| 大目标漏检 | 输入尺寸小 | 增大 imgsz |
| 类别不平衡 | 数据分布不均 | 类别权重或重采样 |

## 执行流程

### 1. 读取训练日志

```bash
读取 experiments/<exp_id>/train.log 或 metrics.json
```

### 2. 分析训练曲线

提取关键数据：
- 每个 epoch 的 train/val Loss
- mAP50, mAP50-95 趋势
- 学习率变化
- 最佳 epoch

### 3. 识别问题模式

```
分析：
- Loss 是否稳定下降？
- val Loss 是否开始上升？
- mAP 是否趋于平稳？
- 是否有异常波动？
```

### 4. 根因分析

根据现象定位原因：
- 优化器问题
- 学习率问题
- 数据问题
- 模型容量问题

### 5. 提出调整方案

生成 2-3 个方案：

```markdown
## 参数调整建议

### 问题分析
- 现象：训练到第 10 epoch 后 Loss 开始震荡
- 原因：学习率过大

### 推荐方案

| 方案 | 调整内容 | 预期效果 | 风险 |
|------|----------|----------|------|
| A (推荐) | lr: 0.01→0.001 | Loss 稳定下降 | 收敛稍慢 |
| B | batch: 16→32 | 梯度更稳定 | 需更多显存 |
| C | 添加 gradient clip | 减少震荡 | 可能欠拟合 |
```

### 6. 生成新配置文件

```yaml
# configs/exp_002_lr_tune.yaml
model: yolov8n
data: datasets/coco.yaml
epochs: 100
batch: 16
imgsz: 640
lr0: 0.001  # 调整：0.01 → 0.001
weight_decay: 0.0005
# ... 其他参数
```

### 7. 记录实验历史

更新 `configs/history.md`:

```markdown
## 2026-03-18 - exp_002: 学习率调优
| 参数 | 调整 |
|------|------|
| lr0 | 0.01 → 0.001 |

原因：exp_001 中 Loss 震荡
结果：待更新
```

## 输出格式

```markdown
## 训练分析结果

### 📊 训练摘要
- 实验：exp_001
- 模型：yolov8n
- 最佳 epoch: 87
- 最佳 mAP50: 0.723

### 🔍 问题分析
- 现象：第 10-20 epoch 期间 Loss 震荡明显
- 可能原因：初始学习率 0.01 过大

### 💡 推荐方案

| 方案 | 调整 | 预期 |
|------|------|------|
| A | lr: 0.01→0.001 | 稳定收敛 |
| B | 添加 warmup | 平滑初期训练 |

### 📁 配置文件
已生成：`configs/exp_002_lr_tune.yaml`

### 🚀 运行命令
```bash
yolo train model=yolov8n data=coco.yaml lr0=0.001 name=exp_002
```
```

## 常用参数推荐

### YOLOv8 默认超参数
```yaml
lr0: 0.01          # SGD
lrf: 0.1           # 最终学习率系数
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
```

### 不同模型的学习率
| 模型 | 推荐 lr0 |
|------|----------|
| YOLOv8n/s | 0.01 |
| YOLOv8m | 0.001 |
| YOLOv8l/x | 0.0005 |
