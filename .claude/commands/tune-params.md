---
description: 根据训练日志分析并调整 YOLO 超参数
argument-hint: <log-file|metrics.json> [--model=yolov8]
allowed-tools: [Read, Write, Glob, TodoWrite, AskUserQuestion]
---

# YOLO 参数调优命令

## 参数解析
用户输入：$ARGUMENTS

解析参数：
- **log**: 训练日志文件或指标 JSON
- **model**: YOLO 模型版本 (默认：yolov8)

## 问题诊断表

| 问题现象 | 可能原因 | 调整建议 |
|----------|----------|----------|
| 训练 Loss 不下降 | 学习率过小 | lr × 10 |
| 训练 Loss 震荡 | 学习率过大 | lr ÷ 10, 增大 batch_size |
| 过拟合 (val Loss 上升) | 模型容量过大 | 增加 weight_decay, 数据增强 |
| 欠拟合 | 模型容量不足 | 减小 weight_decay, 增大模型 |
| 小目标检测差 | Anchor 不匹配 | 重新聚类 anchor boxes |
| 收敛慢 | 学习率策略 | 使用 cosine decay, warmup |

## 执行流程

1. **读取训练日志/指标文件**

2. **分析曲线和趋势**
   - Loss 曲线 (train/val)
   - mAP 趋势 (mAP50, mAP50-95)
   - 学习率变化
   - 过拟合迹象

3. **识别问题类型并根因分析**

4. **提出 2-3 种调整方案**

5. **生成新的配置文件**

6. **记录到实验历史**

## 输出格式

```markdown
## 参数调整建议

### 当前问题分析
- 现象：
- 可能原因：

### 推荐方案
| 方案 | 调整内容 | 预期效果 | 风险 |
|------|----------|----------|------|
| A (推荐) |  |  |  |
| B |  |  |  |

### 配置文件
已生成：`configs/<exp-name>.yaml`

### 运行命令
```bash
yolo train model=<model> data=<data> <params>
```
```
