---
name: yolo-paper-analyze
description: 分析 YOLO 论文 - 结构化提取方法、实验、结果和可复现技巧
argument-hint: <paper-path|url> [--deep]
allowed-tools: [Read, Write, WebFetch, Glob, TodoWrite, Bash]
---

# YOLO 论文分析技能

## 触发条件
当用户请求：
- 分析这篇论文
- 解读这篇 YOLO 文章
- 总结这篇论文的方法
- 提取论文的实验设置

## 参数解析
用户输入：$ARGUMENTS

- **paper**: 论文 PDF 路径或 URL
- **deep**: 是否深度分析 (默认：false)

## 执行流程

### 1. 读取/抓取论文

**本地 PDF**:
```
读取 papers/ 目录下的 PDF 文件
```

**URL**:
```
使用 WebFetch 抓取 arXiv 或其他来源的论文页面
```

### 2. 提取关键信息

使用结构化方法阅读论文：
- 标题和作者
- 摘要 (研究问题)
- 方法部分 (核心创新)
- 实验部分 (设置、数据集、指标)
- 结果表格
- 结论

### 3. 生成分析笔记

```markdown
# 论文分析：<标题>

## 基本信息
- **标题**:
- **作者**:
- **来源**: (arXiv/CVPR/ICCV/ECCV)
- **日期**:
- **代码**: [仓库链接]

## 研究问题
这篇论文要解决什么问题？

## 核心方法
### 创新点 1
### 创新点 2
### 技术细节

## 实验设置
| 配置项 | 值 |
|--------|-----|
| 数据集 | |
| Backbone | |
| Input size | |
| Batch size | |
| Learning rate | |
| Optimizer | |
| Epochs | |
| GPU | |

## 主要结果
| 指标 | 值 | 基线 | 提升 |
|------|-----|------|------|
| mAP50 | | | |
| mAP50-95 | | | |
| FPS | | | |

## 可复现的技巧
1. **技巧 1**: 具体描述
2. **技巧 2**: 具体描述

## 对本项目的适用性
- ✅ 可借鉴：
- ⚠️ 需要注意：
- 💡 建议尝试：
```

### 4. 保存到 notes/ 目录

```
notes/<论文标题或编号>.md
```

### 5. 深度分析 (--deep 选项)

如开启深度分析，额外包括：
- 方法局限性分析
- 与其他工作对比
- 潜在改进方向
- 复现难度评估

## 输出示例

```markdown
# 论文分析：YOLOv10

## 基本信息
- 标题：YOLOv10: Real-Time End-to-End Object Detection
- 来源：arXiv:2405.xxxxx
- 日期：2024 年 5 月
- 代码：https://github.com/THU-MIG/yolov10

## 研究问题
解决 YOLO 系列中 NMS 后处理导致的延迟问题

## 核心方法
- 提出端到端的检测架构
- 使用一致性分配消除 NMS
- 引入高效层聚合结构

## 实验设置
- 数据集：COCO
- Backbone: CSPDarknet
- Input: 640×640
- Batch: 64

## 主要结果
| 模型 | mAP | 延迟 |
|------|-----|------|
| YOLOv9 | 53.0 | 3.2ms |
| YOLOv10 | 53.2 | 2.8ms |

## 可复现的技巧
1. NMS-free 训练需要一致性分配策略
2. 高效层聚合可提升多尺度特征

## 适用性
- ✅ 可直接使用 YOLOv10 替换现有模型
- ⚠️ 需要重新训练
- 💡 建议在小数据集上先验证
```
