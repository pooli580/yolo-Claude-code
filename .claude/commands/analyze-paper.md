---
description: 分析 YOLO 论文 (提取方法、实验、结果)
argument-hint: <paper-path|url> [--deep]
allowed-tools: [Read, Write, WebFetch, Glob, TodoWrite]
---

# YOLO 论文分析命令

## 参数解析
用户输入：$ARGUMENTS

解析参数：
- **paper**: 论文 PDF 路径或 URL
- **deep**: 是否深度分析 (默认：false)

## 执行流程

1. **读取/抓取论文内容**
   - 如为本地路径：读取 PDF 内容
   - 如为 URL：使用 WebFetch 抓取

2. **提取关键信息**
   - 研究问题
   - 核心方法
   - 实验设置
   - 主要结果

3. **使用分析模板结构化输出**

4. **保存到 notes/ 目录**

5. **如开启 --deep**，额外分析:
   - 方法局限性
   - 与其他工作对比
   - 潜在改进方向

## 分析模板

```markdown
## 论文信息
- 标题：
- 来源：
- 日期：
- 代码仓库：

## 核心方法
- 创新点 1:
- 创新点 2:
- 技术细节:

## 实验设置
- 数据集:
- Backbone:
- Input size:
- Batch size:
- Learning rate:
- Optimizer:
- Epochs:

## 主要结果
| 指标 | 值 | 对比基线 |
|------|-----|----------|
| mAP  |     |          |
| FPS  |     |          |

## 可复现的技巧
1.
2.

## 对本项目的适用性
- 可借鉴点:
- 需要注意:
- 建议尝试:
```

## 保存路径

笔记保存到：`notes/<论文标题或编号>.md`
