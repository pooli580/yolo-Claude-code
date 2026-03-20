# YOLO 论文分析命令
# 用法：/analyze-paper <paper-path|url> [--deep]

<args>
<arg name="paper" required="true" description="论文 PDF 路径或 URL"/>
<arg name="deep" default="false" description="是否进行深度分析"/>
</args>

<instructions>
1. 读取或抓取论文内容
2. 使用标准分析模板提取信息:
   - 论文信息 (标题、来源、日期、代码)
   - 核心方法 (创新点、技术细节)
   - 实验设置 (数据集、Backbone、参数)
   - 主要结果 (mAP、FPS 等指标)
   - 可复现的技巧
   - 对本项目的适用性
3. 保存到 notes/ 目录
4. 如开启 --deep，额外分析:
   - 方法局限性
   - 与其他工作对比
   - 潜在改进方向
</instructions>

<template>
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
</template>

<example>
/analyze-paper papers/2403.12345.pdf
/analyze-paper https://arxiv.org/abs/2403.12345 --deep
</example>
