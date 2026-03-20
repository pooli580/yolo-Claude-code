# YOLO 论文智能体使用指南

## 配置完成！

你的专属 YOLO 论文阅读和参数调优智能体已经配置完成。

## 目录结构

```
D:\Claude-prj\
├── agents/
│   └── yolo-paper-agent.md      # 子代理配置
├── skills/
│   ├── yolo-paper-agent/        # 主智能体技能
│   │   └── SKILL.md
│   ├── yolo-paper-search/       # 论文搜索技能
│   │   └── SKILL.md
│   ├── yolo-paper-analyze/      # 论文分析技能
│   │   └── SKILL.md
│   └── yolo-param-tune/         # 参数调优技能
│       └── SKILL.md
├── .claude/
│   ├── commands/
│   │   ├── search-papers.md     # 论文搜索命令
│   │   ├── analyze-paper.md     # 论文分析命令
│   │   └── tune-params.md       # 参数调优命令
│   └── memory/
│       └── yolo-agent.md        # 智能体记忆
├── papers/                       # 下载的论文
├── notes/                        # 论文笔记
├── experiments/
│   └── template.md              # 实验记录模板
├── configs/
│   ├── template.yaml            # YOLO 配置模板
│   └── history.md               # 参数调整历史
└── README-YOLO-AGENT.md          # 使用指南
```

## 可用技能 (Skills)

以下技能会在你提及相关请求时**自动触发**：

| 技能 | 触发条件 | 功能 |
|------|----------|------|
| `yolo-paper-search` | "搜索 YOLO 论文"、"找最新的 YOLO 文章" | 在 arXiv/Google Scholar/CVPR 等搜索论文 |
| `yolo-paper-analyze` | "分析这篇论文"、"解读这篇 YOLO 文章" | 结构化提取方法、实验、结果 |
| `yolo-param-tune` | "怎么调参"、"分析训练日志" | 分析训练结果并给出调参建议 |
| `yolo-paper-agent` | 综合请求 | 主智能体，协调以上技能 |

## 可用命令 (Commands)

你也可以**直接输入**斜杠命令：

### 1. 搜索论文
```bash
/search-papers <关键词> [--source=arxiv] [--limit=10]
```

**示例：**
```bash
# 搜索 arXiv 上最近的 YOLO 注意力机制论文
/search-papers YOLO attention mechanism --source=arxiv --limit=10

# 搜索所有来源的 YOLO 小目标检测论文
/search-papers YOLO small object detection
```

### 2. 分析论文
```bash
/analyze-paper <paper-path|url> [--deep]
```

**示例：**
```bash
# 分析本地 PDF 论文
/analyze-paper papers/2403.12345.pdf

# 深度分析 arXiv 论文
/analyze-paper https://arxiv.org/abs/2403.12345 --deep
```

### 3. 参数调优
```bash
/tune-params <log-file|metrics.json> [--model=yolov8]
```

**示例：**
```bash
# 分析训练日志并给出调参建议
/tune-params experiments/exp_001/train.log

# 根据指标文件调整参数
/tune-params experiments/exp_001/metrics.json --model=yolov9
```

## 工作流程示例

### 工作流 1: 文献调研
```
1. 你说："帮我找最近的 YOLOv10 论文"
   → 自动触发 yolo-paper-search 技能

2. 技能返回搜索结果和摘要表格

3. 你说："分析第一篇论文"
   → 自动触发 yolo-paper-analyze 技能

4. 技能生成详细分析笔记并保存到 notes/
```

### 工作流 2: 参数调优
```
1. 训练完成后，你说："分析一下这个训练日志"
   → 自动触发 yolo-param-tune 技能

2. 技能分析 Loss 曲线和 mAP 趋势

3. 技能给出 2-3 个调参方案

4. 你选择方案 A

5. 技能生成新配置文件 configs/exp_002.yaml

6. 记录到 configs/history.md
```

## MCP 服务器

已配置的 MCP 服务器：

| 服务器 | 用途 | 状态 |
|--------|------|------|
| **Firecrawl** | 论文搜索和抓取 | ✅ |
| **Context7** | 框架文档查询 | ✅ |
| **Sequential Thinking** | 复杂推理 | ✅ |

## 参数调整速查

### 学习率
| 问题 | 调整 |
|------|------|
| Loss 不下降 | ×10 |
| Loss 震荡 | ÷10 |
| 收敛慢 | warmup + cosine decay |

### Batch Size
| GPU 显存 | 推荐 |
|----------|------|
| 8GB | 8-16 |
| 16GB | 16-32 |
| 24GB | 32-64 |

### 过拟合解决
1. 增加 `weight_decay`
2. 增加数据增强 (`mosaic`, `mixup`)
3. 减小模型尺寸

## 配置文件模板

复制 `configs/template.yaml` 创建新实验配置：
```bash
cp configs/template.yaml configs/exp_001.yaml
```

修改参数后运行：
```bash
yolo train model=yolov8n data=coco.yaml <params>
```

## 实验记录

每次实验在 `experiments/` 目录下创建记录：
```
experiments/
└── exp_001_yolov8_baseline/
    ├── README.md        # 使用 template.md 模板
    ├── config.yaml      # 配置文件
    ├── train.log        # 训练日志
    └── metrics.json     # 评估指标
```

## 下一步

1. **测试技能**: 说 "帮我搜索 YOLO 注意力机制的论文"
2. **查看模板**: 阅读 `configs/template.yaml` 了解配置选项
3. **开始实验**: 复制模板创建你的第一个训练配置

## 需要帮助？

随时询问：
- "帮我找最近的 YOLOv10 论文"
- "分析这个训练日志有什么问题"
- "mAP 不提升应该怎么调参"
- "这篇论文讲了什么" (附上 PDF 或 URL)
