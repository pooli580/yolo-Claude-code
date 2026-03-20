# YOLO Claude Code Agent

> 专属 YOLO 论文阅读与参数调优智能体

## 项目简介

这是一个基于 Claude Code 的专属智能体，专注于 YOLO 系列目标检测算法的论文阅读、文献调研和超参数调优。

**AI 助手名称**: 小云
**适用用户**: YOLO 算法研究者、目标检测工程师

---

## 项目结构

```
yolo-Claude-code/
├── agents/                          # 子代理配置
│   └── yolo-paper-agent.md          # YOLO 论文智能体主配置
│
├── skills/                          # 技能定义
│   ├── yolo-paper-agent/            # 主智能体技能
│   │   └── SKILL.md
│   ├── yolo-paper-search/           # 论文搜索技能
│   │   └── SKILL.md
│   ├── yolo-paper-analyze/          # 论文分析技能
│   │   └── SKILL.md
│   └── yolo-param-tune/             # 参数调优技能
│       └── SKILL.md
│
├── .claude/                         # Claude Code 配置
│   ├── commands/                    # 斜杠命令
│   │   ├── search-papers.md         # 论文搜索命令
│   │   ├── analyze-paper.md         # 论文分析命令
│   │   └── tune-params.md           # 参数调优命令
│   ├── memory/
│   │   └── yolo-agent.md            # 智能体记忆
│   ├── settings.json                # 主配置
│   └── settings.local.json          # 本地配置
│
├── configs/                         # YOLO 配置文件
│   ├── template.yaml                # 训练配置模板
│   └── history.md                   # 参数调整历史
│
├── experiments/                     # 实验记录
│   └── template.md                  # 实验报告模板
│
├── CLAUDE.md                        # 项目主文档
├── README.md                        # 项目说明（本文件）
├── README-YOLO-AGENT.md             # 使用指南
└── .mcp.json                        # MCP 服务器配置
```

---

## 核心功能

### 1. 论文搜索 (yolo-paper-search)
在 arXiv、Google Scholar、CVPR/ICCV/ECCV 等学术平台搜索 YOLO 相关论文

### 2. 论文分析 (yolo-paper-analyze)
结构化提取论文的核心方法、实验设置和主要结果

### 3. 参数调优 (yolo-param-tune)
根据训练日志分析 Loss 曲线、mAP 趋势，给出超参数调整建议

---

## 快速开始

### 前置条件

1. **Claude Code** - 安装并配置 Claude Code
2. **API Keys** - 在 `.env` 文件中配置必要的 API 密钥

### 安装

```bash
# 克隆仓库
git clone https://github.com/pooli580/yolo-Claude-code.git
cd yolo-Claude-code

# 安装 Firecrawl SDK (可选)
pip install firecrawl-py
# 或
pnpm add @mendable/firecrawl-js
```

### 配置

1. 复制 `.env.example` 为 `.env`（如需要）
2. 在 `.env` 中配置 API 密钥：
   ```
   FIRECRAWL_API_KEY=your_api_key
   FIRECRAWL_API_URL=https://api.firecrawl.dev/v2
   ```

---

## 使用方法

### 方式一：自然语言触发（推荐）

直接描述你的需求，智能体会自动触发对应技能：

| 你说 | 触发技能 |
|------|----------|
| "搜索 YOLO 论文" | `yolo-paper-search` |
| "分析这篇论文" | `yolo-paper-analyze` |
| "怎么调参" / "分析训练日志" | `yolo-param-tune` |

### 方式二：斜杠命令

#### 1. 搜索论文
```bash
/search-papers <关键词> [--source=arxiv] [--limit=10]
```

**示例：**
```bash
# 搜索 arXiv 上的 YOLO 注意力机制论文
/search-papers YOLO attention mechanism --source=arxiv --limit=10

# 搜索所有来源的 YOLO 小目标检测论文
/search-papers YOLO small object detection
```

#### 2. 分析论文
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

#### 3. 参数调优
```bash
/tune-params <log-file|metrics.json> [--model=yolov8]
```

**示例：**
```bash
# 分析训练日志
/tune-params experiments/exp_001/train.log

# 根据指标文件调整参数
/tune-params experiments/exp_001/metrics.json --model=yolov9
```

---

## 工作流示例

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

---

## MCP 服务器配置

项目使用 MCP (Model Context Protocol) 服务器扩展功能：

| 服务器 | 用途 | 状态 |
|--------|------|------|
| **Firecrawl** | 论文搜索和网页抓取 | ✅ |
| **Context7** | 框架文档查询 | ✅ |
| **Sequential Thinking** | 复杂推理 | ✅ |

---

## 参数调优速查表

### 学习率调整
| 问题 | 调整方案 |
|------|----------|
| Loss 不下降 | ×10 |
| Loss 震荡 | ÷10 |
| 收敛慢 | warmup + cosine decay |

### Batch Size 推荐
| GPU 显存 | 推荐 Batch Size |
|----------|-----------------|
| 8GB | 8-16 |
| 16GB | 16-32 |
| 24GB | 32-64 |

### 过拟合解决方案
1. 增加 `weight_decay`
2. 增加数据增强 (`mosaic`, `mixup`)
3. 减小模型尺寸

---

## 实验管理

### 创建新实验

```bash
# 复制配置模板
cp configs/template.yaml configs/exp_001.yaml

# 创建实验记录
cp experiments/template.md experiments/exp_001_readme.md
```

### 实验记录结构

```
experiments/
└── exp_001_yolov8_baseline/
    ├── README.md        # 实验说明
    ├── config.yaml      # 配置文件
    ├── train.log        # 训练日志
    └── metrics.json     # 评估指标
```

---

## 安全注意事项

- `.env` 文件包含敏感 API 密钥，已添加到 `.gitignore`
- 不要将 `.env` 文件提交到 GitHub
- 定期轮换 API 密钥

---

## 相关资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [arXiv](https://arxiv.org/)
- [CVPR Open Access](https://openaccess.thecvf.com/)
- [Firecrawl 文档](https://docs.firecrawl.dev)

---

## License

MIT

---

**Happy Researching!** 🎯
