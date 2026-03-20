---
name: yolo-paper-search
description: 搜索 YOLO 相关论文 - 在 arXiv、Google Scholar、CVPR/ICCV/ECCV 等来源搜索并分析论文
argument-hint: <关键词> [--source=arxiv] [--limit=10]
allowed-tools: [WebSearch, WebFetch, Read, Write, Glob, TodoWrite, AskUserQuestion, Bash]
---

# YOLO 论文搜索技能

## 触发条件
当用户请求：
- 搜索 YOLO 论文
- 找最新的 YOLO 文章
- 查找 object detection 论文
- 类似的研究搜索请求

## 参数解析
用户输入：$ARGUMENTS

默认参数：
- **query**: 搜索关键词
- **source**: arxiv | google-scholar | cvpr | iccv | eccv | all (默认：all)
- **year**: 年份范围 (默认：2024-2026)
- **limit**: 返回数量 (默认：10)

## 搜索来源

### arXiv
- URL: https://arxiv.org/search/?query={query}&searchtype=all
- 优势：最新预印本，更新快
- 使用：`/search-papers YOLOv10 --source=arxiv`

### Google Scholar
- URL: https://scholar.google.com/scholar?q={query}
- 优势：引用数，相关性排序
- 使用：`/search-papers "YOLO small object" --source=google-scholar`

### 会议网站
- CVPR/ICCV: https://openaccess.thecvf.com/
- ECCV: https://www.ecva.net/

## 执行流程

### 1. 确定搜索策略
根据用户请求确定：
- 主要来源 (arXiv 用于最新，Scholar 用于高引用)
- 关键词组合
- 年份范围

### 2. 执行搜索
使用 WebSearch 或 WebFetch：
```
搜索 "YOLO object detection 2025 2026" site:arxiv.org
```

### 3. 提取论文信息
对每篇论文提取：
- 标题
- 作者
- 摘要
- PDF 链接
- 代码链接 (如有)
- 引用数 (如有)

### 4. 生成摘要表格

```markdown
## 搜索结果："<query>"

共找到 X 篇相关论文

| # | 标题 | 来源 | 日期 | 引用 | 代码 |
|---|------|------|------|------|------|
| 1 | YOLOv10... | arXiv | 2025 | 256 | ✅ |
| 2 | ... | arXiv | 2025 | 128 | ❌ |
```

### 5. Top 3 详细摘要

```markdown
### 重点论文

**1. YOLOv10: Real-Time End-to-End Object Detection**
- 📄 [PDF](https://arxiv.org/abs/xxxx.xxxxx)
- 💻 [代码](https://github.com/...)
- 📊 引用：256
- 核心贡献：提出了...
- 关键结果：mAP 提升 X%，速度提升 Y%
- 推荐理由：SOTA 结果，有官方代码
```

### 6. 询问后续操作

```
需要我：
1. 下载某篇论文的 PDF？
2. 详细分析某篇论文？
3. 搜索更多相关论文？
```

## Firecrawl 集成

如已安装 Firecrawl MCP，可使用：
```
使用 mcp__firecrawl__firecrawl_search 搜索
使用 mcp__firecrawl__firecrawl_scrape 抓取详情
```

## 输出示例

```markdown
## 搜索结果："YOLO attention mechanism"

在 arXiv 上找到 8 篇相关论文 (2024-2026)

| # | 标题 | 来源 | 日期 | 代码 |
|---|------|------|------|------|
| 1 | CBAM: Convolutional Block Attention... | arXiv | 2024 | ✅ |
| 2 | CoAtNet: Hybrid Attention-CNN... | arXiv | 2025 | ✅ |

### 重点论文详解

**1. CBAM in YOLO for Small Object Detection**
- PDF: https://arxiv.org/abs/xxxx.xxxxx
- 代码：https://github.com/...
- 核心：将 CBAM 注意力融入 YOLO 的 neck 部分
- 结果：小目标 mAP 提升 4.2%
- 适用性：高 - 可直接应用到你的模型

---

需要我下载这篇论文的 PDF 或进行详细分析吗？
```
