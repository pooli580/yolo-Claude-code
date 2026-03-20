---
description: 搜索 YOLO 相关论文 (arXiv, Google Scholar, CVPR/ICCV/ECCV)
argument-hint: <关键词> [--source=arxiv] [--year=2024-2026] [--limit=10]
allowed-tools: [WebSearch, WebFetch, Read, Write, Glob, TodoWrite, AskUserQuestion]
---

# YOLO 论文搜索命令

## 参数解析
用户输入：$ARGUMENTS

解析参数：
- **query**: 搜索关键词
- **source**: arxiv | google-scholar | cvpr | iccv | eccv | all (默认：all)
- **year**: 年份范围 (默认：2024-2026)
- **limit**: 返回数量 (默认：10)

## 搜索来源 URL

- **arXiv**: https://arxiv.org/search/?query={query}&searchtype=all
- **Google Scholar**: https://scholar.google.com/scholar?q={query}
- **CVPR**: https://openaccess.thecvf.com/
- **ICCV**: https://openaccess.thecvf.com/
- **ECCV**: https://www.ecva.net/

## 执行流程

1. **使用 Firecrawl 搜索**
   ```
   使用 Firecrawl MCP 搜索指定来源
   ```

2. **提取论文元数据**
   - 标题
   - 作者
   - 摘要
   - PDF 链接
   - 代码链接 (如有)

3. **生成摘要表格**
   | # | 标题 | 来源 | 日期 | 代码 |
   |---|------|------|------|------|
   | 1 | ... | ... | ... | ... |

4. **对 Top 3 论文生成详细摘要**

5. **询问用户是否需要下载 PDF**

## 输出格式

```markdown
## 搜索结果："<query>"

共找到 X 篇论文

| # | 标题 | 来源 | 日期 | 代码 |
|---|------|------|------|------|
| 1 | ... | arXiv | 2025 | ✅ |

### Top 3 详细摘要

**1. 论文标题**
- 链接：[PDF](url) | [代码](url)
- 核心贡献：
- 关键结果：
- 推荐理由：

---

需要我下载哪篇论文的 PDF 到 papers/ 目录？
```
