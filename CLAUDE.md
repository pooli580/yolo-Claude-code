# CLAUDE.md

## 角色设定

- **AI 助手名称**: 小云
- **用户名称**: lk
- **专属智能体**: YOLO 论文阅读与参数调优智能体

#### Claude Code 配置
- `agents/` - 专用子代理定义（用于委托任务）
  - `yolo-paper-agent.md` - YOLO 论文智能体配置
- `skills/` - 技能和工作流定义
  - `yolo-paper-search/` - 论文搜索技能
  - `yolo-paper-analyze/` - 论文分析技能
  - `yolo-param-tune/` - 参数调优技能
  - `yolo-paper-agent/` - 主智能体技能
- `commands/` - 斜杠命令快捷执行
  - `/search-papers` - 搜索论文
  - `/analyze-paper` - 分析论文
  - `/tune-params` - 参数调优
- `rules/` - 编码规则和指南
- `hooks/` - 基于触发的自动化
- `contexts/` - 动态上下文配置
- `examples/` - 示例配置
- `.claude/` - Claude Code 配置和记忆
  - `.claude/memory/yolo-agent.md` - YOLO 智能体记忆

## YOLO 智能体配置

### 核心功能
1. **论文搜索** - 在 arXiv、Google Scholar、CVPR/ICCV/ECCV 搜索 YOLO 相关论文
2. **论文分析** - 结构化提取论文方法、实验设置、主要结果
3. **参数调优** - 根据训练结果分析并调整超参数

### 技能触发

以下请求会**自动触发**对应技能：

| 你说 | 触发技能 |
|------|----------|
| "搜索 YOLO 论文" | `yolo-paper-search` |
| "分析这篇论文" | `yolo-paper-analyze` |
| "怎么调参" / "分析训练日志" | `yolo-param-tune` |

### 快速开始
```bash
# 搜索论文
/search-papers YOLO attention mechanism --source=arxiv --limit=10

# 分析论文
/analyze-paper papers/2403.12345.pdf

# 参数调优
/tune-params experiments/exp_001/train.log
```

### 配置文件
- 模板：`configs/template.yaml`
- 历史：`configs/history.md`
- 实验记录：`experiments/template.md`

### MCP 服务器
| 服务器 | 用途 | 状态 |
|--------|------|------|
| Firecrawl | 论文搜索和抓取 | ✅ |
| Context7 | 框架文档查询 | ✅ |
| Sequential Thinking | 复杂推理 | ✅ |

---

## Firecrawl 配置

### API 信息
- **API Key**: `fc-4dba0181d04e4872a965dbfe6d812763`
- **API URL**: `https://api.firecrawl.dev/v2`
- **文档**: https://docs.firecrawl.dev

### MCP 服务器配置
Firecrawl MCP 服务器已配置在 `.mcp.json` 文件中，使用 `npx -y firecrawl-mcp` 命令运行。

### SDK 安装
```bash
# Python
pip install firecrawl-py

# Node.js
pnpm add @mendable/firecrawl-js
```

### 使用示例
```python
from firecrawl import Firecrawl
app = Firecrawl(api_key="fc-4dba0181d04e4872a965dbfe6d812763")
data = app.scrape_url("https://example.com")
```

```typescript
import Firecrawl from '@mendable/firecrawl-js';
const app = new Firecrawl({ apiKey: "fc-4dba0181d04e4872a965dbfe6d812763" });
const data = await app.scrapeUrl("https://example.com");
```
