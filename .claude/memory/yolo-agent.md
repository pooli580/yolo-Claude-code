# YOLO 论文智能体记忆

## 项目信息
- **名称**: YOLO 论文阅读与参数调优智能体
- **用户**: lk
- **助手**: 小云
- **创建日期**: 2026-03-18

## 核心功能
1. **论文搜索** - arXiv、Google Scholar、CVPR/ICCV/ECCV
2. **论文分析** - 结构化提取方法、实验、结果
3. **参数调优** - 基于训练结果的超参数调整

## 目录结构
```
D:\Claude-prj\
├── agents/           # Agent 配置
│   └── yolo-paper-agent.md
├── skills/           # 技能定义
│   └── yolo-paper-agent/
├── .claude/commands/ # 斜杠命令
│   ├── search-papers.md
│   ├── analyze-paper.md
│   └── tune-params.md
├── papers/           # 下载的论文
├── notes/            # 论文笔记
├── experiments/      # 实验记录
├── configs/          # 配置文件
│   ├── template.yaml
│   └── history.md
└── runs/             # 训练输出 (由 YOLO 生成)
```

## 可用命令
- `/search-papers <query>` - 搜索论文
- `/analyze-paper <path|url>` - 分析论文
- `/tune-params <log>` - 参数调优

## MCP 配置
- **Firecrawl**: 论文搜索和抓取 (已配置)
- **Context7**: 框架文档查询 (已配置)

## Firecrawl API
- API Key: `fc-4dba0181d04e4872a965dbfe6d812763`
- URL: `https://api.firecrawl.dev/v2`

## Context7 API
- API Key: `ctx7sk-d4bf1773-30a0-4a86-912a-ed8de7140f31`

## 常用资源
### YOLO 系列
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv8: https://github.com/ultralytics/ultralytics
- YOLOv9: https://github.com/WongKinYiu/GELAN
- YOLOv10: https://github.com/THU-MIG/yolov10

### 论文来源
- arXiv: https://arxiv.org/
- CVPR: https://openaccess.thecvf.com/
- ICCV: https://openaccess.thecvf.com/
- ECCV: https://www.ecva.net/

## 注意事项
1. 每次实验只调整 1-2 个参数
2. 保持基线配置用于对比
3. 所有实验必须记录完整
4. 优先复现有官方代码的论文
