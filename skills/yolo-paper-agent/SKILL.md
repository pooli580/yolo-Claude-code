# YOLO 论文智能体技能

## 技能名称
`yolo-paper-agent`

## 触发条件
当用户请求以下任务时自动激活：
- 搜索 YOLO 相关论文
- 分析/解读论文
- 调整 YOLO 训练参数
- 分析训练结果
- 改进检测模型

## 核心工作流

### 1. 论文搜索工作流

```
输入：用户查询关键词
  ↓
1. 确定搜索源 (arXiv / Google Scholar / 会议网站)
  ↓
2. 使用 Firecrawl 执行搜索
  ↓
3. 提取论文元数据 (标题、摘要、PDF 链接、代码链接)
  ↓
4. 按相关性排序并过滤
  ↓
5. 生成摘要表格
  ↓
6. 对高相关性论文自动下载 PDF
  ↓
输出：论文列表 + 摘要 + 下载链接
```

### 2. 论文分析工作流

```
输入：论文路径或 URL
  ↓
1. 抓取/读取论文内容
  ↓
2. 提取关键信息:
   - 研究问题
   - 核心方法
   - 实验设置
   - 主要结果
  ↓
3. 使用分析模板结构化输出
  ↓
4. 评估对本项目的适用性
  ↓
5. 保存到 notes/ 目录
  ↓
输出：结构化论文笔记
```

### 3. 参数调优工作流

```
输入：训练日志或指标文件
  ↓
1. 解析训练数据 (Loss 曲线、mAP 趋势等)
  ↓
2. 识别问题类型:
   - 收敛问题
   - 过拟合/欠拟合
   - 精度问题
   - 速度问题
  ↓
3. 根因分析
  ↓
4. 提出调整方案 (2-3 个选项)
  ↓
5. 生成新配置文件
  ↓
6. 记录到实验历史
  ↓
输出：参数调整建议 + 新配置
```

## 可用命令

### `/search-papers`
```bash
/search-papers <关键词> [--source=arxiv] [--year=2024-2026] [--limit=10]
```

### `/analyze-paper`
```bash
/analyze-paper <paper.pdf|url> [--deep]
```

### `/tune-params`
```bash
/tune-params <train.log|metrics.json>
```

### `/exp-history`
```bash
/exp-history [--model=yolov8] [--compare]
```

## 工具集成

### Firecrawl 配置
```json
{
  "apiKey": "fc-4dba0181d04e4872a965dbfe6d812763",
  "sources": {
    "arxiv": "https://arxiv.org/search/?query={query}&searchtype=all",
    "google-scholar": "https://scholar.google.com/scholar?q={query}",
    "cvpr": "https://openaccess.thecvf.com/"
  }
}
```

### Context7 库查询
用于查询:
- `/ultralytics/yolov5` - YOLOv5 文档
- `/ultralytics/ultralytics` - YOLOv8 文档
- `/pytorch/pytorch` - PyTorch 文档

## 参数调整知识库

### 学习率策略
| 场景 | 建议 |
|------|------|
| 初始训练 | 0.01 (SGD), 0.001 (Adam) |
| Loss 震荡 | 降低 10 倍 |
| 收敛慢 | 使用 warmup + cosine decay |
| 微调 | 0.0001 - 0.00001 |

### Batch Size
| GPU 内存 | 建议 batch size |
|----------|----------------|
| 8GB | 8-16 |
| 16GB | 16-32 |
| 24GB | 32-64 |
| 40GB+ | 64-128 |

### Weight Decay
| 情况 | 建议 |
|------|------|
| 默认 | 0.0005 |
| 过拟合 | 0.001 - 0.01 |
| 欠拟合 | 0.0001 - 0.0005 |

### Anchor  boxes
使用 K-means 聚类自定义数据集:
```python
from ultralytics.yolo.utils.tal import generate_anchors
# 或使用 autoanchor
python train.py --data dataset.yaml --cache
```

## 输出模板

### 论文搜索结果
```markdown
## 搜索结果："<query>"

| # | 标题 | 来源 | 日期 | 引用 | 代码 |
|---|------|------|------|------|------|
| 1 | ... | arXiv | 2025 | 128 | ✅ |

### 摘要 Top 3
1. **论文标题**
   - 核心贡献：
   - 关键结果：
   - 推荐理由：
```

### 参数调整建议
```markdown
## 参数调整建议

### 当前问题分析
- 现象：训练 Loss 在 10 epoch 后开始震荡
- 可能原因：学习率过大

### 推荐方案
| 方案 | 调整内容 | 预期效果 | 风险 |
|------|----------|----------|------|
| A (推荐) | lr: 0.01→0.001 | Loss 稳定下降 | 收敛稍慢 |
| B | batch: 16→32 | 更稳定梯度 | 需更多显存 |

### 配置文件
已生成：`configs/exp_002_lr_tune.yaml`
```

## 实验记录规范

每次实验必须记录：
```yaml
experiment:
  id: exp_001
  name: yolov8_baseline
  date: 2026-03-18
  model: yolov8
  data: coco

training:
  epochs: 100
  batch_size: 16
  imgsz: 640
  lr0: 0.01
  optimizer: SGD
  weight_decay: 0.0005

results:
  map50: 0.723
  map50-95: 0.542
  best_epoch: 87

notes: |
  基线实验，无特殊调整
```

## 注意事项

1. **搜索准确性**: 优先选择有官方代码的论文
2. **参数调整**: 每次只改 1-2 个参数
3. **实验复现**: 保存随机种子确保可复现
4. **版本控制**: 配置变更记录到 git
