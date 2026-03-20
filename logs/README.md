# 错误日志使用指南

## 日志文件位置

| 日志类型 | 文件路径 | 用途 |
|---------|---------|------|
| YOLO 模型错误 | `logs/yolo-model-errors.log` | 记录模型训练/推理错误 |
| 对话错误 | `logs/conversation-errors.log` | 记录对话过程中的错误 |


## 智能读取规则
1. **首次启动** → 读取最近 50 行
2. **文件 < 100 行** → 读取全部
3. **文件 > 500 行** → 仅读取：
   - 最近 3 天的所有错误
   - 历史重复错误模式摘要
4. **用户明确请求** → 读取指定日期范围


## 记录错误

### 方法 1：使用脚本工具
```bash
# 记录 YOLO 模型错误
./scripts/log-error.sh yolo "CUDA out of memory at batch 42"

# 记录对话错误
./scripts/log-error.sh conversation "用户输入解析失败"
```

### 方法 2：手动编辑
直接在对应日志文件中按格式添加：

```markdown
## 2026-03-20
- [错误类型] 错误描述
  - 上下文：详细信息
  - 解决方案：xxx (如已解决)
```

## 错误类型分类

### YOLO 模型错误
- `显存不足` - CUDA out of memory
- `数据加载` - DataLoader 错误
- `模型架构` - 层配置错误
- `权重加载` - checkpoint 错误
- `推理错误` - predict/detect 错误

### 对话错误
- `输入解析` - 用户输入理解错误
- `工具调用` - MCP 工具执行失败
- `API 错误` - 外部 API 调用失败
- `文件操作` - 读写文件错误
- `网络错误` - 网络连接问题

## 查看日志

```bash
# 查看全部日志
cat logs/yolo-model-errors.log
cat logs/conversation-errors.log

# 查看今日错误
grep -A 10 "## $(date +%Y-%m-%d)" logs/yolo-model-errors.log

# 查看最近 10 条错误
tail -20 logs/yolo-model-errors.log

# 统计错误数量
grep -c "^\- \[" logs/yolo-model-errors.log
```

## 日志清理

定期清理旧日志（建议每月）：
```bash
# 保留最近 30 天的记录
# 手动编辑文件删除过期条目
```
