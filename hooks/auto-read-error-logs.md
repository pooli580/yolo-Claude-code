# 启动开发时自动读取错误日志
# 触发时机：每次用户开始对话或请求开发帮助时

## 前置条件
- 检查错误日志文件是否存在

## 智能读取策略

| 日志大小 | 读取范围 | Token 消耗 |
|---------|---------|-----------|
| < 100 行 | 读取全部 | 低 |
| 100-500 行 | 最近 50 行 | 中 |
| > 500 行 | 最近 30 行 + 今日全部 | 低 |

## 执行内容

### 1. 读取 YOLO 模型错误日志
```bash
if [ -f "logs/yolo-model-errors.log" ]; then
  LINES=$(wc -l < logs/yolo-model-errors.log)
  echo "=== YOLO 模型错误日志 (共$LINES 行) ==="
  if [ $LINES -lt 100 ]; then
    cat logs/yolo-model-errors.log
  elif [ $LINES -lt 500 ]; then
    tail -50 logs/yolo-model-errors.log
  else
    echo "--- 最近 30 条 ---"
    tail -30 logs/yolo-model-errors.log
    echo "--- 今日错误 ---"
    grep -A 20 "## $(date +%Y-%m-%d)" logs/yolo-model-errors.log
  fi
fi
```

### 2. 读取对话错误日志
```bash
if [ -f "logs/conversation-errors.log" ]; then
  LINES=$(wc -l < logs/conversation-errors.log)
  echo "=== 对话错误日志 (共$LINES 行) ==="
  if [ $LINES -lt 100 ]; then
    cat logs/conversation-errors.log
  elif [ $LINES -lt 500 ]; then
    tail -50 logs/conversation-errors.log
  else
    echo "--- 最近 30 条 ---"
    tail -30 logs/conversation-errors.log
    echo "--- 今日错误 ---"
    grep -A 20 "## $(date +%Y-%m-%d)" logs/conversation-errors.log
  fi
fi
```

### 3. 错误摘要
- 统计当日错误数量
- 识别重复出现的错误模式
- 提供可能的解决方案建议

## 记录新错误格式

### YOLO 模型错误
```markdown
## YYYY-MM-DD
- [错误类型] 错误描述
  - 文件：xxx.py
  - 行号：line xx
  - 错误信息：xxx
  - 解决方案：xxx (如已解决)
```

### 对话错误
```markdown
## YYYY-MM-DD
- [错误类型] 错误描述
  - 上下文：用户请求 xxx
  - 错误信息：xxx
  - 解决方案：xxx (如已解决)
```
