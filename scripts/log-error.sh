#!/usr/bin/env bash

# YOLO 错误日志记录工具
# 用法：./log-error.sh [yolo|conversation] "错误描述"

LOG_DIR="logs"
DATE=$(date +"%Y-%m-%d")
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 参数检查
if [ $# -lt 2 ]; then
    echo "用法：$0 [yolo|conversation] \"错误描述\""
    echo "示例：$0 yolo \"CUDA out of memory\""
    exit 1
fi

TYPE=$1
MESSAGE=$2

# 根据类型选择日志文件
if [ "$TYPE" = "yolo" ]; then
    LOG_FILE="$LOG_DIR/yolo-model-errors.log"
elif [ "$TYPE" = "conversation" ]; then
    LOG_FILE="$LOG_DIR/conversation-errors.log"
else
    echo "错误：类型必须是 yolo 或 conversation"
    exit 1
fi

# 检查文件是否存在，不存在则创建并添加标题
if [ ! -f "$LOG_FILE" ]; then
    if [ "$TYPE" = "yolo" ]; then
        echo "# YOLO 模型运行错误日志" > "$LOG_FILE"
    else
        echo "# 对话过程错误日志" > "$LOG_FILE"
    fi
    echo "" >> "$LOG_FILE"
fi

# 检查今日日期是否已存在
if grep -q "## $DATE" "$LOG_FILE"; then
    # 日期已存在，追加错误
    sed -i "/^## $DATE/a - [新错误] $MESSAGE" "$LOG_FILE"
else
    # 添加新日期条目
    echo "" >> "$LOG_FILE"
    echo "## $DATE" >> "$LOG_FILE"
    echo "- [新错误] $MESSAGE" >> "$LOG_FILE"
fi

echo "错误已记录到：$LOG_FILE"
echo "时间：$TIMESTAMP"
