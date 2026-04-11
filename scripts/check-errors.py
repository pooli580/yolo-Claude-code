#!/usr/bin/env python3
"""
YOLO 自动检错脚本
- 监控日志文件中的错误
- 自动分类错误类型
- 生成修复建议
"""

import os
import re
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path("logs")
YOLO_ERROR_LOG = LOGS_DIR / "yolo-model-errors.log"

ERROR_PATTERNS = {
    "CUDA_OOM": [r"CUDA out of memory", r"cudaMalloc failed", r"GPUpd"],
    "导入错误": [r"ImportError", r"ModuleNotFoundError", r"No module named"],
    "形状错误": [r"shape.*mismatch", r"expected.*got", r"size.*doesn't match"],
    "路径错误": [r"FileNotFoundError", r"no such file", r"cannot find"],
    "类型错误": [r"TypeError"],
    "参数错误": [r"ValueError", r"invalid value", r"must be positive"],
}

CORRECTION_SUGGESTIONS = {
    "CUDA_OOM": "减小 batch_size 或启用 AMP 混合精度训练",
    "导入错误": "运行：pip install -r requirements.txt",
    "形状错误": "检查输入尺寸和模型配置是否匹配",
    "路径错误": "检查数据集路径配置",
    "类型错误": "检查张量类型和设备是否一致",
    "参数错误": "检查超参数范围 (lr: 0.00001-0.1)",
}


def scan_errors():
    """扫描错误日志"""
    if not YOLO_ERROR_LOG.exists():
        print("错误日志文件不存在")
        return []

    errors = []
    with open(YOLO_ERROR_LOG, "r", encoding="utf-8") as f:
        content = f.read()

    for error_type, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                errors.append({
                    "type": error_type,
                    "count": len(matches),
                    "suggestion": CORRECTION_SUGGESTIONS[error_type]
                })

    return errors


def main():
    print("=" * 50)
    print("YOLO 自动检错系统")
    print("=" * 50)

    errors = scan_errors()

    if not errors:
        print("\n[OK] 未发现错误")
        return

    print(f"\n发现 {len(errors)} 类错误:\n")

    for error in errors:
        print(f"┌─ {error['type']} (出现 {error['count']} 次)")
        print(f"│  建议：{error['suggestion']}")
        print()


if __name__ == "__main__":
    main()
