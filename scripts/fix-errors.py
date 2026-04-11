#!/usr/bin/env python3
"""
YOLO 自动纠错脚本
根据错误类型生成修复配置文件
"""

import json
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path("logs")
CONFIGS_DIR = Path("configs")

# 纠错策略
CORRECTIONS = {
    "CUDA_OOM": {
        "action": "adjust_batch_size",
        "config": {"batch_size": "reduce_half", "amp": True}
    },
    "导入错误": {
        "action": "install_deps",
        "packages": ["torch", "ultralytics", "opencv-python"]
    },
    "形状错误": {
        "action": "check_input_size",
        "config": {"imgsz": 640}
    },
    "路径错误": {
        "action": "fix_paths",
        "checks": ["data_path", "weights_path"]
    },
}


def generate_correction_config(errors):
    """生成纠错配置"""
    config = {
        "generated_at": datetime.now().isoformat(),
        "corrections": []
    }

    for error in errors:
        if error["type"] in CORRECTIONS:
            config["corrections"].append({
                "error": error["type"],
                "fix": CORRECTIONS[error["type"]]
            })

    return config


def main():
    print("=" * 50)
    print("YOLO 自动纠错系统")
    print("=" * 50)

    # 导入检错模块
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from check_errors import scan_errors

    errors = scan_errors()

    if not errors:
        print("\n[OK] 无需纠错")
        return

    config = generate_correction_config(errors)

    # 保存配置
    output = CONFIGS_DIR / f"correction-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n已生成纠错配置：{output}")
    print(f"包含 {len(config['corrections'])} 项修复建议")


if __name__ == "__main__":
    main()
