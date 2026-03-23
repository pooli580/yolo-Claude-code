#!/usr/bin/env python3
"""
PDF 论文内容提取 - 提取关键方法部分
"""

import json
from pathlib import Path


def extract_key_sections(json_path: str):
    """提取论文中的关键章节"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content = data["content"]

    # 定义关键章节标识
    section_markers = [
        "abstract",
        "introduction",
        "method",
        " methodology",
        "our approach",
        "network architecture",
        "strip convolution",
        "large strip convolution",
        "detection head",
        "experiment",
        "implementation details"
    ]

    sections = {}
    content_lower = content.lower()

    # 查找章节位置
    for marker in section_markers:
        idx = content_lower.find(marker)
        if idx != -1:
            # 提取章节内容（marker 前后 2000 字符）
            start = max(0, idx - 100)
            end = min(len(content), idx + 3000)
            sections[marker.strip()] = content[start:end]

    # 搜索关键技术词汇
    keywords = [
        "conv.*kernel", "strip.*conv", "attention", "backbone",
        "fpn", "pan", "detection head", "roi", "feature pyramid",
        "stride", "padding", "dilation", "3x3", "5x5", "7x7",
        "concat", "add", "elementwise", "sigmoid", "softmax"
    ]

    print(f"\n=== 论文章节提取：{data['filename']} ===")
    print(f"总页数：{data['total_pages']}, 总字符：{data['total_chars']}\n")

    for section_name, section_content in sections.items():
        print(f"\n--- {section_name.upper()} ---")
        print(section_content[:1500])
        print("...")

    # 保存精简版
    output = {
        "filename": data["filename"],
        "total_pages": data["total_pages"],
        "sections": sections
    }

    output_path = Path(json_path).with_name("strip_rcnn_key_sections.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for name, content in sections.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"SECTION: {name.upper()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(content)
            f.write("\n")

    print(f"\n精简内容已保存到：{output_path}")
    return output


if __name__ == "__main__":
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else \
        "D:/Claude-prj/papers/Technical-literature/下采样 检测头/Strip R-CNN Large Strip Convolution for Remote Sensing Object Detection.json"

    extract_key_sections(json_path)
