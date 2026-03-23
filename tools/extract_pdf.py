#!/usr/bin/env python3
"""
PDF 论文内容提取工具
提取论文文本内容，用于后续代码生成
"""

import pdfplumber
import sys
import json
from pathlib import Path


def extract_pdf_content(pdf_path: str, output_path: str = None):
    """提取 PDF 文件的完整文本内容"""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"错误：文件不存在 - {pdf_path}")
        return None

    print(f"正在处理：{pdf_path.name}")

    full_text = []
    pages_info = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"总页数：{len(pdf.pages)}")

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                full_text.append(text)

                # 提取页面关键信息
                page_info = {
                    "page": i + 1,
                    "char_count": len(text),
                    "first_lines": text[:500].replace("\n", " ")[:200] if text else ""
                }
                pages_info.append(page_info)

                # 识别关键章节
                if any(kw in text.lower() for kw in ["method", "architecture", "network", "proposed", "our approach"]):
                    print(f"  - 第 {i+1} 页：包含方法/架构内容")
                if any(kw in text.lower() for kw in ["figure", "fig.", "table"]):
                    print(f"  - 第 {i+1} 页：包含图表")
                if any(kw in text.lower() for kw in ["formula", "equation", "conv", "layer", "attention"]):
                    print(f"  - 第 {i+1} 页：包含公式/层定义")

        # 合并全文
        full_content = "\n\n".join(full_text)

        # 输出结果
        result = {
            "filename": pdf_path.name,
            "total_pages": len(pdf.pages),
            "total_chars": len(full_content),
            "content": full_content,
            "pages_summary": pages_info
        }

        # 保存到文件
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = pdf_path.with_suffix(".json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n提取完成，已保存到：{output_file}")
        print(f"总字符数：{len(full_content)}")

        return result

    except Exception as e:
        print(f"提取失败：{e}")
        return None


def analyze_content(content: str):
    """分析提取的内容，识别关键信息"""

    keywords = {
        "network_architecture": ["backbone", "neck", "head", "fpn", "pan", "attention"],
        "convolution": ["conv", "kernel", "stride", "padding", "dilation"],
        "pooling": ["pool", "maxpool", "avgpool", "spp", "sppf"],
        "upsample": ["upsample", "deconv", "transpose", "interpolate"],
        "normalization": ["batchnorm", "bn", "layernorm", "gn", "instancenorm"],
        "activation": ["relu", "gelu", "silu", "sigmoid", "softmax"],
        "attention": ["attention", "se", "cbam", "transformer", "self-attention"],
    }

    analysis = {}
    content_lower = content.lower()

    for category, words in keywords.items():
        count = sum(content_lower.count(w) for w in words)
        analysis[category] = {"mentions": count, "relevant": count > 5}

    return analysis


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python extract_pdf.py <pdf_path> [output_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = extract_pdf_content(pdf_path, output_path)

    if result:
        analysis = analyze_content(result["content"])
        print("\n=== 内容分析 ===")
        for category, info in analysis.items():
            if info["relevant"]:
                print(f"  [+] {category}: {info['mentions']} mentions")
