"""
织物缺陷检测 - 验证和预测脚本

用法:
    # 验证模型
    python val_fabric.py --weights runs/detect/fabric-defect-exp/weights/best.pt --data fabric-defect.yaml

    # 预测单张图片
    python predict_fabric.py --weights best.pt --source image.jpg

    # 预测整个目录
    python predict_fabric.py --weights best.pt --source path/to/images/ --save-txt --save-conf
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# 添加项目路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# 验证脚本
# ============================================================================

def val_model(weights, data, batch=16, imgsz=640, device='0'):
    """
    验证模型性能

    Args:
        weights: 权重文件路径
        data: 数据集配置文件
        batch: 批次大小
        imgsz: 图像尺寸
        device: CUDA 设备
    """
    print("=" * 60)
    print("织物缺陷检测 - 模型验证")
    print("=" * 60)

    # 加载模型
    model = YOLO(weights)

    # 验证
    results = model.val(
        data=data,
        batch=batch,
        imgsz=imgsz,
        device=device,
        verbose=True,
    )

    # 显示结果
    print("\n" + "=" * 60)
    print("验证结果:")
    print(f"  mAP50:    {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall:    {results.box.mr:.4f}")
    print("=" * 60)

    return results


# ============================================================================
# 预测脚本
# ============================================================================

def predict(source, weights, imgsz=640, device='0', conf=0.5, save=True,
            save_txt=False, save_conf=False, show=False):
    """
    使用训练好的模型进行预测

    Args:
        source: 输入源 (图片/视频/目录/摄像头)
        weights: 权重文件路径
        imgsz: 图像尺寸
        device: CUDA 设备
        conf: 置信度阈值
        save: 是否保存结果
        save_txt: 是否保存标注到 txt
        save_conf: 是否在 txt 中保存置信度
        show: 是否显示结果
    """
    print("=" * 60)
    print("织物缺陷检测 - 预测")
    print("=" * 60)
    print(f"  权重：{weights}")
    print(f"  输入：{source}")
    print(f"  置信度阈值：{conf}")
    print()

    # 加载模型
    model = YOLO(weights)

    # 预测
    results = model.predict(
        source=source,
        imgsz=imgsz,
        device=device,
        conf=conf,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        show=show,
    )

    # 处理结果
    print("\n预测结果:")
    for i, result in enumerate(results):
        print(f"\n  图像 {i+1}: {result.path}")

        if result.boxes is not None:
            print(f"    检测到 {len(result.boxes)} 个目标:")
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()
                print(f"      - {result.names[cls]}: {conf:.3f} (bbox: {bbox})")
        else:
            print("    未检测到目标")

    print("\n" + "=" * 60)

    return results


# ============================================================================
# 批量预测
# ============================================================================

def batch_predict(weights, source_dir, output_dir, imgsz=640, device='0', conf=0.5):
    """
    批量预测目录中的所有图片

    Args:
        weights: 权重文件路径
        source_dir: 输入图片目录
        output_dir: 输出结果目录
        imgsz: 图像尺寸
        device: CUDA 设备
        conf: 置信度阈值
    """
    from PIL import Image
    import os

    print("=" * 60)
    print("织物缺陷检测 - 批量预测")
    print("=" * 60)
    print(f"  输入目录：{source_dir}")
    print(f"  输出目录：{output_dir}")
    print()

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = YOLO(weights)

    # 获取所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [
        f for f in Path(source_dir).iterglob('*')
        if f.suffix.lower() in image_extensions
    ]

    print(f"  找到 {len(image_files)} 张图片")
    print()

    # 批量处理
    results_summary = []
    for img_file in image_files:
        print(f"  处理：{img_file.name}")

        results = model.predict(
            source=str(img_file),
            imgsz=imgsz,
            device=device,
            conf=conf,
            save=True,
            project=output_dir,
            name='predictions',
            exist_ok=True,
        )

        # 统计结果
        for result in results:
            num_detections = len(result.boxes) if result.boxes else 0
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf_val = float(box.conf)
                    detections.append({
                        'class': result.names[cls],
                        'confidence': conf_val,
                    })

            results_summary.append({
                'file': img_file.name,
                'num_detections': num_detections,
                'detections': detections,
            })

    # 保存总结
    import json
    summary_file = Path(output_dir) / 'results_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  结果总结已保存到：{summary_file}")
    print("=" * 60)

    return results_summary


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='织物缺陷检测 - 验证和预测')

    subparsers = parser.add_subparsers(dest='command', help='命令类型')

    # 验证命令
    val_parser = subparsers.add_parser('val', help='验证模型')
    val_parser.add_argument('--weights', type=str, required=True,
                            help='权重文件路径')
    val_parser.add_argument('--data', type=str, default='fabric-defect.yaml',
                            help='数据集配置文件')
    val_parser.add_argument('--batch', type=int, default=16,
                            help='批次大小')
    val_parser.add_argument('--imgsz', type=int, default=640,
                            help='图像尺寸')
    val_parser.add_argument('--device', type=str, default='0',
                            help='CUDA 设备')

    # 预测命令
    pred_parser = subparsers.add_parser('predict', help='预测单张图片')
    pred_parser.add_argument('--weights', type=str, required=True,
                             help='权重文件路径')
    pred_parser.add_argument('--source', type=str, required=True,
                             help='输入图片路径')
    pred_parser.add_argument('--imgsz', type=int, default=640,
                             help='图像尺寸')
    pred_parser.add_argument('--device', type=str, default='0',
                             help='CUDA 设备')
    pred_parser.add_argument('--conf', type=float, default=0.5,
                             help='置信度阈值')
    pred_parser.add_argument('--save-txt', action='store_true',
                             help='保存标注到 txt')
    pred_parser.add_argument('--save-conf', action='store_true',
                             help='在 txt 中保存置信度')
    pred_parser.add_argument('--show', action='store_true',
                             help='显示结果')

    # 批量预测命令
    batch_parser = subparsers.add_parser('batch', help='批量预测')
    batch_parser.add_argument('--weights', type=str, required=True,
                              help='权重文件路径')
    batch_parser.add_argument('--source-dir', type=str, required=True,
                              help='输入图片目录')
    batch_parser.add_argument('--output-dir', type=str, required=True,
                              help='输出结果目录')
    batch_parser.add_argument('--imgsz', type=int, default=640,
                              help='图像尺寸')
    batch_parser.add_argument('--device', type=str, default='0',
                              help='CUDA 设备')
    batch_parser.add_argument('--conf', type=float, default=0.5,
                              help='置信度阈值')

    args = parser.parse_args()

    if args.command == 'val':
        val_model(args.weights, args.data, args.batch, args.imgsz, args.device)
    elif args.command == 'predict':
        predict(args.source, args.weights, args.imgsz, args.device,
                args.conf, save_txt=args.save_txt, save_conf=args.save_conf,
                show=args.show)
    elif args.command == 'batch':
        batch_predict(args.weights, args.source_dir, args.output_dir,
                      args.imgsz, args.device, args.conf)
    else:
        parser.print_help()
