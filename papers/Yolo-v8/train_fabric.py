"""
织物缺陷检测 - 训练脚本
用于训练 YOLOv8-Strip 模型进行织物缺陷检测

用法:
    # 基础训练
    python train_fabric.py --data fabric-defect.yaml --epochs 100 --batch 16

    # 指定模型
    python train_fabric.py --model yolov8s-strip.yaml --data fabric-defect.yaml

    # 多 GPU 训练
    python train_fabric.py --device 0,1 --batch 64

    # 断点续训
    python train_fabric.py --resume runs/detect/exp/weights/last.pt
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='织物缺陷检测训练脚本')

    # 模型配置
    parser.add_argument('--model', type=str, default='yolov8s-strip.yaml',
                        help='模型配置文件路径 (e.g., yolov8s-strip.yaml)')
    parser.add_argument('--weights', type=str, default='',
                        help='预训练权重路径 (留空则从头训练)')

    # 数据集配置
    parser.add_argument('--data', type=str, default='fabric-defect.yaml',
                        help='数据集配置文件路径 (e.g., fabric-defect.yaml)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', '--img', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA 设备 (e.g., 0 或 0,1,2,3 或 cpu)')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数')

    # 优化器参数
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型')

    # 保存和恢复
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='保存目录')
    parser.add_argument('--name', type=str, default='fabric-defect-exp',
                        help='实验名称')
    parser.add_argument('--exist-ok', action='store_true',
                        help='覆盖已有实验')
    parser.add_argument('--resume', type=str, default='',
                        help='从指定权重恢复训练')

    # 其他选项
    parser.add_argument('--patience', type=int, default=30,
                        help='早停耐心值')
    parser.add_argument('--save-period', type=int, default=10,
                        help='多少轮保存一次权重')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细训练信息')
    parser.add_argument('--hyp', type=str, default='',
                        help='超参数配置文件路径')

    return parser.parse_args()


def main(args):
    """主训练函数"""
    print("=" * 60)
    print("织物缺陷检测 - YOLOv8-Strip 训练")
    print("=" * 60)

    # 显示配置
    print(f"\n配置信息:")
    print(f"  模型：{args.model}")
    print(f"  数据集：{args.data}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch: {args.batch}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  device: {args.device}")
    print(f"  保存路径：{args.project}/{args.name}")
    print()

    # 加载或创建模型
    if args.weights:
        print(f"加载预训练权重：{args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"创建新模型：{args.model}")
        model = YOLO(args.model)

    # 训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'patience': args.patience,
        'save_period': args.save_period,
        'verbose': args.verbose,
    }

    # 如果有超参数配置文件
    if args.hyp:
        train_args['hyp'] = args.hyp

    # 恢复训练
    if args.resume:
        print(f"\n从权重恢复训练：{args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        # 开始训练
        print("\n开始训练...")
        results = model.train(**train_args)

        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳模型路径：{args.project}/{args.name}/weights/best.pt")
        print("=" * 60)

        return results

    return None


if __name__ == '__main__':
    args = parse_args()
    main(args)
