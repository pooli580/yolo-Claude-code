"""
数据集划分脚本
将数据集按指定比例划分为训练集和验证集

用法:
    # 8:2 划分
    python split_dataset.py --source datasets/fabric-defect --split 0.8

    # 9:1 划分
    python split_dataset.py --source datasets/fabric-defect --split 0.9
"""

import argparse
import random
import shutil
from pathlib import Path


def split_dataset(source_dir, split_ratio=0.8, seed=42):
    """
    划分数据集为训练集和验证集

    Args:
        source_dir: 数据集根目录 (包含 images 和 labels 子目录)
        split_ratio: 训练集比例 (默认 0.8)
        seed: 随机种子
    """
    random.seed(seed)

    source = Path(source_dir)
    images_dir = source / 'images'
    labels_dir = source / 'labels'

    # 检查目录结构
    if not images_dir.exists():
        print(f"错误：找不到 images 目录：{images_dir}")
        return

    # 获取所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    all_images = list(images_dir.glob(f'*'))
    all_images = [f for f in all_images if f.suffix.lower() in image_extensions]

    print(f"找到 {len(all_images)} 张图片")

    # 随机打乱
    random.shuffle(all_images)

    # 计算划分点
    split_idx = int(len(all_images) * split_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"训练集：{len(train_images)} 张 ({split_ratio*100:.0f}%)")
    print(f"验证集：{len(val_images)} 张 ({(1-split_ratio)*100:.0f}%)")

    # 创建目标目录
    train_images_dir = source / 'images' / 'train'
    val_images_dir = source / 'images' / 'val'
    train_labels_dir = source / 'labels' / 'train'
    val_labels_dir = source / 'labels' / 'val'

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # 复制训练集
    print("\n复制训练集...")
    for img_file in train_images:
        # 复制图片
        shutil.copy(img_file, train_images_dir / img_file.name)

        # 复制对应的标注文件
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, train_labels_dir / label_file.name)
        else:
            # 创建空标注文件
            (train_labels_dir / f"{img_file.stem}.txt").touch()

    # 复制验证集
    print("复制验证集...")
    for img_file in val_images:
        # 复制图片
        shutil.copy(img_file, val_images_dir / img_file.name)

        # 复制对应的标注文件
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, val_labels_dir / label_file.name)
        else:
            # 创建空标注文件
            (val_labels_dir / f"{img_file.stem}.txt").touch()

    print("\n数据集划分完成!")
    print(f"  训练集：{train_images_dir}")
    print(f"  验证集：{val_images_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据集划分脚本')

    parser.add_argument('--source', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--split', type=float, default=0.8,
                        help='训练集比例 (默认 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    split_dataset(args.source, args.split, args.seed)
