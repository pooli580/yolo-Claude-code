"""
示例 4: 准备自定义数据集

展示如何准备 YOLO 格式的数据集用于训练
"""

import os
import yaml
from pathlib import Path

# ============================================================================
# 1. YOLO 数据集格式说明
# ============================================================================
print("=" * 60)
print("YOLO 数据集格式")
print("=" * 60)

"""
目录结构:
data/your_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    └── val/
        ├── img001.txt
        ├── img002.txt
        └── ...

标注文件格式 (每行一个目标):
<class_id> <x_center> <y_center> <width> <height>

所有值归一化到 [0, 1] 范围
"""

# ============================================================================
# 2. 创建数据集配置文件
# ============================================================================
print("\n" + "=" * 60)
print("创建数据集配置")
print("=" * 60)

# 示例：织物缺陷数据集配置
fabric_config = {
    'path': 'data/fabric',
    'train': 'images/train',
    'val': 'images/val',
    'nc': 4,
    'names': {
        0: 'weaver_break',    # 断经
        1: 'weft_stop',       # 断纬
        2: 'hole',            # 破洞
        3: 'oil_stain',       # 油污
    }
}

# 保存配置文件
config_path = Path('custom_dataset.yaml')
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(fabric_config, f, allow_unicode=True, default_flow_style=None)

print(f"数据集配置已保存：{config_path}")
print("\n配置内容:")
print(yaml.dump(fabric_config, allow_unicode=True))

# ============================================================================
# 3. 转换标注格式（从其他格式转为 YOLO）
# ============================================================================
print("\n" + "=" * 60)
print("标注格式转换")
print("=" * 60)

def voc_to_yolo(voc_xml_path, yolo_txt_path, class_mapping):
    """
    将 VOC XML 格式转换为 YOLO 格式

    Args:
        voc_xml_path: VOC XML 文件路径
        yolo_txt_path: 输出 YOLO TXT 文件路径
        class_mapping: 类别名称到 ID 的映射
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(voc_xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # 解析目标
    yolo_labels = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        cls_id = class_mapping.get(cls_name, -1)

        if cls_id == -1:
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 转换为 YOLO 格式 (x_center, y_center, w, h) 归一化
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存
    with open(yolo_txt_path, 'w') as f:
        f.write('\n'.join(yolo_labels))

    return len(yolo_labels)


# 示例用法
"""
class_mapping = {
    'weaver_break': 0,
    'weft_stop': 1,
    'hole': 2,
    'oil_stain': 3,
}

# 批量转换
from pathlib import Path

voc_dir = Path('data/voc_annotations')
yolo_dir = Path('data/yolo_annotations')
yolo_dir.mkdir(parents=True, exist_ok=True)

for voc_file in voc_dir.glob('*.xml'):
    yolo_file = yolo_dir / f'{voc_file.stem}.txt'
    num_labels = voc_to_yolo(str(voc_file), str(yolo_file), class_mapping)
    print(f'{voc_file.name}: {num_labels} 个目标')
"""

print("提示：取消注释批量转换代码")

# ============================================================================
# 4. 数据集统计
# ============================================================================
print("\n" + "=" * 60)
print("数据集统计")
print("=" * 60)

def dataset_statistics(data_path, split='train'):
    """
    统计数据集信息

    Args:
        data_path: 数据集根目录
        split: train 或 val
    """
    images_dir = Path(data_path) / 'images' / split
    labels_dir = Path(data_path) / 'labels' / split

    if not images_dir.exists():
        print(f"目录不存在：{images_dir}")
        return

    num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
    num_labels = len(list(labels_dir.glob('*.txt')))

    print(f"\n{split} 集统计:")
    print(f"  图像数：{num_images}")
    print(f"  标注数：{num_labels}")

    # 统计类别分布
    class_counts = {}
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    print(f"  类别分布:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"    类别{cls_id}: {count} 个目标")


# 示例用法
# dataset_statistics('data/fabric', split='train')
# dataset_statistics('data/fabric', split='val')

print("提示：取消注释查看数据集统计")

# 清理测试文件
if config_path.exists():
    config_path.unlink()
    print(f"\n测试文件已清理：{config_path}")

print("\n" + "=" * 60)
print("示例 4 完成!")
print("=" * 60)
