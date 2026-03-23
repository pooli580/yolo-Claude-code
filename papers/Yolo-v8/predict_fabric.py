"""
织物缺陷检测 - 预测脚本

用法:
    # 预测单张图片
    python predict_fabric.py --weights best.pt --source image.jpg

    # 预测整个目录
    python predict_fabric.py --weights best.pt --source path/to/images/ --save-txt
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

# 添加项目路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def predict(source, weights, imgsz=640, device='0', conf=0.5, save=True,
            save_txt=False, save_conf=False, show=False, project='runs/detect',
            name='predict'):
    """
    使用训练好的模型进行预测
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
        project=project,
        name=name,
        exist_ok=True,
    )

    # 处理结果
    print("\n预测结果:")
    for i, result in enumerate(results):
        print(f"\n  图像 {i+1}: {Path(result.path).name}")

        if result.boxes is not None:
            print(f"    检测到 {len(result.boxes)} 个目标:")
            for box in result.boxes:
                cls = int(box.cls)
                conf_val = float(box.conf)
                bbox = box.xyxy[0].tolist()
                print(f"      - {result.names[cls]}: {conf_val:.3f}")
        else:
            print("    未检测到目标")

    print("\n" + "=" * 60)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='织物缺陷检测 - 预测脚本')

    parser.add_argument('--weights', type=str, required=True,
                        help='权重文件路径 (e.g., runs/detect/exp/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源 (图片/视频/目录/摄像头)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA 设备 (e.g., 0 或 cpu)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存标注到 txt 文件')
    parser.add_argument('--save-conf', action='store_true',
                        help='在 txt 中保存置信度')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='保存目录')
    parser.add_argument('--name', type=str, default='predict',
                        help='实验名称')

    args = parser.parse_args()

    predict(
        source=args.source,
        weights=args.weights,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=args.project,
        name=args.name,
    )
