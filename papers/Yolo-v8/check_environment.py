"""
YOLOv8 训练环境检查脚本
用法：python check_environment.py
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_cmd(cmd, desc):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {desc}: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {desc}: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {desc}: {str(e)}")
        return False

def check_import(module, desc):
    try:
        __import__(module)
        print(f"✅ {desc}: 已安装")
        return True
    except ImportError as e:
        print(f"❌ {desc}: 未安装 - {str(e)}")
        return False

def check_file(path, desc):
    p = Path(path)
    if p.exists():
        print(f"✅ {desc}: {p.absolute()}")
        return True
    else:
        print(f"❌ {desc}: 不存在 - {path}")
        return False

def check_dataset(cfg_path):
    """检查数据集配置"""
    try:
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        path = Path(cfg.get('path', ''))
        train_img = path / cfg.get('train', '') / 'images'
        val_img = path / cfg.get('val', '') / 'images'

        print(f"✅ 数据集配置：{cfg_path}")
        print(f"   类别数：{cfg.get('nc', '未知')}")
        print(f"   类别：{list(cfg.get('names', {}).values())}")

        if path.exists():
            print(f"✅ 数据集根目录：{path.absolute()}")
            if train_img.exists():
                n_train = len(list(train_img.glob('*')))
                print(f"✅ 训练图片数：{n_train}")
            else:
                print(f"⚠️  训练目录不存在：{train_img}")

            if val_img.exists():
                n_val = len(list(val_img.glob('*')))
                print(f"✅ 验证图片数：{n_val}")
            else:
                print(f"⚠️  验证目录不存在：{val_img}")
        else:
            print(f"⚠️  数据集目录不存在：{path}")
        return True
    except Exception as e:
        print(f"❌ 数据集检查失败：{str(e)}")
        return False

def main():
    print_header("YOLOv8 训练环境检查")

    results = []

    # 1. Python 版本
    print_header("1. Python 环境")
    results.append(run_cmd("python --version", "Python 版本"))
    results.append(run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "GPU 信息"))

    # 2. 关键包
    print_header("2. Python 包")
    results.append(check_import("torch", "PyTorch"))
    results.append(check_import("torchvision", "Torchvision"))
    results.append(check_import("ultralytics", "Ultralytics"))
    results.append(check_import("cv2", "OpenCV"))
    results.append(check_import("yaml", "PyYAML"))

    # 3. CUDA
    print_header("3. CUDA 支持")
    run_cmd("nvcc --version", "CUDA 版本")
    run_cmd("python -c \"import torch; print('CUDA 可用' if torch.cuda.is_available() else 'CUDA 不可用')\"", "CUDA 可用性")
    run_cmd("python -c \"import torch; print(f'GPU 数量：{torch.cuda.device_count()}')\"", "GPU 数量")
    run_cmd("python -c \"import torch; print(f'当前 GPU: {torch.cuda.get_device_name(0)}')\"", "当前 GPU")

    # 4. 项目文件
    print_header("4. 项目文件")
    results.append(check_file("train_fabric.py", "训练脚本"))
    results.append(check_file("fabric-defect.yaml", "数据集配置"))
    results.append(check_file("yolov8s-strip.yaml", "模型配置"))

    # 5. 数据集
    print_header("5. 数据集检查")
    check_dataset("fabric-defect.yaml")

    # 6. 快速测试
    print_header("6. 快速功能测试")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ 模型加载测试：成功")
        results.append(True)
    except Exception as e:
        print(f"❌ 模型加载测试：失败 - {str(e)}")
        results.append(False)

    # 总结
    print_header("检查总结")
    passed = sum(results)
    total = len(results)
    print(f"通过：{passed}/{total}")

    if all(results):
        print("\n🎉 所有检查通过！可以开始训练")
        print("\n推荐训练命令:")
        print("  python train_fabric.py --model yolov8s-strip.yaml --data fabric-defect.yaml --epochs 100 --batch 16 --device 0")
    else:
        print("\n⚠️  部分检查未通过，请先修复问题")

    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
