@echo off
chcp 65001 >nul
echo ========================================
echo   YOLOv8 CPU 环境安装脚本
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [信息] Python 已安装
python --version
echo.

REM 检查虚拟环境是否存在
if not exist ".venv" (
    echo [信息] 创建虚拟环境...
    python -m venv .venv
    echo [成功] 虚拟环境创建完成
) else (
    echo [信息] 虚拟环境已存在
)
echo.

REM 激活虚拟环境
echo [信息] 激活虚拟环境...
call .venv\Scripts\activate.bat
echo [成功] 虚拟环境已激活
echo.

REM 升级 pip
echo [信息] 升级 pip...
python -m pip install --upgrade pip --quiet
echo [成功] pip 已升级
echo.

REM 安装 CPU 版 PyTorch
echo [信息] 安装 PyTorch CPU 版 (这可能需要几分钟)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo [错误] PyTorch 安装失败
    pause
    exit /b 1
)
echo [成功] PyTorch CPU 版已安装
echo.

REM 安装 Ultralytics
echo [信息] 安装 Ultralytics (YOLOv8)...
pip install ultralytics
if %errorlevel% neq 0 (
    echo [错误] Ultralytics 安装失败
    pause
    exit /b 1
)
echo [成功] Ultralytics 已安装
echo.

REM 安装其他依赖
echo [信息] 安装其他依赖...
pip install opencv-python matplotlib pillow
echo [成功] 其他依赖已安装
echo.

REM 验证安装
echo ========================================
echo   验证安装
echo ========================================
python -c "import torch; print(f'PyTorch 版本：{torch.__version__}'); print(f'CUDA 可用：{torch.cuda.is_available()}'); from ultralytics import YOLO; print('YOLOv8 安装成功！')"
echo.

echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 使用方法:
echo 1. 激活环境：.venv\Scripts\activate
echo 2. 运行训练：python train.py
echo 3. 运行测试：python test_cpu.py
echo.
pause
