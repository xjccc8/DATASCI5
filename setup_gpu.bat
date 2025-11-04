@echo off
echo ================================================================================
echo GPU SETUP FOR RTX 3070 - DATASCI5 Project
echo ================================================================================
echo.
echo This script will install CUDA-enabled PyTorch and required packages
echo for your RTX 3070 GPU optimization.
echo.
echo Your Hardware:
echo   - GPU: NVIDIA RTX 3070 (8GB VRAM)
echo   - CPU: AMD Ryzen 7 5800X
echo   - RAM: 32GB @ 3600MHz
echo.
pause
echo.

echo ================================================================================
echo Step 1: Installing CUDA-enabled PyTorch (CUDA 11.8)
echo ================================================================================
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch installation failed!
    pause
    exit /b 1
)
echo.
echo [OK] PyTorch installed successfully!
echo.

echo ================================================================================
echo Step 2: Installing GPU-accelerated packages
echo ================================================================================
echo.

echo Installing Kornia (GPU augmentation)...
pip install kornia
echo.

echo Installing NVIDIA monitoring...
pip install nvidia-ml-py3
echo.

echo Installing other dependencies...
pip install opencv-python numpy matplotlib albumentations tqdm pillow scikit-image scipy
echo.

if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages may have failed. Check errors above.
) else (
    echo [OK] All packages installed successfully!
)
echo.

echo ================================================================================
echo Step 3: Verifying GPU Setup
echo ================================================================================
echo.

python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not found\"}')"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] GPU verification failed!
    echo Please check:
    echo   1. NVIDIA drivers are installed
    echo   2. CUDA toolkit is installed
    echo   3. PyTorch CUDA version matches your CUDA version
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Step 4: Running GPU Monitor Check
echo ================================================================================
echo.

python 02_scripts\gpu_monitor.py check

echo.
echo ================================================================================
echo SETUP COMPLETE!
echo ================================================================================
echo.
echo Next Steps:
echo   1. Review: GPU_SETUP_GUIDE.md
echo   2. Monitor GPU: python 02_scripts\gpu_monitor.py watch
echo   3. Run preprocessing: python 02_scripts\gpu_optimized_preprocessing.py
echo.
echo Press any key to exit...
pause >nul
