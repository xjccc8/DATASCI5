@echo off
REM Installation script for GPU Mask Generation V2 additional requirements

echo ================================================================================
echo Installing Additional Requirements for GPU Mask Generation V2
echo ================================================================================
echo.

echo Installing albumentations (data augmentation library)...
pip install albumentations

echo.
echo Installing torchvision (for pretrained ResNet34)...
pip install torchvision

echo.
echo Verifying installations...
python -c "import albumentations; print('[OK] albumentations version:', albumentations.__version__)"
python -c "import torchvision; print('[OK] torchvision version:', torchvision.__version__)"

echo.
echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.
echo You can now run: python 02_scripts\gpu_optimized_mask_generation_v2.py
echo.
pause
