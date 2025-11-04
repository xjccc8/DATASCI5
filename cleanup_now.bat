@echo off
echo Cleaning up old CPU scripts...

REM Create archive folder structure
if not exist "_archive_old_cpu_scripts" mkdir "_archive_old_cpu_scripts"
if not exist "_archive_old_cpu_scripts\scripts" mkdir "_archive_old_cpu_scripts\scripts"
if not exist "_archive_old_cpu_scripts\notebooks" mkdir "_archive_old_cpu_scripts\notebooks"

REM Move old scripts
if exist "02_scripts\run_complete_preprocessing.py" (
    move "02_scripts\run_complete_preprocessing.py" "_archive_old_cpu_scripts\scripts\" >nul 2>&1
    echo [OK] Archived run_complete_preprocessing.py
)

if exist "02_scripts\generate_masks.py" (
    move "02_scripts\generate_masks.py" "_archive_old_cpu_scripts\scripts\" >nul 2>&1
    echo [OK] Archived generate_masks.py
)

if exist "02_scripts\generate_masks_strategy3.py" (
    move "02_scripts\generate_masks_strategy3.py" "_archive_old_cpu_scripts\scripts\" >nul 2>&1
    echo [OK] Archived generate_masks_strategy3.py
)

REM Move old notebook
if exist "01_notebooks\2_enhanced_preprocessing_pipeline.ipynb" (
    move "01_notebooks\2_enhanced_preprocessing_pipeline.ipynb" "_archive_old_cpu_scripts\notebooks\" >nul 2>&1
    echo [OK] Archived 2_enhanced_preprocessing_pipeline.ipynb
)

REM Create archive README
echo # Archived CPU-Only Scripts> "_archive_old_cpu_scripts\README.md"
echo.>> "_archive_old_cpu_scripts\README.md"
echo **Date Archived:** %DATE% %TIME%>> "_archive_old_cpu_scripts\README.md"
echo **Reason:** Replaced by GPU-optimized versions>> "_archive_old_cpu_scripts\README.md"
echo.>> "_archive_old_cpu_scripts\README.md"
echo ## Archived Files>> "_archive_old_cpu_scripts\README.md"
echo.>> "_archive_old_cpu_scripts\README.md"
echo ### Scripts>> "_archive_old_cpu_scripts\README.md"
echo - run_complete_preprocessing.py - Replaced by gpu_optimized_preprocessing.py>> "_archive_old_cpu_scripts\README.md"
echo - generate_masks.py - Replaced by gpu_optimized_mask_generation.py>> "_archive_old_cpu_scripts\README.md"
echo - generate_masks_strategy3.py - Replaced by gpu_optimized_mask_generation.py>> "_archive_old_cpu_scripts\README.md"
echo.>> "_archive_old_cpu_scripts\README.md"
echo ### Notebooks>> "_archive_old_cpu_scripts\README.md"
echo - 2_enhanced_preprocessing_pipeline.ipynb - Functionality in GPU scripts>> "_archive_old_cpu_scripts\README.md"

echo.
echo ========================================
echo CLEANUP COMPLETE!
echo ========================================
echo.
echo Archived 4 old CPU-only files to: _archive_old_cpu_scripts\
echo Active files remaining: 5 GPU-optimized scripts
echo.
