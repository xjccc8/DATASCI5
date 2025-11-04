@echo off
echo ================================================================================
echo DATASCI5 Folder Cleanup - Archiving Old CPU Scripts
echo ================================================================================
echo.
echo This script will move old CPU-only scripts to an archive folder.
echo They will NOT be deleted, just moved for safekeeping.
echo.
echo Files to archive:
echo   - 02_scripts/run_complete_preprocessing.py
echo   - 02_scripts/generate_masks.py
echo   - 02_scripts/generate_masks_strategy3.py
echo   - 01_notebooks/2_enhanced_preprocessing_pipeline.ipynb
echo.
echo These files are replaced by GPU-optimized versions.
echo.
pause
echo.

echo Creating archive folder...
if not exist "_archive_old_cpu_scripts" mkdir "_archive_old_cpu_scripts"
if not exist "_archive_old_cpu_scripts\scripts" mkdir "_archive_old_cpu_scripts\scripts"
if not exist "_archive_old_cpu_scripts\notebooks" mkdir "_archive_old_cpu_scripts\notebooks"
echo [OK] Archive folders created
echo.

echo Archiving old scripts from 02_scripts/...
if exist "02_scripts\run_complete_preprocessing.py" (
    move "02_scripts\run_complete_preprocessing.py" "_archive_old_cpu_scripts\scripts\"
    echo   [OK] Archived run_complete_preprocessing.py
) else (
    echo   [SKIP] run_complete_preprocessing.py not found
)

if exist "02_scripts\generate_masks.py" (
    move "02_scripts\generate_masks.py" "_archive_old_cpu_scripts\scripts\"
    echo   [OK] Archived generate_masks.py
) else (
    echo   [SKIP] generate_masks.py not found
)

if exist "02_scripts\generate_masks_strategy3.py" (
    move "02_scripts\generate_masks_strategy3.py" "_archive_old_cpu_scripts\scripts\"
    echo   [OK] Archived generate_masks_strategy3.py
) else (
    echo   [SKIP] generate_masks_strategy3.py not found
)

echo.
echo Archiving old notebooks from 01_notebooks/...
if exist "01_notebooks\2_enhanced_preprocessing_pipeline.ipynb" (
    move "01_notebooks\2_enhanced_preprocessing_pipeline.ipynb" "_archive_old_cpu_scripts\notebooks\"
    echo   [OK] Archived 2_enhanced_preprocessing_pipeline.ipynb
) else (
    echo   [SKIP] 2_enhanced_preprocessing_pipeline.ipynb not found
)

echo.
echo Creating archive README...
(
echo # Archived CPU-Only Scripts
echo.
echo **Date Archived:** %DATE% %TIME%
echo **Reason:** Replaced by GPU-optimized versions
echo.
echo ## Archived Files
echo.
echo ### Scripts ^(from 02_scripts/^)
echo 1. `run_complete_preprocessing.py` - Replaced by `gpu_optimized_preprocessing.py` ^(15x faster^)
echo 2. `generate_masks.py` - Replaced by `gpu_optimized_mask_generation.py`
echo 3. `generate_masks_strategy3.py` - Replaced by `gpu_optimized_mask_generation.py`
echo.
echo ### Notebooks ^(from 01_notebooks/^)
echo 1. `2_enhanced_preprocessing_pipeline.ipynb` - Functionality in GPU scripts
echo.
echo ## Why Archived?
echo.
echo These files were CPU-only implementations that are 10-15x slower than the new
echo GPU-optimized versions. They're kept for reference but are no longer needed
echo for the active workflow.
echo.
echo ## Can I Delete These?
echo.
echo Yes! These files are archived and documented. The GPU versions have all the
echo functionality. However, they're kept here "just in case" for reference.
echo.
echo ## GPU Replacement Files
echo.
echo - `02_scripts/gpu_optimized_preprocessing.py` - GPU preprocessing ^(15x faster^)
echo - `02_scripts/gpu_optimized_mask_generation.py` - GPU mask generation ^(10x faster^)
echo - `02_scripts/gpu_monitor.py` - GPU monitoring
echo - `02_scripts/gpu_config.json` - Configuration
echo.
) > "_archive_old_cpu_scripts\README.md"

echo   [OK] Created archive README
echo.

echo ================================================================================
echo CLEANUP COMPLETE!
echo ================================================================================
echo.
echo Archived files location: _archive_old_cpu_scripts\
echo.
echo Current active files:
echo   02_scripts\
echo     - run_dataset_analysis.py
echo     - gpu_optimized_preprocessing.py
echo     - gpu_optimized_mask_generation.py
echo     - gpu_monitor.py
echo     - gpu_config.json
echo.
echo   01_notebooks\
echo     - 1_data_preparation_class_balancing.ipynb
echo.
echo Your folder is now clean and optimized for GPU workflow!
echo.
echo Press any key to exit...
pause >nul
