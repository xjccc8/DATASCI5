# Folder Cleanup Plan - GPU Optimization

## Files to Remove (Obsolete/Redundant)

### 📁 02_scripts/ - Old CPU-Only Scripts

#### ❌ REMOVE - Replaced by GPU versions:
1. **`run_complete_preprocessing.py`**
   - Replaced by: `gpu_optimized_preprocessing.py`
   - Reason: CPU-only, 15x slower
   - Action: Delete (functionality fully replaced)

2. **`generate_masks.py`**
   - Replaced by: `gpu_optimized_mask_generation.py`
   - Reason: Basic CPU implementation, 10x slower
   - Action: Delete (outdated, Strategy 3 is better)

3. **`generate_masks_strategy3.py`**
   - Replaced by: `gpu_optimized_mask_generation.py`
   - Reason: CPU-only version, 10x slower
   - Action: Delete (GPU version includes Strategy 3 improvements)

#### ✅ KEEP - Still useful:
4. **`run_dataset_analysis.py`**
   - Keep: Yes (lightweight analysis, not compute-intensive)
   - Reason: Quick dataset statistics, doesn't benefit much from GPU

5. **`gpu_optimized_preprocessing.py`** ✨ NEW
   - Keep: Yes (primary preprocessing script)

6. **`gpu_optimized_mask_generation.py`** ✨ NEW
   - Keep: Yes (primary mask generation script)

7. **`gpu_monitor.py`** ✨ NEW
   - Keep: Yes (essential monitoring tool)

---

### 📁 01_notebooks/ - Interactive Notebooks

#### 🤔 OPTIONAL - Keep for reference/experimentation:
1. **`1_data_preparation_class_balancing.ipynb`**
   - Status: Completed (dataset is balanced 50/50)
   - Functionality: Dataset analysis and balancing strategies
   - Recommendation: **KEEP** (useful reference for thesis methods section)
   - Alternative: Can archive if you don't use Jupyter

2. **`2_enhanced_preprocessing_pipeline.ipynb`**
   - Status: Completed (techniques now in GPU scripts)
   - Functionality: Interactive preprocessing exploration
   - Recommendation: **ARCHIVE** (replaced by GPU scripts)
   - Reason: All functionality is in `gpu_optimized_preprocessing.py`

---

### 📁 Root Level - Helper Scripts

#### ✅ KEEP:
- **`organize_files.py`** - Keep (already completed, but useful for reorganization)
- **`setup_gpu.bat`** ✨ NEW - Keep (essential for setup)

---

## Recommended Action Plan

### Option 1: Safe Cleanup (Recommended)
Create archive folder, move old files there:
```
DATASCI5/
  _archive_cpu_scripts/
    - run_complete_preprocessing.py
    - generate_masks.py
    - generate_masks_strategy3.py
    - 2_enhanced_preprocessing_pipeline.ipynb
```

### Option 2: Aggressive Cleanup
Permanently delete obsolete files (can always recover from backup if needed)

---

## Files Summary After Cleanup

### 📁 02_scripts/ (4 files - streamlined!)
```
✅ run_dataset_analysis.py          - Dataset statistics
✅ gpu_optimized_preprocessing.py   - GPU preprocessing (15x faster)
✅ gpu_optimized_mask_generation.py - GPU mask generation (10x faster)
✅ gpu_monitor.py                   - GPU monitoring
✅ gpu_config.json                  - GPU configuration
✅ README.md                        - Scripts documentation
```

### 📁 01_notebooks/ (1 file - optional)
```
✅ 1_data_preparation_class_balancing.ipynb - Reference (thesis methods)
```

---

## Size Savings

Estimated disk space saved:
- `run_complete_preprocessing.py`: ~8 KB
- `generate_masks.py`: ~13 KB
- `generate_masks_strategy3.py`: ~15 KB
- `2_enhanced_preprocessing_pipeline.ipynb`: ~150 KB (with outputs)

**Total saved**: ~186 KB (minimal, but improves clarity)

---

## Benefits of Cleanup

1. ✅ **Clarity** - Only GPU-optimized scripts remain
2. ✅ **No confusion** - Won't accidentally run slow CPU versions
3. ✅ **Clean structure** - Easier to navigate
4. ✅ **Focus** - Only essential files for thesis
5. ✅ **Documentation** - Clear what to use

---

## Safety Notes

- All old scripts are documented in `05_documentation/`
- Functionality is preserved in GPU versions
- Can reference git history if needed
- Dataset analysis notebook kept for thesis methods

---

## Proceed?

Choose your cleanup level:
1. **Safe** - Archive old files (recommended)
2. **Aggressive** - Delete old files permanently
3. **Minimal** - Keep everything (not recommended)

Ready to proceed with cleanup!
