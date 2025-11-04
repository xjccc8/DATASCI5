# DATASCI5 - Glaucoma Detection Thesis Project

## Comparative Study: 4 Deep Learning Architectures

**Dataset:** EyePACS + REFUGE2 (19,080 images)
**Task:** Glaucoma classification from fundus images

---

## Folder Structure

DATASCI5/
  01_notebooks/          - Interactive Jupyter notebooks
  02_scripts/            - Python scripts for automation
  03_config/             - Configuration files (JSON)
  04_visualizations/     - Generated figures
  05_documentation/      - Complete documentation
  train/                 - Training data
  validation/            - Validation data
  test/                  - Test data

---

## Quick Start

### GPU-Accelerated (RTX 3070 - RECOMMENDED)
1. GPU Setup:           setup_gpu.bat (one-time, 5 min)
2. Verify GPU:          python 02_scripts/gpu_monitor.py check
3. Preprocessing:       python 02_scripts/gpu_optimized_preprocessing.py (~4 min)
4. Mask Generation:     python 02_scripts/gpu_optimized_mask_generation.py (~52 min)

### CPU-Only (Slower)
1. Data Preparation:    cd 02_scripts && python run_dataset_analysis.py
2. Preprocessing:       python run_complete_preprocessing.py (~60 min)
3. Mask Generation:     python generate_masks_strategy3.py (~3.5 hours)

---

## Primary Documentation

**GPU Setup:** GPU_OPTIMIZATION_SUMMARY.md (NEW - Start here for GPU!)
**Project Guide:** 05_documentation/FINAL_SUMMARY_AND_INSTRUCTIONS.md
**GPU Guide:** GPU_SETUP_GUIDE.md
**Quick Ref:** GPU_QUICK_REFERENCE.md

---

## Four Models

1. Model 1: U-Net + DCNN Classifier (512x512)
2. Model 2: ResNet-50 + DCNN Classifier (256x256)
3. Model 3: U-Net + Canny Edge + DCNN (512x512)
4. Model 4: ResNet-50 + Canny Edge + DCNN (256x256)

---

Organization Date: 2025-10-22
GPU Optimization: 2025-11-03
Status: ✅ GPU-optimized and ready for training (10-15x faster!)

Hardware: RTX 3070 (8GB) + Ryzen 7 5800X + 32GB RAM
Expected Speedup: 15x preprocessing, 10x mask gen, 10-12x training
