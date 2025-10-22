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

1. Data Preparation:    cd 02_scripts && python run_dataset_analysis.py
2. Preprocessing:       python run_complete_preprocessing.py
3. Mask Generation:     python generate_masks_strategy3.py

---

## Primary Documentation

START HERE: 05_documentation/FINAL_SUMMARY_AND_INSTRUCTIONS.md

---

## Four Models

1. Model 1: U-Net + DCNN Classifier (512x512)
2. Model 2: ResNet-50 + DCNN Classifier (256x256)
3. Model 3: U-Net + Canny Edge + DCNN (512x512)
4. Model 4: ResNet-50 + Canny Edge + DCNN (256x256)

---

Organization Date: 2025-10-22
Status: Ready for model training
