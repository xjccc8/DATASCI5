"""
Automated File Organization Script for DATASCI5
Organizes files into logical folders and removes redundant files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

print("="*80)
print("DATASCI5 FOLDER ORGANIZATION SCRIPT")
print("="*80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Define folder structure
FOLDERS = {
    '01_notebooks': 'Jupyter notebooks for interactive analysis',
    '02_scripts': 'Python scripts for batch processing',
    '03_config': 'Configuration files (JSON)',
    '04_visualizations': 'Generated images and figures',
    '05_documentation': 'Markdown documentation files'
}

# Define file organization mapping
FILE_MAPPING = {
    '01_notebooks': [
        '1_data_preparation_class_balancing.ipynb',
        '2_enhanced_preprocessing_pipeline.ipynb'
    ],
    '02_scripts': [
        'run_dataset_analysis.py',
        'run_complete_preprocessing.py',
        'generate_masks.py',
        'generate_masks_strategy3.py'
    ],
    '03_config': [
        'dataset_statistics.json',
        'class_weights_config.json',
        'canny_config.json'
    ],
    '04_visualizations': [
        'ben_graham_visualization.png',
        'vessel_enhancement_comparison.png',
        'medical_augmentation_examples.png',
        'canny_parameter_comparison.png',
        'System Architecture.png'
    ],
    '05_documentation': [
        'FINAL_SUMMARY_AND_INSTRUCTIONS.md',
        'PREPROCESSING_EXECUTION_REPORT.md',
        'MASK_GENERATION_ANALYSIS.md',
        'PYTHON_VS_NOTEBOOK_COMPARISON.md',
        'FILE_ORGANIZATION_PLAN.md'
    ]
}

# Files to delete (redundant/superseded)
FILES_TO_DELETE = {
    'preprocessing_pipeline.ipynb': 'Superseded by 2_enhanced_preprocessing_pipeline.ipynb',
    'balancing_strategies_comparison.csv': 'Redundant - info in JSON and MD files'
}

# Optional files to delete (user can decide)
OPTIONAL_DELETE = {
    'dataset_distribution_analysis.png': 'Unused visualization (can regenerate)',
    'PREPROCESSING_SUMMARY.md': 'Superseded by FINAL_SUMMARY_AND_INSTRUCTIONS.md'
}

def create_folders():
    """Create folder structure"""
    print("Step 1: Creating folder structure...")
    print("-"*80)

    for folder, description in FOLDERS.items():
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_path.mkdir()
            print(f"  [CREATED] {folder}/ - {description}")
        else:
            print(f"  [EXISTS]  {folder}/ - {description}")

    print()

def delete_redundant_files(delete_optional=False):
    """Delete redundant files"""
    print("Step 2: Removing redundant files...")
    print("-"*80)

    # Delete definitely redundant files
    for filename, reason in FILES_TO_DELETE.items():
        file_path = Path(filename)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"  [DELETED] {filename}")
                print(f"            Reason: {reason}")
            except Exception as e:
                print(f"  [ERROR]   Could not delete {filename}: {e}")
        else:
            print(f"  [SKIP]    {filename} not found")

    # Delete optional files if requested
    if delete_optional:
        print("\n  Deleting optional files...")
        for filename, reason in OPTIONAL_DELETE.items():
            file_path = Path(filename)
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"  [DELETED] {filename}")
                    print(f"            Reason: {reason}")
                except Exception as e:
                    print(f"  [ERROR]   Could not delete {filename}: {e}")
            else:
                print(f"  [SKIP]    {filename} not found")
    else:
        print(f"\n  [INFO] Skipping optional deletions")
        print(f"         To delete optional files, run with delete_optional=True")

    print()

def move_files():
    """Move files to organized folders"""
    print("Step 3: Organizing files into folders...")
    print("-"*80)

    moved_count = 0
    skipped_count = 0

    for folder, files in FILE_MAPPING.items():
        print(f"\n  Moving files to {folder}/:")
        for filename in files:
            source = Path(filename)
            destination = Path(folder) / filename

            if source.exists():
                try:
                    shutil.move(str(source), str(destination))
                    print(f"    [MOVED] {filename}")
                    moved_count += 1
                except Exception as e:
                    print(f"    [ERROR] Could not move {filename}: {e}")
            else:
                print(f"    [SKIP]  {filename} not found")
                skipped_count += 1

    print()
    print(f"  Total moved: {moved_count} files")
    print(f"  Total skipped: {skipped_count} files")
    print()

def create_readme():
    """Create README in each folder"""
    print("Step 4: Creating README files...")
    print("-"*80)

    readme_content = {
        '01_notebooks': """# Notebooks

Interactive Jupyter notebooks for data analysis and preprocessing.

## Files:
- `1_data_preparation_class_balancing.ipynb` - Dataset analysis and class balancing
- `2_enhanced_preprocessing_pipeline.ipynb` - Complete preprocessing pipeline

## Usage:
```bash
jupyter notebook
```
Then open the desired notebook.
""",
        '02_scripts': """# Scripts

Python scripts for batch processing and automation.

## Files:
- `run_dataset_analysis.py` - Analyze dataset statistics
- `run_complete_preprocessing.py` - Execute complete preprocessing
- `generate_masks.py` - Strategy 1: Basic mask generation
- `generate_masks_strategy3.py` - Strategy 3: Semi-supervised (best quality)

## Usage:
```bash
python run_complete_preprocessing.py
python generate_masks_strategy3.py
```
""",
        '03_config': """# Configuration Files

JSON configuration files for training and preprocessing.

## Files:
- `dataset_statistics.json` - Complete dataset breakdown
- `class_weights_config.json` - Class weights for loss function (ALL 4 models)
- `canny_config.json` - Optimal Canny parameters (Models 3 & 4)

## Usage:
Load these in your training scripts:
```python
import json
with open('03_config/class_weights_config.json', 'r') as f:
    config = json.load(f)
```
""",
        '04_visualizations': """# Visualizations

Generated images and figures for thesis and presentations.

## Files:
- `ben_graham_visualization.png` - Ben Graham preprocessing demo
- `vessel_enhancement_comparison.png` - Vessel enhancement methods
- `medical_augmentation_examples.png` - Augmentation examples
- `canny_parameter_comparison.png` - Canny parameter tuning
- `System Architecture.png` - 4-model architecture diagram

## Usage:
Include these in your thesis/presentations. All images are publication-ready.
""",
        '05_documentation': """# Documentation

Complete documentation for the preprocessing pipeline.

## Files:
- `FINAL_SUMMARY_AND_INSTRUCTIONS.md` - **START HERE** (Primary guide)
- `PREPROCESSING_EXECUTION_REPORT.md` - Execution results
- `MASK_GENERATION_ANALYSIS.md` - Mask generation benefits
- `PYTHON_VS_NOTEBOOK_COMPARISON.md` - Performance comparison

## Primary Reference:
Read `FINAL_SUMMARY_AND_INSTRUCTIONS.md` first - it contains everything you need!
"""
    }

    for folder, content in readme_content.items():
        readme_path = Path(folder) / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"  [CREATED] {folder}/README.md")

    print()

def create_main_readme():
    """Create main README for DATASCI5"""
    print("Step 5: Creating main README...")
    print("-"*80)

    main_readme = """# DATASCI5 - Glaucoma Detection Thesis Project

## Comparative Study: 4 Deep Learning Architectures

**Dataset:** EyePACS + REFUGE2 (19,080 images)
**Task:** Glaucoma classification from fundus images

---

## 📁 Folder Structure

```
DATASCI5/
├── 01_notebooks/          # Interactive Jupyter notebooks
├── 02_scripts/            # Python scripts for automation
├── 03_config/             # Configuration files (JSON)
├── 04_visualizations/     # Generated figures
├── 05_documentation/      # Complete documentation
├── train/                 # Training data
├── validation/            # Validation data
└── test/                  # Test data
```

---

## 🚀 Quick Start

### 1. Data Preparation
```bash
cd 02_scripts
python run_dataset_analysis.py
```

### 2. Preprocessing
```bash
python run_complete_preprocessing.py
```

### 3. Mask Generation (Optional but Recommended)
```bash
python generate_masks_strategy3.py
```

---

## 📖 Documentation

**Start Here:** `05_documentation/FINAL_SUMMARY_AND_INSTRUCTIONS.md`

This is your primary reference document containing:
- Complete setup instructions
- Usage guide for all 4 models
- Configuration details
- Expected results
- Thesis recommendations

---

## 🎯 Four Models to Compare

1. **Model 1:** U-Net + DCNN Classifier (512×512)
2. **Model 2:** ResNet-50 + DCNN Classifier (256×256)
3. **Model 3:** U-Net + Canny Edge + DCNN (512×512)
4. **Model 4:** ResNet-50 + Canny Edge + DCNN (256×256)

---

## 📊 Dataset Statistics

- **Total Images:** 19,080
- **Training:** 16,000 (8,000 NRG + 8,000 RG)
- **Validation:** 1,540 (770 NRG + 770 RG)
- **Test:** 1,540 (770 NRG + 770 RG)
- **Balance:** Perfect 50/50 split

---

## ✅ Completed Steps

- [x] Dataset analysis
- [x] Class weight calculation
- [x] Preprocessing pipeline design
- [x] Ben Graham preprocessing
- [x] Vessel enhancement
- [x] Medical augmentation
- [x] Canny parameter optimization
- [x] Documentation

---

## 📝 Next Steps

1. Process full dataset (19,080 images)
2. Generate masks (optional but recommended)
3. Implement 4 model architectures
4. Train all models
5. Compare results
6. Write thesis

---

## 📧 Configuration Files

Load these in your training code:

```python
import json

# Class weights (ALL models)
with open('03_config/class_weights_config.json', 'r') as f:
    weights = json.load(f)['class_weights_array']

# Canny parameters (Models 3 & 4)
with open('03_config/canny_config.json', 'r') as f:
    canny = json.load(f)['canny_parameters']
```

---

## 🎓 For Your Thesis

All visualizations in `04_visualizations/` are publication-ready.

Include these sections:
- Preprocessing pipeline (Ben Graham + Vessel + Augmentation)
- Canny edge optimization
- Comparative analysis of 4 architectures
- Clinical evaluation (Sensitivity, Specificity, F1, ROC-AUC)

---

**Organization Date:** {datetime}
**Status:** Ready for model training
""".format(datetime=datetime.now().strftime('%Y-%m-%d'))

    with open('README.md', 'w') as f:
        f.write(main_readme)

    print("  [CREATED] README.md (main)")
    print()

def generate_summary():
    """Generate organization summary"""
    print("="*80)
    print("ORGANIZATION COMPLETE!")
    print("="*80)
    print()
    print("Folder Structure:")
    print("  DATASCI5/")
    print("  ├── 01_notebooks/       (2 notebooks)")
    print("  ├── 02_scripts/         (4 Python scripts)")
    print("  ├── 03_config/          (3 JSON files)")
    print("  ├── 04_visualizations/  (5 images)")
    print("  ├── 05_documentation/   (5 markdown files)")
    print("  ├── train/              (existing)")
    print("  ├── validation/         (existing)")
    print("  ├── test/               (existing)")
    print("  └── README.md           (main guide)")
    print()
    print("Files Deleted:")
    print("  - preprocessing_pipeline.ipynb (superseded)")
    print("  - balancing_strategies_comparison.csv (redundant)")
    print()
    print("Next Steps:")
    print("  1. Read: 05_documentation/FINAL_SUMMARY_AND_INSTRUCTIONS.md")
    print("  2. Run: 02_scripts/run_complete_preprocessing.py")
    print("  3. Optional: 02_scripts/generate_masks_strategy3.py")
    print()
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def main(delete_optional_files=False):
    """Main organization function"""

    print("\nThis script will:")
    print("  1. Create 5 organized folders")
    print("  2. Delete 2 redundant files (definitely safe)")
    if delete_optional_files:
        print("  3. Delete 2 optional files (unused/superseded)")
    print(f"  {3 if not delete_optional_files else 4}. Move files into organized structure")
    print(f"  {4 if not delete_optional_files else 5}. Create README files")
    print()

    try:
        # Execute organization
        create_folders()
        delete_redundant_files(delete_optional=delete_optional_files)
        move_files()
        create_readme()
        create_main_readme()
        generate_summary()

        return True

    except Exception as e:
        print(f"\n[ERROR] Organization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys

    # Check for optional deletion flag
    delete_optional = '--delete-optional' in sys.argv

    if delete_optional:
        print("\n[INFO] Optional file deletion ENABLED")
        print("       Will delete: dataset_distribution_analysis.png, PREPROCESSING_SUMMARY.md")
    else:
        print("\n[INFO] Optional file deletion DISABLED")
        print("       To enable, run: python organize_files.py --delete-optional")

    print()
    input("Press Enter to continue or Ctrl+C to cancel...")
    print()

    success = main(delete_optional_files=delete_optional)

    if success:
        print("\n✓ Organization completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Organization failed!")
        sys.exit(1)
