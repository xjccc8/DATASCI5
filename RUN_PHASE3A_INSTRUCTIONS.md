# Model 1 Phase 3A - Multi-Task Learning

## Quick Start Guide

### What Was Fixed?

**Problem**: Original Phase 3A couldn't find ground-truth masks because:
- `processed_gpu` contains only EyePACS images (no REFUGE-2)
- REFUGE-2 images with masks are in `train/refuge2/images/`
- Naming mismatch: EyePACS uses `EyePACS-*.jpg`, REFUGE-2 uses `g*.jpg` and `n*.jpg`

**Solution**: Updated dataset loader to combine BOTH sources:
1. **EyePACS images** from `processed_gpu` (no masks) → ~7,600 images
2. **REFUGE-2 images** from `refuge2/images` (with ground-truth masks) → ~400 images
3. **Total**: ~8,000 training images, with 400 having ground-truth segmentation masks

### Expected Dataset Size

| Split | EyePACS (no masks) | REFUGE-2 (with masks) | Total | Masks |
|-------|-------------------|----------------------|-------|-------|
| Train | ~7,600 images | ~400 images | ~8,000 | ~400 |
| Val | ~770 images | ~400 images | ~1,170 | ~400 |
| Test | ~770 images | ~400 images | ~1,170 | ~400 |

### How to Run Manually

```bash
cd C:\Users\natha\Documents\DATASCI5
python 03_models\model1_unet_densenet121_phase3a.py
```

**Expected Output** (first few lines):
```
================================================================================
Model 1 PHASE 3A: Multi-Task Learning Training
================================================================================

Phase 3A Improvements:
- U-Net decoder: TRAINABLE (learns task-specific features)
- Ground-truth mask supervision from REFUGE-2 dataset
- Dice Loss for segmentation task
- Joint optimization: Classification + Segmentation

Device: cuda

Loading datasets...
Loading TRAIN set (EyePACS + REFUGE-2)...
Dataset: 8400 images (4040 RG, 4360 NRG)
  Images with ground-truth masks: 400  <-- THIS SHOULD BE 400, NOT 0!

Loading VALIDATION set (EyePACS + REFUGE-2)...
Dataset: 1170 images (...)
  Images with ground-truth masks: 400

Loading TEST set (EyePACS + REFUGE-2)...
Dataset: 1170 images (...)
  Images with ground-truth masks: 400
```

**✅ If you see "400 masks" in each split** → Working correctly!
**❌ If you see "0 masks"** → Data path issue, let me know

### Training Configuration

- **Batch Size**: 8 (reduced for multi-task learning memory)
- **Max Epochs**: 75
- **Early Stopping**: 20 epochs patience
- **Learning Rates**:
  - U-Net decoder: 1e-5 (fine-tuning)
  - DenseNet: 1e-4 (standard)
- **Loss Function**:
  - Classification: Focal Loss (α=0.25, γ=2.0)
  - Segmentation: Dice Loss
  - Total Loss = 1.0 × Classification + 0.5 × Segmentation

### Expected Training Time

- **Per Epoch**: ~2-3 minutes (1,050 batches @ 8 images/batch)
- **Total Training**: 2-3 hours (with early stopping expected around epoch 40-50)
- **GPU Memory**: ~5.5-6 GB (slightly higher than Phase 2 due to segmentation head)

### Expected Performance Improvement

| Metric | Phase 2 Baseline | Phase 3A Target | Improvement |
|--------|------------------|-----------------|-------------|
| Accuracy | 79.48% | **85-88%** | **+5.5-8.5%** |
| AUC | 0.8838 | **0.92-0.95** | **+0.036-0.066** |
| Sensitivity | 80.52% | **87-92%** | **+6.5-11.5%** |

### Key Files Generated

After training completes:

1. **`model1_best_phase3a.pth`** - Best model checkpoint
2. **`model1_results_phase3a.json`** - Complete training history and metrics

### Monitoring Training

Watch for these indicators:

**Good Signs**:
- ✅ Segmentation loss decreases (starts ~0.6-0.7, should reach ~0.2-0.3)
- ✅ Classification loss stable/decreasing
- ✅ Validation AUC increases steadily
- ✅ Both losses balanced (not one dominating)

**Potential Issues**:
- ⚠️ Segmentation loss = 0.0 throughout → Masks not loading (check dataset output)
- ⚠️ GPU OOM error → Reduce batch size to 4 in Config.BATCH_SIZE
- ⚠️ Validation AUC not improving → May need to adjust λ_segmentation weight

### Troubleshooting

**If training fails to start:**

1. Check CUDA availability:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show RTX 3070
```

2. Verify data paths exist:
```bash
ls processed_gpu/train/RG | wc -l  # Should be ~4000
ls train/refuge2/images | wc -l    # Should be ~400
ls train/refuge2/mask | wc -l      # Should be ~400
```

3. Test mask loading:
```python
import cv2
import numpy as np
mask = cv2.imread('train/refuge2/mask/g0001.bmp', cv2.IMREAD_GRAYSCALE)
print(f"Shape: {mask.shape}, Values: {np.unique(mask)}")
# Should show: Shape: (2056, 2124), Values: [0, 128, 255]
```

### After Training Completes

Compare with Phase 2 results:

```python
import json

# Load Phase 2 results
with open('model1_results_improved.json') as f:
    phase2 = json.load(f)

# Load Phase 3A results
with open('model1_results_phase3a.json') as f:
    phase3a = json.load(f)

print(f"Phase 2 Test AUC: {phase2['test_metrics']['auc']:.4f}")
print(f"Phase 3A Test AUC: {phase3a['test_metrics']['auc']:.4f}")
print(f"Improvement: {(phase3a['test_metrics']['auc'] - phase2['test_metrics']['auc']):.4f}")
```

### Next Steps After Phase 3A

Once Phase 3A training completes successfully:

1. ✅ **Validate performance gains** (should see +4-7% improvement)
2. ✅ **Proceed to Models 2, 3, 4** for comprehensive comparison
3. ✅ **Implement Test-Time Augmentation** for additional +2-3% boost
4. ✅ **Final comparison** of all models to select best performer

---

## Technical Details

### Multi-Task Learning Architecture

```
Input Image (512×512×3)
      ↓
┌─────────────────┐
│   U-Net Decoder │ ← TRAINABLE (3.2M params)
│  (ResNet34 enc) │ ← FROZEN (21.3M params)
└─────────────────┘
      ↓
 Segmentation Logits (512×512×3)
      ↓
┌─────────────────┐
│   Dice Loss     │ ← Ground-truth masks (REFUGE-2 only)
└─────────────────┘
      ↓
  Resize to 224×224
      ↓
┌─────────────────┐
│  DenseNet-121   │ ← TRAINABLE (7.6M params)
└─────────────────┘
      ↓
 Classification Logits (1)
      ↓
┌─────────────────┐
│   Focal Loss    │ ← All images
└─────────────────┘
```

### Why This Works

1. **Trainable U-Net Decoder**: Learns glaucoma-specific segmentation features
2. **Ground-Truth Supervision**: High-quality masks guide learning
3. **Joint Optimization**: Segmentation improves classification features
4. **No Train/Test Mismatch**: Same U-Net used during training and inference

### Key Innovation vs Phase 2

| Aspect | Phase 2 | Phase 3A |
|--------|---------|----------|
| U-Net | Frozen | **Decoder Trainable** |
| Masks | V5.1 generated (62.5% Dice) | **Ground-truth (100% accurate)** |
| Supervision | Classification only | **Classification + Segmentation** |
| Learning | Single-task | **Multi-task** |

---

**Ready to run!** Just execute:
```bash
python 03_models\model1_unet_densenet121_phase3a.py
```

And monitor the output to ensure "400 masks" appear in each dataset split.
