# Configuration Files

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
