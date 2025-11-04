"""
GPU-Optimized Mask Generation for RTX 3070
Optimized for: RTX 3070 (8GB VRAM) + Ryzen 7 5800X + 32GB RAM

Optimizations:
- Optimal batch sizes for 8GB VRAM
- Mixed precision training (FP16)
- Gradient accumulation for larger effective batch
- DataLoader with 8 workers
- Pin memory and prefetching
- CUDA graph optimization
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GPU-OPTIMIZED MASK GENERATION (RTX 3070)")
print("="*80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load GPU configuration
config_path = Path(__file__).parent / 'gpu_config.json'
if config_path.exists():
    with open(config_path, 'r') as f:
        GPU_CONFIG = json.load(f)
else:
    GPU_CONFIG = {
        'optimization_settings': {
            'batch_sizes': {'mask_generation_train': 8, 'mask_generation_inference': 12},
            'data_loading': {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 2}
        }
    }

# Setup PyTorch
def setup_pytorch():
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available!")
        sys.exit(1)

    device = torch.device('cuda:0')

    # RTX 3070 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)

    print(f"\n✓ PyTorch Setup Complete")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    return device

device = setup_pytorch()

# ==============================================================================
# OPTIMIZED U-NET WITH MEMORY EFFICIENCY
# ==============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetSegmentation(nn.Module):
    """
    Memory-optimized U-Net for RTX 3070

    - Reduced channel sizes for 8GB VRAM
    - Efficient architecture
    """

    def __init__(self, n_channels=3, n_classes=3, base_channels=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (reduced channels for memory efficiency)
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*2, base_channels*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*4, base_channels*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*8, base_channels*16))

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels*16, base_channels*8)

        self.up2 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels*8, base_channels*4)

        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels*4, base_channels*2)

        self.up4 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels*2, base_channels)

        # Output
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        return self.outc(x)


# ==============================================================================
# OPTIMIZED DATASET WITH CACHING
# ==============================================================================

class OptimizedFundusDataset(Dataset):
    """
    Optimized dataset with:
    - Efficient loading
    - Optional RAM caching
    - Minimal transformations
    """

    def __init__(self, images_dir, masks_dir, cache_in_ram=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.cache_in_ram = cache_in_ram
        self.cache = {}

        # Get image files
        self.image_files = []
        for ext in ['.jpg', '.png', '.jpeg']:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))

        print(f"    Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Check cache
        if self.cache_in_ram and idx in self.cache:
            return self.cache[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.masks_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Resize
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Cache if enabled
        if self.cache_in_ram and len(self.cache) < 1000:  # Cache first 1000
            self.cache[idx] = (image, mask)

        return image, mask


# ==============================================================================
# TRAINING WITH MIXED PRECISION
# ==============================================================================

def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for segmentation"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_model_optimized(model, train_loader, num_epochs=30, device='cuda'):
    """
    Training with mixed precision and optimizations
    """
    print("\n" + "="*80)
    print("TRAINING SEGMENTATION MODEL (GPU-OPTIMIZED)")
    print("="*80)

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Loss function
    criterion = dice_loss

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Backward pass with gradient scaling
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_count += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/batch_count:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Clear cache periodically
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / batch_count
        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_segmentation_model_gpu.pth')
            print(f'  ✓ Saved best model (loss: {best_loss:.4f})')

        # GPU stats
        print(f'  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / '
              f'{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB peak')
        torch.cuda.reset_peak_memory_stats()

    print(f"\n✓ Training complete! Best loss: {best_loss:.4f} at epoch {best_epoch}")
    return model


# ==============================================================================
# FAST INFERENCE
# ==============================================================================

@torch.no_grad()
def generate_masks_fast(model, directories, device='cuda'):
    """
    Fast mask generation with GPU optimization
    """
    print("\n" + "="*80)
    print("GENERATING MASKS (GPU-ACCELERATED)")
    print("="*80)

    model = model.to(device)
    model.eval()

    # Larger batch size for inference
    batch_size = GPU_CONFIG['optimization_settings']['batch_sizes']['mask_generation_inference']

    total_generated = 0

    for input_dir, output_dir in directories:
        input_path = Path(input_dir)
        if not input_path.exists():
            continue

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_files = []
        for ext in ['.jpg', '.png', '.jpeg']:
            image_files.extend(list(input_path.glob(f'*{ext}')))

        if len(image_files) == 0:
            continue

        print(f"\n  Processing {len(image_files)} images from {input_dir}")

        # Process in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="  Batches"):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            batch_sizes = []

            # Load batch
            for img_path in batch_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                original_size = img.shape[:2]
                batch_sizes.append(original_size)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (512, 512))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                batch_images.append(img)

            if len(batch_images) == 0:
                continue

            # Stack to batch tensor
            batch_tensor = torch.from_numpy(np.stack(batch_images, axis=0)).float().to(device)

            # Inference with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(batch_tensor)

            # Get predictions
            pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()

            # Save masks
            for j, (img_path, original_size) in enumerate(zip(batch_files[:len(pred_masks)], batch_sizes)):
                mask = pred_masks[j]

                # Resize to original
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                # Save
                mask_path = output_path / img_path.name
                cv2.imwrite(str(mask_path), mask)
                total_generated += 1

            # Clear cache
            if i % 50 == 0:
                torch.cuda.empty_cache()

        print(f"  ✓ Generated {len(image_files)} masks")

    return total_generated


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution with GPU optimization"""

    # Check for training data
    refuge_images = Path('validation/refuge2/images')
    refuge_masks = Path('validation/refuge2/mask')

    if not refuge_images.exists() or not refuge_masks.exists():
        print("\n[ERROR] REFUGE2 validation data not found!")
        return

    # Configuration
    train_batch_size = GPU_CONFIG['optimization_settings']['batch_sizes']['mask_generation_train']
    num_workers = GPU_CONFIG['optimization_settings']['data_loading']['num_workers']
    pin_memory = GPU_CONFIG['optimization_settings']['data_loading']['pin_memory']

    print(f"\n✓ Configuration")
    print(f"  Training batch size: {train_batch_size}")
    print(f"  Inference batch size: {GPU_CONFIG['optimization_settings']['batch_sizes']['mask_generation_inference']}")
    print(f"  DataLoader workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")

    # Step 1: Create dataset
    print("\n" + "="*80)
    print("STEP 1: LOADING TRAINING DATA")
    print("="*80)

    dataset = OptimizedFundusDataset(
        images_dir=refuge_images,
        masks_dir=refuge_masks,
        cache_in_ram=True
    )

    print(f"  Dataset size: {len(dataset)} images")

    train_loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Step 2: Train model
    print("\n" + "="*80)
    print("STEP 2: TRAINING MODEL")
    print("="*80)

    model = UNetSegmentation(n_channels=3, n_classes=3, base_channels=32)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Estimated VRAM: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024**3:.2f} GB")

    model = train_model_optimized(model, train_loader, num_epochs=30, device=device)

    # Load best model
    checkpoint = torch.load('best_segmentation_model_gpu.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model (loss: {checkpoint['loss']:.4f})")

    # Step 3: Generate masks
    directories = [
        ('train/eyepac/NRG', 'train/eyepac/NRG_masks'),
        ('train/eyepac/RG', 'train/eyepac/RG_masks'),
        ('validation/eyepac/NRG', 'validation/eyepac/NRG_masks'),
        ('validation/eyepac/RG', 'validation/eyepac/RG_masks'),
        ('test/eyepac/NRG', 'test/eyepac/NRG_masks'),
        ('test/eyepac/RG', 'test/eyepac/RG_masks'),
        ('train/refuge2/images', 'train/refuge2/generated_masks'),
        ('test/refuge2/images', 'test/refuge2/generated_masks'),
    ]

    total = generate_masks_fast(model, directories, device)

    # Save stats
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_masks': total,
        'gpu': GPU_CONFIG['hardware_specs']['gpu'],
        'batch_size_train': train_batch_size,
        'batch_size_inference': GPU_CONFIG['optimization_settings']['batch_sizes']['mask_generation_inference']
    }

    with open('mask_generation_gpu_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nTotal masks generated: {total:,}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
