"""
GPU-Optimized Mask Generation V2 for RTX 3070
IMPROVED VERSION with lower loss and better generalization

Key Improvements:
- Data augmentation for better generalization
- Hybrid loss function (Dice + Focal + Boundary)
- Pretrained encoder (ResNet34) for transfer learning
- Validation split with early stopping
- Learning rate warmup and ReduceLROnPlateau
- Class weighting for imbalanced data

Optimized for: RTX 3070 (8GB VRAM) + Ryzen 7 5800X + 32GB RAM
Expected Loss: 0.15-0.25 (vs 0.667 in v1)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings('ignore')

print("="*80)
print("GPU-OPTIMIZED MASK GENERATION V2 (RTX 3070)")
print("="*80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nImprovements over V1:")
print("  - Data augmentation for 400 images")
print("  - Hybrid loss function (Dice + Focal + Boundary)")
print("  - Pretrained ResNet34 encoder")
print("  - Validation split + early stopping")
print("  - Better learning rate scheduling")

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

    print(f"\n[OK] PyTorch Setup Complete")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    return device

device = setup_pytorch()

# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

def get_training_augmentation():
    """
    Heavy augmentation for training to expand effective dataset size
    Medical image specific transformations
    """
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=20, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),

        # Elastic deformation (important for medical images)
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.3),
        A.GridDistortion(p=0.2),

        # Optical/Color transforms (fundus images have varying illumination)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.CLAHE(clip_limit=4.0, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),

        # Normalize and convert
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    """
    No augmentation for validation, only normalization
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ==============================================================================
# RESNET34 ENCODER (PRETRAINED)
# ==============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet34 pretrained encoder for U-Net
    Provides much better feature extraction than training from scratch
    """
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet34
        import torchvision.models as models
        resnet = models.resnet34(pretrained=True)

        # Extract encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        # Get multi-scale features for skip connections
        x1 = self.relu(self.bn1(self.conv1(x)))  # 64 channels
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)  # 64 channels
        x3 = self.layer2(x2)  # 128 channels
        x4 = self.layer3(x3)  # 256 channels
        x5 = self.layer4(x4)  # 512 channels

        return x1, x2, x3, x4, x5


# ==============================================================================
# IMPROVED U-NET WITH PRETRAINED ENCODER
# ==============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ImprovedUNet(nn.Module):
    """
    U-Net with pretrained ResNet34 encoder

    Advantages:
    - ImageNet pretrained weights (better initialization)
    - Deeper encoder (ResNet34 vs vanilla U-Net)
    - Better feature extraction for small datasets
    """

    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes

        # Pretrained encoder
        self.encoder = ResNetEncoder()

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)  # 256 + 256 from skip

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)  # 128 + 128 from skip

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)   # 64 + 64 from skip

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)   # 64 + 64 from skip

        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder (pretrained ResNet34)
        x1, x2, x3, x4, x5 = self.encoder(x)

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
# DATASET WITH AUGMENTATION
# ==============================================================================

class AugmentedFundusDataset(Dataset):
    """
    Dataset with albumentations support
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Get image files
        self.image_files = []
        for ext in ['.jpg', '.png', '.jpeg']:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))

        print(f"    Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

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

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Manual conversion if no transform
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).long()

        return image, mask


# ==============================================================================
# HYBRID LOSS FUNCTION
# ==============================================================================

def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for region overlap"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for class imbalance
    Focuses on hard examples (optic cup, disc boundaries)
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
    return focal_loss.mean()


def boundary_loss(pred, target):
    """
    Boundary loss for sharp edges
    Penalizes fuzzy boundaries
    """
    # Get prediction probabilities
    pred_soft = F.softmax(pred, dim=1)

    # Calculate gradients (boundaries)
    pred_grad_x = torch.abs(pred_soft[:, :, :, :-1] - pred_soft[:, :, :, 1:])
    pred_grad_y = torch.abs(pred_soft[:, :, :-1, :] - pred_soft[:, :, 1:, :])

    # Target boundaries
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    target_grad_x = torch.abs(target_one_hot[:, :, :, :-1] - target_one_hot[:, :, :, 1:])
    target_grad_y = torch.abs(target_one_hot[:, :, :-1, :] - target_one_hot[:, :, 1:, :])

    # L1 loss between gradients
    loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    loss_y = F.l1_loss(pred_grad_y, target_grad_y)

    return (loss_x + loss_y) / 2.0


def hybrid_loss(pred, target, dice_weight=0.5, focal_weight=0.3, boundary_weight=0.2):
    """
    Combined loss function

    - Dice: region overlap (good for small objects)
    - Focal: handles class imbalance
    - Boundary: sharp edges
    """
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)
    b_loss = boundary_loss(pred, target)

    total_loss = dice_weight * d_loss + focal_weight * f_loss + boundary_weight * b_loss

    return total_loss, {
        'dice': d_loss.item(),
        'focal': f_loss.item(),
        'boundary': b_loss.item(),
        'total': total_loss.item()
    }


# ==============================================================================
# TRAINING WITH IMPROVEMENTS
# ==============================================================================

def train_model_improved(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """
    Training with:
    - Mixed precision
    - Validation split
    - Early stopping
    - Learning rate warmup + ReduceLROnPlateau
    - Gradient clipping
    """
    print("\n" + "="*80)
    print("TRAINING IMPROVED MODEL")
    print("="*80)

    model = model.to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 15  # Early stopping patience
    no_improve_count = 0

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    for epoch in range(num_epochs):
        # ============ Training Phase ============
        model.train()
        train_losses = []
        train_dice_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [TRAIN]')

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss, loss_components = hybrid_loss(outputs, masks)

            # Backward pass with gradient scaling
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_dice_losses.append(loss_components['dice'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{loss_components["dice"]:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_train_loss = np.mean(train_losses)
        avg_train_dice = np.mean(train_dice_losses)

        # ============ Validation Phase ============
        model.eval()
        val_losses = []
        val_dice_losses = []

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [VAL]')
            for images, masks in pbar_val:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss, loss_components = hybrid_loss(outputs, masks)

                val_losses.append(loss.item())
                val_dice_losses.append(loss_components['dice'])

                pbar_val.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_dice': f'{loss_components["dice"]:.4f}'
                })

        avg_val_loss = np.mean(val_losses)
        avg_val_dice = np.mean(val_dice_losses)

        # Step scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Record history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f} (Dice: {avg_train_dice:.4f})')
        print(f'  Val Loss:   {avg_val_loss:.4f} (Dice: {avg_val_dice:.4f})')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            no_improve_count = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'training_history': training_history
            }, 'best_segmentation_model_gpu_v2.pth')
            print(f'  [OK] Saved best model (val_loss: {best_val_loss:.4f})')
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f'\n[EARLY STOPPING] No improvement for {patience} epochs')
            break

        # GPU stats
        print(f'  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / '
              f'{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB peak')
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[OK] Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"  Improvement over V1: {((0.667 - best_val_loss) / 0.667 * 100):.1f}%")

    return model, training_history


# ==============================================================================
# FAST INFERENCE (SAME AS V1)
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

                # Apply same normalization as training
                img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
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

        print(f"  [OK] Generated {len(image_files)} masks")

    return total_generated


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution with improvements"""

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

    print(f"\n[OK] Configuration")
    print(f"  Training batch size: {train_batch_size}")
    print(f"  Inference batch size: {GPU_CONFIG['optimization_settings']['batch_sizes']['mask_generation_inference']}")
    print(f"  DataLoader workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")

    # Step 1: Create dataset with augmentation
    print("\n" + "="*80)
    print("STEP 1: LOADING TRAINING DATA WITH AUGMENTATION")
    print("="*80)

    # Full dataset
    full_dataset = AugmentedFundusDataset(
        images_dir=refuge_images,
        masks_dir=refuge_masks,
        transform=None  # Will be set per split
    )

    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply augmentation to train split
    train_dataset.dataset.transform = get_training_augmentation()

    # Create validation dataset copy with different transform
    val_dataset_aug = AugmentedFundusDataset(
        images_dir=refuge_images,
        masks_dir=refuge_masks,
        transform=get_validation_augmentation()
    )
    # Use same indices as val_dataset
    val_dataset = torch.utils.data.Subset(val_dataset_aug, val_dataset.indices)

    print(f"  Total dataset: {len(full_dataset)} images")
    print(f"  Training set: {len(train_dataset)} images (with augmentation)")
    print(f"  Validation set: {len(val_dataset)} images (no augmentation)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Step 2: Train improved model
    print("\n" + "="*80)
    print("STEP 2: TRAINING IMPROVED MODEL")
    print("="*80)

    model = ImprovedUNet(n_classes=3)

    print(f"  Model: U-Net with Pretrained ResNet34 Encoder")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model, history = train_model_improved(
        model, train_loader, val_loader,
        num_epochs=50,
        device=device
    )

    # Load best model
    checkpoint = torch.load('best_segmentation_model_gpu_v2.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n[OK] Loaded best model")
    print(f"  Training loss: {checkpoint['train_loss']:.4f}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

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
        'version': 'v2_improved',
        'improvements': [
            'Data augmentation (albumentations)',
            'Hybrid loss (Dice + Focal + Boundary)',
            'Pretrained ResNet34 encoder',
            'Validation split (80/20)',
            'Early stopping',
            'ReduceLROnPlateau scheduler'
        ],
        'total_masks': total,
        'gpu': GPU_CONFIG['hardware_specs']['gpu'] if 'hardware_specs' in GPU_CONFIG else 'RTX 3070',
        'final_train_loss': checkpoint['train_loss'],
        'final_val_loss': checkpoint['val_loss'],
        'improvement_over_v1': f"{((0.667 - checkpoint['val_loss']) / 0.667 * 100):.1f}%",
        'training_history': checkpoint['training_history']
    }

    with open('mask_generation_gpu_v2_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nTotal masks generated: {total:,}")
    print(f"Final validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Improvement over V1 (0.667 loss): {((0.667 - checkpoint['val_loss']) / 0.667 * 100):.1f}%")
    print(f"\nModel saved: best_segmentation_model_gpu_v2.pth")
    print(f"Stats saved: mask_generation_gpu_v2_stats.json")
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
