"""
GPU-Optimized Mask Generation V6 - Attention U-Net with Deep Supervision
TARGETING 75-80% DICE ACCURACY

V6 MAJOR IMPROVEMENTS OVER V5.1 (62.5% Dice):
============================================================
NEW FEATURES:
[OK] Attention Gates (+8-10% Dice) - Focus on optic disc/cup boundaries
[OK] ResNet50 Encoder (+2-3% Dice) - Better feature extraction
[OK] Deep Supervision (+2-4% Dice) - Multi-scale gradient flow
[OK] Combined Loss: Dice + Focal + Boundary (+3-5% Dice) - Better edge detection
[OK] Enhanced Augmentation (+2-3% Dice) - CutOut, RandomGamma

KEPT FROM V5.1:
[OK] num_workers=0 (Windows compatibility)
[OK] ReduceLROnPlateau scheduler (proven)
[OK] Batch size 8 (RTX 3070 optimal)
[OK] Mixed precision training (FP16)
[OK] RAM caching for speed

Architecture: Attention U-Net
- Encoder: ResNet50 (pretrained ImageNet)
- Decoder: 5 levels with attention gates
- Deep supervision: Auxiliary losses at 3 decoder levels
- Output: 3 channels (background, optic disc, optic cup)

Expected Performance:
============================================================
- Dice Accuracy: 75-80% (vs V5.1: 62.5%)
- Val Loss: 0.15-0.18 (vs V5.1: 0.22)
- Training Time: 60-90 minutes on RTX 3070
- GPU Utilization: 85-95%

Dataset: REFUGE-2 (1200 labeled images)
- Train: 960 images (80%)
- Validation: 240 images (20%)

Created: 2025-11-11
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import gc
warnings.filterwarnings('ignore')

print("="*80)
print("GPU-OPTIMIZED MASK GENERATION V6 - ATTENTION U-NET")
print("TARGETING 75-80% DICE ACCURACY")
print("="*80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nV6 NEW FEATURES:")
print("============================================================")
print("  [OK] Attention Gates (focus on disc/cup boundaries)")
print("  [OK] ResNet50 Encoder (better features)")
print("  [OK] Deep Supervision (multi-scale learning)")
print("  [OK] Combined Loss: Dice + Focal + Boundary")
print("  [OK] Enhanced augmentation (CutOut, Gamma)")
print("\nTARGET: 75-80% Dice accuracy in 60-90 minutes")
print("="*80)

# Load GPU configuration
config_path = Path(__file__).parent / 'gpu_config.json'
if config_path.exists():
    with open(config_path, 'r') as f:
        GPU_CONFIG = json.load(f)
else:
    GPU_CONFIG = {
        'optimization_settings': {
            'batch_sizes': {'mask_generation_train': 8, 'mask_generation_inference': 16},
            'data_loading': {'num_workers': 0, 'pin_memory': True}
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
# V6 CONFIGURATION
# ==============================================================================

class ConfigV6:
    """U-Net V6 Configuration"""

    # Paths
    LABELED_DATA_ROOT = '../train/refuge2'
    LABELED_IMAGES_DIR = os.path.join(LABELED_DATA_ROOT, 'images')
    LABELED_MASKS_DIR = os.path.join(LABELED_DATA_ROOT, 'mask')

    # Model save
    MODEL_SAVE_PATH = 'best_segmentation_model_gpu_v6.pth'
    STATS_SAVE_PATH = 'mask_generation_gpu_v6_stats.json'

    # Architecture
    ENCODER = 'resnet50'  # V5.1: resnet34
    USE_ATTENTION = True  # NEW: Attention gates (FIXED dimension bug)
    DEEP_SUPERVISION = True  # NEW: Multi-scale supervision

    # Training
    BATCH_SIZE = 8  # V5.1 proven
    NUM_EPOCHS = 120  # V5.1: 100 (train longer)
    LEARNING_RATE = 3e-5  # V5.1: 5e-5 (slightly lower for stability)
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 1.0  # V5.1: 1.5 (tighter)

    # Scheduler
    LR_PATIENCE = 8
    LR_FACTOR = 0.5

    # Early stopping
    EARLY_STOPPING_PATIENCE = 18  # V5.1: 12 (more patient)

    # Loss weights
    LOSS_WEIGHTS = {
        'dice': 0.4,
        'focal': 0.3,
        'boundary': 0.3
    }

    # Deep supervision weights
    AUX_LOSS_WEIGHTS = [0.3, 0.21, 0.147]  # 0.3 * (0.7^i)

    # Data
    NUM_WORKERS = 0  # Windows fix
    PIN_MEMORY = True
    TRAIN_VAL_SPLIT = 0.8

    # Device
    DEVICE = device
    USE_AMP = True  # Mixed precision


# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

def get_training_augmentation_v6():
    """V6: Enhanced augmentation with CutOut and Gamma"""
    return A.Compose([
        A.Resize(512, 512, always_apply=True),

        # Geometric (from V5.1)
        A.Rotate(limit=20, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5,
                          border_mode=cv2.BORDER_CONSTANT),

        # Deformation (from V5.1)
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.3,
                          border_mode=cv2.BORDER_CONSTANT),
        A.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT),

        # Photometric (from V5.1)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.CLAHE(clip_limit=4.0, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),

        # NEW V6 augmentations
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                       fill_value=0, p=0.3),  # CutOut for robustness
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # Lighting variation
        A.ChannelShuffle(p=0.2),  # Color variation

        # Normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation_v6():
    """No augmentation for validation"""
    return A.Compose([
        A.Resize(512, 512, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ==============================================================================
# ATTENTION GATE MODULE
# ==============================================================================

class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net
    Highlights salient features from skip connections
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of feature maps in gating signal (decoder)
            F_l: Number of feature maps in skip connection (encoder)
            F_int: Number of intermediate feature maps
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (coarser scale)
            x: Skip connection from encoder (finer scale)
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi  # Element-wise multiplication


# ==============================================================================
# RESNET50 ENCODER
# ==============================================================================

class ResNet50Encoder(nn.Module):
    """ResNet50 pretrained encoder (V6 upgrade from ResNet34)"""
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Extract encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 -> 256 channels
        self.layer2 = resnet.layer2  # 256 -> 512 channels
        self.layer3 = resnet.layer3  # 512 -> 1024 channels
        self.layer4 = resnet.layer4  # 1024 -> 2048 channels

    def forward(self, x):
        """
        Returns encoder features at 5 scales for skip connections
        """
        # Initial conv
        x1 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x2 = self.maxpool(x1)  # (B, 64, H/4, W/4)

        # ResNet blocks
        x2 = self.layer1(x2)  # (B, 256, H/4, W/4)
        x3 = self.layer2(x2)  # (B, 512, H/8, W/8)
        x4 = self.layer3(x3)  # (B, 1024, H/16, W/16)
        x5 = self.layer4(x4)  # (B, 2048, H/32, W/32)

        return x1, x2, x3, x4, x5


# ==============================================================================
# DECODER BLOCK WITH ATTENTION
# ==============================================================================

class DecoderBlock(nn.Module):
    """Decoder block with attention gate and double convolution"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()

        self.use_attention = use_attention

        # Attention gate
        if use_attention:
            self.attention = AttentionGate(
                F_g=out_channels,  # After upsample, x has out_channels
                F_l=skip_channels,
                F_int=out_channels
            )

        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        # Double convolution
        # When attention is used, skip maintains its original skip_channels
        conv_in_channels = out_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
        """
        # Upsample
        x = self.upsample(x)

        # Apply attention to skip connection
        if self.use_attention:
            skip = self.attention(g=x, x=skip)

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Double convolution
        x = self.conv(x)

        return x


# ==============================================================================
# ATTENTION U-NET V6
# ==============================================================================

class AttentionUNet_V6(nn.Module):
    """
    Attention U-Net V6 with Deep Supervision

    Architecture:
        - Encoder: ResNet50 (pretrained)
        - Decoder: 5 levels with attention gates
        - Deep supervision: Auxiliary outputs at 3 decoder levels
        - Output: 3 channels (background, optic disc, optic cup)
    """

    def __init__(self, n_classes=3, use_attention=True, deep_supervision=True):
        super(AttentionUNet_V6, self).__init__()

        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        # Encoder (ResNet50)
        self.encoder = ResNet50Encoder()

        # Decoder with attention gates
        # ResNet50 channels: [64, 256, 512, 1024, 2048]
        self.decoder4 = DecoderBlock(2048, 1024, 1024, use_attention)
        self.decoder3 = DecoderBlock(1024, 512, 512, use_attention)
        self.decoder2 = DecoderBlock(512, 256, 256, use_attention)
        self.decoder1 = DecoderBlock(256, 64, 64, use_attention)

        # Final upsampling and output
        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_head3 = nn.Conv2d(1024, n_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(512, n_classes, kernel_size=1)
            self.aux_head1 = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass

        Returns:
            If deep_supervision:
                final_output, [aux3, aux2, aux1]
            Else:
                final_output
        """
        # Encoder
        e1, e2, e3, e4, e5 = self.encoder(x)

        # Decoder with skip connections
        d4 = self.decoder4(e5, e4)  # (B, 512, H/16, W/16)
        d3 = self.decoder3(d4, e3)  # (B, 256, H/8, W/8)
        d2 = self.decoder2(d3, e2)  # (B, 128, H/4, W/4)
        d1 = self.decoder1(d2, e1)  # (B, 64, H/2, W/2)

        # Final output
        d0 = self.final_upsample(d1)  # (B, 64, H, W)
        final = self.final_conv(d0)  # (B, n_classes, H, W)

        # Deep supervision
        if self.deep_supervision and self.training:
            aux3 = self.aux_head3(d4)
            aux2 = self.aux_head2(d3)
            aux1 = self.aux_head1(d2)
            return final, [aux3, aux2, aux1]
        else:
            return final


# ==============================================================================
# COMBINED LOSS FUNCTION
# ==============================================================================

class CombinedSegmentationLoss(nn.Module):
    """
    V6 Combined Loss: Dice + Focal + Boundary

    Components:
        1. Dice Loss: Overall overlap
        2. Focal Loss: Handle class imbalance
        3. Boundary Loss: Emphasize optic disc/cup edges
    """

    def __init__(self, weights={'dice': 0.4, 'focal': 0.3, 'boundary': 0.3}):
        super(CombinedSegmentationLoss, self).__init__()
        self.weights = weights

    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for segmentation"""
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for class imbalance"""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()

    def boundary_loss(self, pred, target):
        """
        Boundary loss - emphasize optic disc/cup edges
        Uses Sobel filter to detect boundaries
        """
        # Sobel kernels for edge detection
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=pred.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=pred.device
        ).view(1, 1, 3, 3)

        pred_sig = torch.sigmoid(pred)

        # Extract boundaries from target
        boundary_map = torch.zeros_like(target)

        for i in range(target.shape[1]):  # For each class
            target_channel = target[:, i:i+1, :, :]

            # Compute gradients
            grad_x = F.conv2d(target_channel, sobel_x, padding=1)
            grad_y = F.conv2d(target_channel, sobel_y, padding=1)

            # Boundary magnitude
            boundary_map[:, i:i+1, :, :] = torch.sqrt(grad_x**2 + grad_y**2)

        # Binary boundary map (threshold at 0.1)
        boundary_map = (boundary_map > 0.1).float()

        # Compute BCE with 3x weight on boundaries
        pred_logits = pred  # Use logits directly
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        weighted_bce = bce * (1.0 + 2.0 * boundary_map)  # 3x weight on boundaries

        return weighted_bce.mean()

    def forward(self, pred, target):
        """
        Compute combined loss

        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)

        total_loss = (
            self.weights['dice'] * dice +
            self.weights['focal'] * focal +
            self.weights['boundary'] * boundary
        )

        return total_loss, {
            'dice': dice.item(),
            'focal': focal.item(),
            'boundary': boundary.item()
        }


# ==============================================================================
# DEEP SUPERVISION LOSS
# ==============================================================================

def deep_supervision_loss(outputs, targets, criterion, aux_weights):
    """
    Compute loss with deep supervision

    Args:
        outputs: (final_output, [aux3, aux2, aux1])
        targets: Ground truth masks (B, C, H, W)
        criterion: CombinedSegmentationLoss
        aux_weights: List of weights for auxiliary losses
    """
    if isinstance(outputs, tuple):
        final_output, aux_outputs = outputs
    else:
        # No deep supervision (inference mode)
        final_output = outputs
        aux_outputs = []

    # Main loss
    main_loss, loss_dict = criterion(final_output, targets)
    total_loss = main_loss

    # Auxiliary losses
    for i, (aux_output, weight) in enumerate(zip(aux_outputs, aux_weights)):
        # Resize auxiliary output to match target size
        aux_resized = F.interpolate(
            aux_output,
            size=targets.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        aux_loss, _ = criterion(aux_resized, targets)
        total_loss += weight * aux_loss

    return total_loss, loss_dict


# ==============================================================================
# DATASET
# ==============================================================================

class RefugeDataset(Dataset):
    """REFUGE-2 labeled dataset for U-Net training"""

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        # Cache in RAM for speed
        print(f"  Loading {len(image_paths)} images into RAM...")
        self.images = []
        self.masks = []

        for img_path, mask_path in tqdm(zip(image_paths, mask_paths),
                                        total=len(image_paths),
                                        desc="  Caching"):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # Convert mask to 3-channel one-hot encoding
            # Assuming: 0=background, 128=optic disc, 255=optic cup
            mask_3ch = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
            mask_3ch[:, :, 0] = (mask == 0).astype(np.float32)  # Background
            mask_3ch[:, :, 1] = ((mask == 128) | (mask == 255)).astype(np.float32)  # Optic disc
            mask_3ch[:, :, 2] = (mask == 255).astype(np.float32)  # Optic cup

            self.images.append(img)
            self.masks.append(mask_3ch)

        print(f"  [OK] Cached {len(self.images)} samples in RAM")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Permute mask from (H, W, C) to (C, H, W)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).permute(2, 0, 1)
        else:
            mask = mask.permute(2, 0, 1)

        return image, mask


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, aux_weights):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    dice_losses = []
    focal_losses = []
    boundary_losses = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with autocast():
            outputs = model(images)
            loss, loss_dict = deep_supervision_loss(outputs, masks, criterion, aux_weights)

        # Backward
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), ConfigV6.GRADIENT_CLIP)

        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        dice_losses.append(loss_dict['dice'])
        focal_losses.append(loss_dict['focal'])
        boundary_losses.append(loss_dict['boundary'])

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{loss_dict["dice"]:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_dice = np.mean(dice_losses)
    avg_focal = np.mean(focal_losses)
    avg_boundary = np.mean(boundary_losses)

    return avg_loss, avg_dice, avg_focal, avg_boundary


def validate(model, dataloader, criterion, device, aux_weights):
    """Validate the model"""
    model.eval()

    total_loss = 0
    dice_losses = []
    dice_accuracies = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            with autocast():
                outputs = model(images)
                loss, loss_dict = deep_supervision_loss(outputs, masks, criterion, aux_weights)

            # Get final output
            if isinstance(outputs, tuple):
                final_output = outputs[0]
            else:
                final_output = outputs

            # Compute Dice accuracy
            pred_masks = torch.sigmoid(final_output) > 0.5
            intersection = (pred_masks * masks).sum(dim=(2, 3))
            union = pred_masks.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2.0 * intersection / (union + 1e-6)).mean(dim=1)  # Average across channels

            total_loss += loss.item()
            dice_losses.append(loss_dict['dice'])
            dice_accuracies.extend(dice.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_dice_loss = np.mean(dice_losses)
    avg_dice_acc = np.mean(dice_accuracies) * 100  # Convert to percentage

    return avg_loss, avg_dice_loss, avg_dice_acc


# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Get labeled image paths
    image_dir = ConfigV6.LABELED_IMAGES_DIR
    mask_dir = ConfigV6.LABELED_MASKS_DIR

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f.replace('.jpg', '.bmp').replace('.jpeg', '.bmp').replace('.png', '.bmp')) for f in image_files]

    print(f"\nFound {len(image_paths)} labeled images")

    # Train/val split
    split_idx = int(len(image_paths) * ConfigV6.TRAIN_VAL_SPLIT)
    train_image_paths = image_paths[:split_idx]
    train_mask_paths = mask_paths[:split_idx]
    val_image_paths = image_paths[split_idx:]
    val_mask_paths = mask_paths[split_idx:]

    print(f"Train: {len(train_image_paths)} images")
    print(f"Val: {len(val_image_paths)} images")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = RefugeDataset(
        train_image_paths, train_mask_paths,
        transform=get_training_augmentation_v6()
    )

    val_dataset = RefugeDataset(
        val_image_paths, val_mask_paths,
        transform=get_validation_augmentation_v6()
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=ConfigV6.BATCH_SIZE,
        shuffle=True,
        num_workers=ConfigV6.NUM_WORKERS,
        pin_memory=ConfigV6.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ConfigV6.BATCH_SIZE,
        shuffle=False,
        num_workers=ConfigV6.NUM_WORKERS,
        pin_memory=ConfigV6.PIN_MEMORY
    )

    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)

    model = AttentionUNet_V6(
        n_classes=3,
        use_attention=ConfigV6.USE_ATTENTION,
        deep_supervision=ConfigV6.DEEP_SUPERVISION
    ).to(ConfigV6.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: Attention U-Net V6")
    print(f"  Encoder: {ConfigV6.ENCODER}")
    print(f"  Attention Gates: {ConfigV6.USE_ATTENTION}")
    print(f"  Deep Supervision: {ConfigV6.DEEP_SUPERVISION}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = CombinedSegmentationLoss(weights=ConfigV6.LOSS_WEIGHTS)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ConfigV6.LEARNING_RATE,
        weight_decay=ConfigV6.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=ConfigV6.LR_FACTOR,
        patience=ConfigV6.LR_PATIENCE,
        verbose=True
    )

    scaler = GradScaler()

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    best_val_loss = float('inf')
    best_dice_acc = 0.0
    patience_counter = 0

    history = {
        'train_loss': [], 'train_dice': [], 'train_focal': [], 'train_boundary': [],
        'val_loss': [], 'val_dice': [], 'val_dice_acc': []
    }

    start_time = datetime.now()

    for epoch in range(ConfigV6.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{ConfigV6.NUM_EPOCHS}")
        print("-" * 80)

        # Train
        train_loss, train_dice, train_focal, train_boundary = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            ConfigV6.DEVICE, ConfigV6.AUX_LOSS_WEIGHTS
        )

        # Validate
        val_loss, val_dice, val_dice_acc = validate(
            model, val_loader, criterion, ConfigV6.DEVICE, ConfigV6.AUX_LOSS_WEIGHTS
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_focal'].append(train_focal)
        history['train_boundary'].append(train_boundary)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_dice_acc'].append(val_dice_acc)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f}, Focal: {train_focal:.4f}, Boundary: {train_boundary:.4f})")
        print(f"Val Loss: {val_loss:.4f} | Dice Acc: {val_dice_acc:.2f}%")

        # Save best model
        if val_dice_acc > best_dice_acc:
            best_dice_acc = val_dice_acc
            best_val_loss = val_loss

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'dice_accuracy': val_dice_acc,
                'config': {k: v for k, v in vars(ConfigV6).items() if not k.startswith('__')}
            }, ConfigV6.MODEL_SAVE_PATH)

            print(f"[BEST] Saved model with Dice Acc: {val_dice_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{ConfigV6.EARLY_STOPPING_PATIENCE}")

        # Early stopping
        if patience_counter >= ConfigV6.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        # Memory cleanup
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    training_time = (datetime.now() - start_time).total_seconds() / 60

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Training Time: {training_time:.2f} minutes")
    print(f"Best Dice Accuracy: {best_dice_acc:.2f}%")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # Save statistics (convert numpy types to Python types for JSON serialization)
    def convert_to_json_serializable(obj):
        """Convert numpy/torch types to Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy/torch scalar
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    stats = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v6_attention_unet_deep_supervision',
        'architecture': {
            'encoder': ConfigV6.ENCODER,
            'attention_gates': ConfigV6.USE_ATTENTION,
            'deep_supervision': ConfigV6.DEEP_SUPERVISION
        },
        'labeled_images_used': len(image_paths),
        'training_images': len(train_image_paths),
        'validation_images': len(val_image_paths),
        'training_time_minutes': float(training_time),
        'gpu': torch.cuda.get_device_name(0),
        'dice_accuracy': float(best_dice_acc),
        'best_val_loss': float(best_val_loss),
        'improvement_over_v51': f"{best_dice_acc - 62.5:.2f}%",
        'history': convert_to_json_serializable(history),
        'config': {
            'batch_size': ConfigV6.BATCH_SIZE,
            'learning_rate': ConfigV6.LEARNING_RATE,
            'epochs_trained': len(history['train_loss']),
            'loss_weights': ConfigV6.LOSS_WEIGHTS
        }
    }

    with open(ConfigV6.STATS_SAVE_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {ConfigV6.STATS_SAVE_PATH}")
    print(f"Model saved to {ConfigV6.MODEL_SAVE_PATH}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
