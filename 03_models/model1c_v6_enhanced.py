"""
Model 1c-V6 Enhanced: Multi-Scale Feature Fusion + Clinical Features
================================================================================

Architecture:
    Fundus Image (512×512)
    → U-Net V6 (frozen, 95.98% Dice) with Attention Gates
    → Extract Multi-Scale Features (decoder4, decoder3, decoder2, decoder1)
    → Extract Final Segmentation Mask (512×512×3)
    → Extract Clinical Features (C/D ratio, ISNT rule, etc.)
    → Canny Edge Detection
    → ResNet-50 Classifier (9 channels: RGB + Mask + Edges)
    → Fuse: ResNet Features + U-Net Decoder Features + Clinical Features
    → Glaucoma Probability [0, 1]

Purpose:
- Maximize pipeline potential with high-quality V6 masks (95.98% Dice)
- Strategy 1: Multi-scale feature fusion from attention-weighted decoder layers
- Strategy 3: Clinical feature extraction (C/D ratio, ISNT, rim asymmetry)
- Baseline: Model 1c-V6 (84.94% acc, 0.9327 AUC)
- Target: 88-92% accuracy (closing gap to Model 4: 94.80%)

Key Features:
- U-Net V6 (frozen, 95.98% Dice) with Attention Gates + Deep Supervision
- Multi-scale decoder features (attention-weighted hierarchical representations)
- Clinical features from near-perfect masks (C/D ratio, morphology)
- Canny edge detection on high-quality segmentation masks
- ResNet-50 classifier with feature fusion
- Expected: 88-92% accuracy (enhanced pipeline)

Thesis Contributions:
1. Demonstrates Attention U-Net provides value beyond final masks
2. Shows 95.98% Dice enables accurate clinical measurements
3. Proves pipeline approach competitive with direct classification

Created: 2025-11-15
Baseline: model1c_unet_canny_resnet50.py (84.94% acc, 0.9327 AUC)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler

import os
import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import time
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated operations
import kornia


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Model 1c-V6 Enhanced Configuration"""

    # Paths (relative to project root)
    DATA_ROOT = '../processed_gpu'
    TRAIN_RG = os.path.join(DATA_ROOT, 'train', 'RG')
    TRAIN_NRG = os.path.join(DATA_ROOT, 'train', 'NRG')
    VAL_RG = os.path.join(DATA_ROOT, 'validation', 'RG')
    VAL_NRG = os.path.join(DATA_ROOT, 'validation', 'NRG')
    TEST_RG = os.path.join(DATA_ROOT, 'test', 'RG')
    TEST_NRG = os.path.join(DATA_ROOT, 'test', 'NRG')

    # U-Net V6 weights (95.98% Dice)
    UNET_WEIGHTS = '../02_scripts/best_segmentation_model_gpu_v6.pth'

    # Output
    MODEL_SAVE_PATH = 'model1c_v6_enhanced_best.pth'
    RESULTS_PATH = 'model1c_v6_enhanced_results.json'

    # Training hyperparameters (GPU-optimized for speed)
    BATCH_SIZE = 24  # Maximized for RTX 3070
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 15

    # Dropout rates
    DROPOUT_RATES = [0.6, 0.35, 0.25]

    # Focal Loss parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Architecture
    UNET_INPUT_SIZE = 512
    RESNET50_INPUT_SIZE = 256
    NUM_CLASSES = 1

    # Canny edge detection
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    # Clinical features (GPU-optimized)
    NUM_CLINICAL_FEATURES = 10  # GPU-only features (removed slow CPU operations)

    # Multi-scale fusion
    DECODER_CHANNELS = [1024, 512, 256, 64]  # V6 decoder channels
    FUSION_OUTPUT_DIM = 512

    # Device (GPU-optimized)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 6  # Increased for better data loading throughput
    PIN_MEMORY = True
    PREFETCH_FACTOR = 3  # Prefetch batches to reduce GPU idle time
    PERSISTENT_WORKERS = True  # Keep worker processes alive

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Clinical Feature Extraction
# ============================================================================

class ClinicalFeatureExtractor(nn.Module):
    """
    GPU-optimized clinical feature extraction from segmentation masks

    Simplified to 10 GPU-only features (removed slow CPU operations):
    1. Cup-to-Disc (C/D) ratio (area-based)
    2. Vertical C/D ratio
    3. Horizontal C/D ratio
    4. ISNT rule violation score
    5. Rim asymmetry (quadrant-based)
    6. Cup eccentricity (cup center vs disc center distance)
    7. Disc area (normalized)
    8. Cup area (normalized)
    9. Rim area (normalized)
    10. Cup/Disc aspect ratio difference
    """

    def __init__(self):
        super(ClinicalFeatureExtractor, self).__init__()

    def forward(self, mask):
        """
        GPU-optimized clinical feature extraction (all operations on GPU)

        Args:
            mask: (B, 3, H, W) tensor with [background, disc, cup] probabilities

        Returns:
            features: (B, 10) tensor with clinical features
        """
        B, C, H, W = mask.shape

        # Separate disc and cup channels
        disc_mask = mask[:, 1, :, :]  # (B, H, W)
        cup_mask = mask[:, 2, :, :]   # (B, H, W)

        # Threshold to binary masks
        disc = (disc_mask > 0.5).float()
        cup = (cup_mask > 0.5).float()

        # === AREA-BASED FEATURES (GPU) ===
        disc_area = disc.sum(dim=[1, 2])  # (B,)
        cup_area = cup.sum(dim=[1, 2])    # (B,)
        rim_area = disc_area - cup_area

        # 1. C/D Ratio (area)
        cd_ratio = cup_area / (disc_area + 1e-8)

        # 2. Vertical C/D Ratio (GPU)
        disc_vertical = (disc.sum(dim=2) > 0).float().sum(dim=1)  # (B,)
        cup_vertical = (cup.sum(dim=2) > 0).float().sum(dim=1)
        vcd_ratio = cup_vertical / (disc_vertical + 1e-8)

        # 3. Horizontal C/D Ratio (GPU)
        disc_horizontal = (disc.sum(dim=1) > 0).float().sum(dim=1)  # (B,)
        cup_horizontal = (cup.sum(dim=1) > 0).float().sum(dim=1)
        hcd_ratio = cup_horizontal / (disc_horizontal + 1e-8)

        # 4. ISNT Rule Violation (GPU - quadrant-based)
        # Divide into quadrants: I(bottom), S(top), N(left), T(right)
        mid_h, mid_w = H // 2, W // 2
        inferior = cup[:, mid_h:, :].sum(dim=[1, 2])
        superior = cup[:, :mid_h, :].sum(dim=[1, 2])
        nasal = cup[:, :, :mid_w].sum(dim=[1, 2])
        temporal = cup[:, :, mid_w:].sum(dim=[1, 2])

        # ISNT rule violations (I≥S≥N≥T)
        violations = ((inferior < superior).float() +
                     (superior < nasal).float() +
                     (nasal < temporal).float()) / 3.0  # (B,)

        # 5. Rim Asymmetry (GPU - quadrant coefficient of variation)
        rim = disc - cup
        q1 = rim[:, :mid_h, :mid_w].sum(dim=[1, 2])
        q2 = rim[:, :mid_h, mid_w:].sum(dim=[1, 2])
        q3 = rim[:, mid_h:, :mid_w].sum(dim=[1, 2])
        q4 = rim[:, mid_h:, mid_w:].sum(dim=[1, 2])

        quadrants = torch.stack([q1, q2, q3, q4], dim=1)  # (B, 4)
        rim_mean = quadrants.mean(dim=1, keepdim=True)
        rim_std = quadrants.std(dim=1)
        rim_asym = rim_std / (rim_mean.squeeze() + 1e-8)  # (B,)

        # 6. Cup Eccentricity (GPU - centroid distance)
        # Disc centroid
        y_coords = torch.arange(H, device=mask.device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=mask.device).view(1, 1, W).expand(B, H, W)

        disc_y = (disc * y_coords).sum(dim=[1, 2]) / (disc_area + 1e-8)
        disc_x = (disc * x_coords).sum(dim=[1, 2]) / (disc_area + 1e-8)

        # Cup centroid
        cup_y = (cup * y_coords).sum(dim=[1, 2]) / (cup_area + 1e-8)
        cup_x = (cup * x_coords).sum(dim=[1, 2]) / (cup_area + 1e-8)

        # Euclidean distance (normalized by image size)
        eccentricity = torch.sqrt((disc_y - cup_y)**2 + (disc_x - cup_x)**2) / H

        # 7-9. Normalized areas (GPU)
        total_pixels = H * W
        disc_area_norm = disc_area / total_pixels
        cup_area_norm = cup_area / total_pixels
        rim_area_norm = rim_area / total_pixels

        # 10. Aspect ratio difference (GPU)
        aspect_ratio_diff = torch.abs(vcd_ratio - hcd_ratio)

        # Stack all features
        features = torch.stack([
            cd_ratio,           # 1
            vcd_ratio,          # 2
            hcd_ratio,          # 3
            violations,         # 4
            rim_asym,           # 5
            eccentricity,       # 6
            disc_area_norm,     # 7
            cup_area_norm,      # 8
            rim_area_norm,      # 9
            aspect_ratio_diff   # 10
        ], dim=1)  # (B, 10)

        return features


# ============================================================================
# Multi-Scale Feature Fusion
# ============================================================================

class FeaturePyramidFusion(nn.Module):
    """
    Fuse multi-scale features from U-Net V6 decoder layers

    Input: List of decoder features at different scales
        - decoder4: (B, 1024, 32, 32)
        - decoder3: (B, 512, 64, 64)
        - decoder2: (B, 256, 128, 128)
        - decoder1: (B, 64, 256, 256)

    Output: Fused feature vector (B, 512)
    """

    def __init__(self, in_channels=[1024, 512, 256, 64], out_dim=512):
        super(FeaturePyramidFusion, self).__init__()

        # Adaptive pooling to fixed size
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in in_channels
        ])

        # Project each scale to same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, out_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ) for ch in in_channels
        ])

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, decoder_features):
        """
        Args:
            decoder_features: List of 4 tensors [dec4, dec3, dec2, dec1]

        Returns:
            fused: (B, out_dim) fused feature vector
        """
        pooled = []
        for i, feat in enumerate(decoder_features):
            # Global average pooling: (B, C, H, W) → (B, C, 1, 1)
            p = self.pools[i](feat)
            p = p.flatten(1)  # (B, C)
            # Project to same dimension
            p = self.projections[i](p)  # (B, out_dim // 4)
            pooled.append(p)

        # Concatenate all scales
        concat = torch.cat(pooled, dim=1)  # (B, out_dim)

        # Final fusion
        fused = self.fusion(concat)

        return fused


# ============================================================================
# Canny Edge Detection Module
# ============================================================================

class CannyEdgeDetector(nn.Module):
    """GPU-accelerated Canny edge detection using Kornia"""

    def __init__(self, threshold1=50, threshold2=150):
        super(CannyEdgeDetector, self).__init__()
        # Normalize thresholds to [0, 1] for Kornia
        self.low_threshold = threshold1 / 255.0
        self.high_threshold = threshold2 / 255.0

    def forward(self, mask):
        """
        Apply GPU-accelerated Canny edge detection to segmentation mask

        Args:
            mask: (B, 3, H, W) tensor in [0, 1] range on GPU

        Returns:
            edges: (B, 3, H, W) tensor with detected edges on GPU
        """
        B, C, H, W = mask.shape
        edges = []

        # Process each channel separately (Kornia Canny expects single channel)
        for c in range(C):
            channel = mask[:, c:c+1, :, :]  # (B, 1, H, W)

            # Apply Kornia Canny (fully on GPU)
            # Returns tuple: (gradients, edges)
            _, edge_map = kornia.filters.canny(
                channel,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                kernel_size=(5, 5),
                sigma=(1.0, 1.0),
                hysteresis=True,
                eps=1e-6
            )

            edges.append(edge_map)  # (B, 1, H, W)

        # Concatenate all channels
        edges = torch.cat(edges, dim=1)  # (B, 3, H, W)

        return edges


# ============================================================================
# ResNet-50 Classifier (9-channel input: RGB + Mask + Edges)
# ============================================================================

class ResNet50Classifier_9Channel(nn.Module):
    """
    ResNet-50 classifier modified for 9-channel input

    Input: 9 channels (3 RGB + 3 mask + 3 Canny edges)
    Output: Feature vector (2048-dim) before final classification
    """

    def __init__(self):
        super(ResNet50Classifier_9Channel, self).__init__()

        # Load pretrained ResNet-50
        resnet50 = models.resnet50(pretrained=True)

        # Modify first conv layer for 9-channel input
        self.conv1 = nn.Conv2d(
            9, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize with pretrained weights
        with torch.no_grad():
            pretrained_weight = resnet50.conv1.weight.clone()
            # RGB channels
            self.conv1.weight[:, :3, :, :] = pretrained_weight
            # Mask channels
            self.conv1.weight[:, 3:6, :, :] = pretrained_weight
            # Edge channels
            self.conv1.weight[:, 6:9, :, :] = pretrained_weight

        # ResNet-50 backbone
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: (B, 9, H, W) tensor

        Returns:
            features: (B, 2048) feature vector
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = x.flatten(1)  # (B, 2048)

        return features


# ============================================================================
# Enhanced Model 1c-V6
# ============================================================================

class Model1c_V6_Enhanced(nn.Module):
    """
    Enhanced Model 1c-V6: Multi-Scale Features + Clinical Features

    Pipeline:
        RGB Image (512×512×3)
        ↓
        [Frozen U-Net V6 with Attention Gates]
        ↓
        Outputs:
          1. Final Mask (512×512×3) - 95.98% Dice
          2. Decoder Features (multi-scale, attention-weighted)
        ↓
        [Feature Extraction]
          1. Clinical Features (15-dim): C/D ratio, ISNT, morphology
          2. Multi-Scale Features (512-dim): Fused decoder features
          3. Canny Edges (512×512×3)
        ↓
        [ResNet-50 on RGB + Mask + Edges]
        ↓
        ResNet Features (2048-dim)
        ↓
        [Feature Fusion]
        Concatenate: ResNet (2048) + Multi-Scale (512) + Clinical (15)
        ↓
        [Classification Head]
        → Glaucoma Probability
    """

    def __init__(self, unet_weights_path=None, freeze_unet=True):
        super(Model1c_V6_Enhanced, self).__init__()

        # Load U-Net V6
        from pathlib import Path
        if unet_weights_path and Path(unet_weights_path).exists():
            print(f"\n[Enhanced Model] Loading U-Net V6 from {unet_weights_path}...")
            checkpoint = torch.load(unet_weights_path, map_location='cpu', weights_only=False)
            dice_acc = checkpoint.get('dice_accuracy', 'N/A')
            print(f"[OK] U-Net V6 Dice Accuracy: {dice_acc}%")

            # Import V6 architecture
            import sys
            sys.path.insert(0, str(Path(unet_weights_path).parent))
            from gpu_optimized_mask_generation_v6 import AttentionUNet_V6

            self.unet = AttentionUNet_V6(n_classes=3, use_attention=True, deep_supervision=True)
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Loaded V6 Attention U-Net with decoder features")
        else:
            raise FileNotFoundError(f"U-Net V6 weights not found at {unet_weights_path}")

        # Freeze U-Net
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()
            print("[OK] U-Net V6 frozen (74.7M parameters)")

        # Clinical feature extractor
        self.clinical_extractor = ClinicalFeatureExtractor()

        # Multi-scale feature fusion
        self.feature_fusion = FeaturePyramidFusion(
            in_channels=Config.DECODER_CHANNELS,
            out_dim=Config.FUSION_OUTPUT_DIM
        )

        # Canny edge detector
        self.canny = CannyEdgeDetector(
            threshold1=Config.CANNY_THRESHOLD1,
            threshold2=Config.CANNY_THRESHOLD2
        )

        # ResNet-50 classifier (9-channel: RGB + Mask + Edges)
        self.resnet_backbone = ResNet50Classifier_9Channel()

        # Final classification head
        # Input: ResNet (2048) + Multi-Scale (512) + Clinical (15) = 2575 dim
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATES[0]),
            nn.Linear(2048 + Config.FUSION_OUTPUT_DIM + Config.NUM_CLINICAL_FEATURES, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATES[1]),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATES[2]),
            nn.Linear(512, 1)
        )

    def forward(self, x, return_intermediates=False):
        """
        Forward pass through enhanced pipeline

        Args:
            x: RGB fundus image (B, 3, 512, 512)
            return_intermediates: If True, return intermediate outputs

        Returns:
            logits: (B, 1) classification logits
            (optional) intermediates: dict with mask, features, etc.
        """
        B = x.size(0)

        # === STAGE 1: U-Net V6 (Frozen) ===
        with torch.no_grad():
            # Forward through encoder
            enc_features = self.unet.encoder(x)
            # enc_features: [enc1, enc2, enc3, enc4, enc5]
            # Channels: [64, 256, 512, 1024, 2048]

            # Forward through decoder (with attention gates)
            dec4 = self.unet.decoder4(enc_features[-1], enc_features[-2])  # (B, 1024, 32, 32)
            dec3 = self.unet.decoder3(dec4, enc_features[-3])              # (B, 512, 64, 64)
            dec2 = self.unet.decoder2(dec3, enc_features[-4])              # (B, 256, 128, 128)
            dec1 = self.unet.decoder1(dec2, enc_features[-5])              # (B, 64, 256, 256)

            # Final segmentation mask
            d0 = self.unet.final_upsample(dec1)  # (B, 64, 512, 512)
            mask_logits = self.unet.final_conv(d0)  # (B, 3, 512, 512)
            mask = torch.softmax(mask_logits, dim=1)  # (B, 3, 512, 512) in [0, 1]

        # === STAGE 2: Feature Extraction ===
        # 1. Multi-scale decoder features (attention-weighted)
        decoder_features = [dec4, dec3, dec2, dec1]
        multiscale_features = self.feature_fusion(decoder_features)  # (B, 512)

        # 2. Clinical features from high-quality masks
        clinical_features = self.clinical_extractor(mask)  # (B, 15)

        # 3. Canny edges
        edges = self.canny(mask)  # (B, 3, 512, 512)

        # === STAGE 3: ResNet-50 on RGB + Mask + Edges ===
        # Concatenate RGB + Mask + Edges
        combined = torch.cat([x, mask, edges], dim=1)  # (B, 9, 512, 512)

        # Resize to ResNet input size (256×256)
        combined_resized = F.interpolate(
            combined,
            size=(Config.RESNET50_INPUT_SIZE, Config.RESNET50_INPUT_SIZE),
            mode='bilinear',
            align_corners=False
        )

        # Extract ResNet features
        resnet_features = self.resnet_backbone(combined_resized)  # (B, 2048)

        # === STAGE 4: Feature Fusion ===
        # Concatenate all feature sources
        all_features = torch.cat([
            resnet_features,      # (B, 2048) - Direct classification features
            multiscale_features,  # (B, 512)  - Attention-weighted decoder features
            clinical_features     # (B, 15)   - Clinical measurements
        ], dim=1)  # (B, 2575)

        # === STAGE 5: Final Classification ===
        logits = self.classifier(all_features)  # (B, 1)

        if return_intermediates:
            return logits, {
                'mask': mask,
                'edges': edges,
                'clinical_features': clinical_features,
                'multiscale_features': multiscale_features,
                'resnet_features': resnet_features
            }

        return logits

    def count_parameters(self):
        """Count trainable and total parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }


# ============================================================================
# Dataset
# ============================================================================

class GlaucomaDataset(Dataset):
    """Dataset for fundus images"""

    def __init__(self, rg_dir, nrg_dir, transform=None):
        self.transform = transform

        # Load RG (glaucoma) images
        self.rg_images = [
            os.path.join(rg_dir, f) for f in os.listdir(rg_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Load NRG (normal) images
        self.nrg_images = [
            os.path.join(nrg_dir, f) for f in os.listdir(nrg_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Combine and create labels
        self.images = self.rg_images + self.nrg_images
        self.labels = [1] * len(self.rg_images) + [0] * len(self.nrg_images)

        print(f"Dataset: {len(self.images)} images ({len(self.rg_images)} RG, {len(self.nrg_images)} NRG)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


# ============================================================================
# Loss Function
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, 1) logits
            targets: (B,) labels in {0, 1}
        """
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs.squeeze(), targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    # Keep U-Net frozen
    model.unet.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_labels = (all_preds >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, pred_labels)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0

    return avg_loss, accuracy, auc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_labels = (all_preds >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, pred_labels)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0

    return avg_loss, accuracy, auc


def test_model(model, dataloader, device):
    """Test the model and return detailed metrics"""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(device)

            with autocast():
                logits = model(images)

            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_labels = (all_preds >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, pred_labels)
    sensitivity = recall_score(all_labels, pred_labels, zero_division=0)
    precision = precision_score(all_labels, pred_labels, zero_division=0)
    f1 = f1_score(all_labels, pred_labels, zero_division=0)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, pred_labels)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("=" * 80)
    print("Model 1c-V6 Enhanced: Multi-Scale Features + Clinical Features")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Data Source: {Config.DATA_ROOT}/")
    print(f"U-Net V6 Weights: {Config.UNET_WEIGHTS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print("=" * 80)
    print("\nEnhancements:")
    print("  [Strategy 1] Multi-scale decoder feature fusion (512-dim)")
    print("  [Strategy 3] Clinical features from 95.98% Dice masks (15-dim)")
    print("  [Baseline] ResNet-50 on RGB + Mask + Edges (2048-dim)")
    print("  [Total] Fused features: 2048 + 512 + 15 = 2575-dim")
    print("=" * 80)
    print()

    # Data transforms (simplified for faster training)
    train_transform = transforms.Compose([
        transforms.Resize((Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

    # Load datasets
    print("Loading data...")
    train_dataset = GlaucomaDataset(Config.TRAIN_RG, Config.TRAIN_NRG, train_transform)
    val_dataset = GlaucomaDataset(Config.VAL_RG, Config.VAL_NRG, test_transform)
    test_dataset = GlaucomaDataset(Config.TEST_RG, Config.TEST_NRG, test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    # Create enhanced model
    print("\nCreating Enhanced Model 1c-V6...")
    model = Model1c_V6_Enhanced(
        unet_weights_path=Config.UNET_WEIGHTS,
        freeze_unet=True
    ).to(Config.DEVICE)

    # Print parameter counts
    param_counts = model.count_parameters()
    print("\nModel Parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen (U-Net V6): {param_counts['frozen']:,}")

    # Loss and optimizer
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Enhanced Model 1c-V6 Training")
    print("=" * 80)

    best_val_auc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 80)

        # Train
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, Config.DEVICE
        )

        # Validate
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        # Update scheduler
        scheduler.step(val_auc)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc
            }, Config.MODEL_SAVE_PATH)
            print(f"[BEST] Saved model with Val AUC: {val_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.2f} minutes")

    # Load best model and test
    print("\n" + "=" * 80)
    print("Testing Best Enhanced Model")
    print("=" * 80)

    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = test_model(model, test_loader, Config.DEVICE)

    print("\nTest Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {test_metrics['confusion_matrix']}")

    # Save results
    results = {
        'model': 'Model 1c-V6 Enhanced: Multi-Scale Features + Clinical Features',
        'architecture': {
            'unet': 'U-Net V6 with Attention Gates (frozen, 95.98% Dice)',
            'enhancements': [
                'Multi-scale decoder feature fusion (512-dim)',
                'Clinical features (C/D ratio, ISNT, morphology) (15-dim)',
                'ResNet-50 on RGB + Mask + Edges (2048-dim)',
                'Total fused features: 2575-dim'
            ],
            'strategy_1': 'Multi-scale feature fusion from attention-weighted decoders',
            'strategy_3': 'Clinical feature extraction from 95.98% Dice masks',
            'baseline': 'Model 1c-V6: 84.94% acc, 0.9327 AUC'
        },
        'data_source': Config.DATA_ROOT,
        'training': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'epochs_trained': len(history['train_loss']),
            'best_val_auc': best_val_auc,
            'training_time_minutes': training_time / 60
        },
        'test_metrics': test_metrics,
        'history': history,
        'model_parameters': param_counts,
        'comparison': {
            'baseline_v6': {
                'test_acc': 0.8494,
                'test_auc': 0.9327
            },
            'enhanced_v6': {
                'test_acc': test_metrics['accuracy'],
                'test_auc': test_metrics['auc']
            },
            'improvement': {
                'acc_delta': test_metrics['accuracy'] - 0.8494,
                'auc_delta': test_metrics['auc'] - 0.9327
            }
        }
    }

    with open(Config.RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {Config.RESULTS_PATH}")
    print(f"Best model saved to {Config.MODEL_SAVE_PATH}")

    print("\n" + "=" * 80)
    print("Comparison with Baseline:")
    print("=" * 80)
    print(f"Baseline (Model 1c-V6):  {0.8494:.4f} acc, {0.9327:.4f} AUC")
    print(f"Enhanced (This model):   {test_metrics['accuracy']:.4f} acc, {test_metrics['auc']:.4f} AUC")
    print(f"Improvement:             {test_metrics['accuracy'] - 0.8494:+.4f} acc, {test_metrics['auc'] - 0.9327:+.4f} AUC")
    print("=" * 80)


if __name__ == '__main__':
    main()
