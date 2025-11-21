"""
Model 4: U-Net V6 + CED + DenseNet121 Enhanced
================================================================================

Architecture:
    Fundus Image (512×512)
    → U-Net V6 (frozen, 95.98% Dice) with Attention Gates
    → Extract Multi-Scale Features (decoder4, decoder3, decoder2, decoder1)
    → Extract Final Segmentation Mask (512×512×3)
    → Extract Clinical Features (C/D ratio, ISNT rule, etc.)
    → Canny Edge Detection (GPU-accelerated)
    → DenseNet121 Classifier (9 channels: RGB + Mask + Edges)
    → Fuse: DenseNet Features + U-Net Decoder Features + Clinical Features
    → Glaucoma Probability [0, 1]

Purpose:
- Combine DenseNet121 backbone with full enhancements
- Test if DenseNet121 + enhancements > ResNet50 + enhancements
- Expected: Best overall performance (combining Model 3's backbone + Model 1c's features)

Key Features:
- U-Net V6 (frozen, 95.98% Dice) with Attention Gates + Deep Supervision
- Multi-scale decoder features (attention-weighted hierarchical representations)
- Clinical features from near-perfect masks (5 key features - optimized for speed)
- Canny edge detection on high-quality segmentation masks (Kornia GPU)
- DenseNet121 classifier with 9-channel input
- Feature fusion: DenseNet (1024) + Multi-scale (512) + Clinical (5) = 1541-dim
- Speed optimizations: Batch size 32, operations on 224x224 masks

Expected Performance:
- Test AUC: 0.985-0.992 (best of all models)
- Test Acc: 94-96%

Created: 2025-11-16
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
    """Model 4 Enhanced Configuration"""

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
    MODEL_SAVE_PATH = 'model4_best.pth'
    RESULTS_PATH = 'model4_results.json'

    # Training hyperparameters (optimized for speed)
    BATCH_SIZE = 32  # Increased from 24
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
    DENSENET_INPUT_SIZE = 224
    NUM_CLASSES = 1

    # Canny edge detection (GPU-accelerated)
    CANNY_LOW_THRESHOLD = 0.1
    CANNY_HIGH_THRESHOLD = 0.2

    # Clinical features (simplified for speed)
    NUM_CLINICAL_FEATURES = 5  # Reduced from 10

    # Multi-scale fusion
    DECODER_CHANNELS = [1024, 512, 256, 64]  # V6 decoder channels
    FUSION_OUTPUT_DIM = 512

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 6
    PIN_MEMORY = True
    PREFETCH_FACTOR = 3
    PERSISTENT_WORKERS = True

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Clinical Feature Extraction (GPU-Optimized)
# ============================================================================

class ClinicalFeatureExtractor(nn.Module):
    """
    GPU-optimized clinical feature extraction (simplified for speed)

    5 key features:
    1. Cup-to-Disc (C/D) ratio (area-based) - most important
    2. Vertical C/D ratio
    3. Horizontal C/D ratio
    4. Disc area (normalized)
    5. Cup area (normalized)
    """

    def __init__(self):
        super(ClinicalFeatureExtractor, self).__init__()

    def forward(self, mask):
        """
        Args:
            mask: (B, 3, H, W) tensor with [background, disc, cup] probabilities

        Returns:
            features: (B, 5) tensor
        """
        B, C, H, W = mask.shape

        # Separate disc and cup channels
        disc_mask = mask[:, 1, :, :]
        cup_mask = mask[:, 2, :, :]

        # Threshold to binary masks
        disc = (disc_mask > 0.5).float()
        cup = (cup_mask > 0.5).float()

        # Area-based features
        disc_area = disc.sum(dim=[1, 2])
        cup_area = cup.sum(dim=[1, 2])

        # 1. C/D Ratio (area) - most important clinical feature
        cd_ratio = cup_area / (disc_area + 1e-8)

        # 2. Vertical C/D Ratio
        disc_vertical = (disc.sum(dim=2) > 0).float().sum(dim=1)
        cup_vertical = (cup.sum(dim=2) > 0).float().sum(dim=1)
        vcd_ratio = cup_vertical / (disc_vertical + 1e-8)

        # 3. Horizontal C/D Ratio
        disc_horizontal = (disc.sum(dim=1) > 0).float().sum(dim=1)
        cup_horizontal = (cup.sum(dim=1) > 0).float().sum(dim=1)
        hcd_ratio = cup_horizontal / (disc_horizontal + 1e-8)

        # 4-5. Normalized areas
        total_area = H * W
        disc_area_norm = disc_area / total_area
        cup_area_norm = cup_area / total_area

        # Stack all features
        features = torch.stack([
            cd_ratio, vcd_ratio, hcd_ratio, disc_area_norm, cup_area_norm
        ], dim=1)

        return features


# ============================================================================
# Canny Edge Detection (GPU-Accelerated with Kornia)
# ============================================================================

class CannyEdgeDetector(nn.Module):
    """GPU-accelerated Canny edge detection using Kornia"""

    def __init__(self, low_threshold=0.1, high_threshold=0.2):
        super(CannyEdgeDetector, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, mask):
        """
        Args:
            mask: (B, 3, H, W) segmentation mask

        Returns:
            edges: (B, 3, H, W) edge maps
        """
        B, C, H, W = mask.shape
        edges = []

        for c in range(C):
            channel = mask[:, c:c+1, :, :]
            _, edge_map = kornia.filters.canny(
                channel,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                kernel_size=(5, 5),
                sigma=(1.0, 1.0),
                hysteresis=True
            )
            edges.append(edge_map)

        return torch.cat(edges, dim=1)


# ============================================================================
# U-Net V6 Architecture (ResNet50 Encoder + Attention Decoder)
# ============================================================================

class AttentionGate(nn.Module):
    """Attention gate for U-Net V6"""
    def __init__(self, F_g, F_l, F_int):
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
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResNet50Encoder(nn.Module):
    """ResNet50 pretrained encoder"""
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


class DecoderBlock(nn.Module):
    """Decoder block with attention gate and double convolution"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()

        self.use_attention = use_attention

        if use_attention:
            self.attention = AttentionGate(
                F_g=out_channels,
                F_l=skip_channels,
                F_int=out_channels
            )

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

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
        x = self.upsample(x)
        if self.use_attention:
            skip = self.attention(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet_V6(nn.Module):
    """U-Net V6 with ResNet50 Encoder, Attention Gates and Deep Supervision"""
    def __init__(self, n_classes=3, use_attention=True, deep_supervision=True):
        super(AttentionUNet_V6, self).__init__()

        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        self.encoder = ResNet50Encoder()

        self.decoder4 = DecoderBlock(2048, 1024, 1024, use_attention)
        self.decoder3 = DecoderBlock(1024, 512, 512, use_attention)
        self.decoder2 = DecoderBlock(512, 256, 256, use_attention)
        self.decoder1 = DecoderBlock(256, 64, 64, use_attention)

        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

        if deep_supervision:
            self.aux_head3 = nn.Conv2d(1024, n_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(512, n_classes, kernel_size=1)
            self.aux_head1 = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        d0 = self.final_upsample(d1)
        final = self.final_conv(d0)

        if self.deep_supervision and self.training:
            aux3 = self.aux_head3(d4)
            aux2 = self.aux_head2(d3)
            aux1 = self.aux_head1(d2)
            return final, [aux3, aux2, aux1], [d4, d3, d2, d1]
        else:
            return final, [d4, d3, d2, d1]


# ============================================================================
# DenseNet121 9-Channel Classifier
# ============================================================================

class DenseNet121_9Channel(nn.Module):
    """DenseNet121 modified for 9-channel input (RGB + Mask + Edges)"""
    def __init__(self, pretrained=True):
        super(DenseNet121_9Channel, self).__init__()

        densenet = models.densenet121(pretrained=pretrained)

        # Modify first conv layer: 3 → 9 channels
        original_conv = densenet.features.conv0
        self.conv0 = nn.Conv2d(
            9, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new conv layer
        with torch.no_grad():
            # Copy RGB weights
            self.conv0.weight[:, :3, :, :] = original_conv.weight
            # Initialize mask and edge channels
            self.conv0.weight[:, 3:6, :, :] = original_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * 0.1
            self.conv0.weight[:, 6:9, :, :] = original_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * 0.05

        # Copy rest of features
        self.features = nn.Sequential(
            self.conv0,
            *list(densenet.features.children())[1:]
        )

        self.num_features = densenet.classifier.in_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


# ============================================================================
# Multi-Scale Feature Fusion
# ============================================================================

class MultiScaleFeatureFusion(nn.Module):
    """Fuses multi-scale features from U-Net decoder"""

    def __init__(self, decoder_channels, output_dim):
        super(MultiScaleFeatureFusion, self).__init__()

        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in decoder_channels
        ])

        total_channels = sum(decoder_channels)
        self.fusion = nn.Sequential(
            nn.Linear(total_channels, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, decoder_features):
        """
        Args:
            decoder_features: List of [d4, d3, d2, d1] tensors

        Returns:
            fused: (B, output_dim) tensor
        """
        pooled = []
        for feat, pool in zip(decoder_features, self.pools):
            pooled.append(pool(feat).flatten(1))

        concatenated = torch.cat(pooled, dim=1)
        fused = self.fusion(concatenated)
        return fused


# ============================================================================
# Model 4: U-Net V6 + CED + DenseNet121 Enhanced
# ============================================================================

class Model4_Enhanced(nn.Module):
    """
    Model 4: U-Net V6 + Canny + DenseNet121 + Multi-Scale + Clinical

    Complete enhanced pipeline with DenseNet121 backbone
    """
    def __init__(self, unet_weights_path, dropout_rates=[0.6, 0.35, 0.25]):
        super(Model4_Enhanced, self).__init__()

        # 1. U-Net V6 (frozen, with decoder access)
        self.unet = AttentionUNet_V6(n_classes=3, use_attention=True, deep_supervision=True)
        checkpoint = torch.load(unet_weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.unet.load_state_dict(checkpoint)
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        # 2. Clinical feature extractor
        self.clinical_extractor = ClinicalFeatureExtractor()

        # 3. Canny edge detector (GPU)
        self.canny = CannyEdgeDetector(
            low_threshold=Config.CANNY_LOW_THRESHOLD,
            high_threshold=Config.CANNY_HIGH_THRESHOLD
        )

        # 4. DenseNet121 feature extractor (9 channels)
        self.densenet = DenseNet121_9Channel(pretrained=True)
        densenet_features = self.densenet.num_features  # 1024

        # 5. Multi-scale feature fusion
        self.multiscale_fusion = MultiScaleFeatureFusion(
            decoder_channels=Config.DECODER_CHANNELS,
            output_dim=Config.FUSION_OUTPUT_DIM
        )

        # 6. Final fusion and classification
        total_features = densenet_features + Config.FUSION_OUTPUT_DIM + Config.NUM_CLINICAL_FEATURES
        # total_features = 1024 + 512 + 5 = 1541

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[0]),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[2]),

            nn.Linear(64, 1)
        )

    def forward(self, image):
        """
        Args:
            image: (B, 3, 512, 512) RGB fundus images

        Returns:
            logits: (B, 1) glaucoma classification logits
        """
        B = image.size(0)

        # 1. Generate masks and extract decoder features with U-Net V6
        with torch.no_grad():
            unet_output = self.unet(image)
            # U-Net returns: (final, [aux], decoder_features) or just (final, decoder_features)
            if len(unet_output) == 3:
                mask, _, decoder_features = unet_output
            else:
                mask, decoder_features = unet_output

            if isinstance(mask, tuple):
                mask = mask[0]
            mask = torch.sigmoid(mask)  # (B, 3, 512, 512)

        # 2. Pre-process mask and image (do interpolation once)
        # Resize to DenseNet input size early to reduce computation
        image_small = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        mask_small = F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)

        # 3. Extract clinical features (on smaller mask for speed)
        clinical_features = self.clinical_extractor(mask_small)  # (B, 5)

        # 4. Generate Canny edges (on smaller mask)
        edges = self.canny(mask_small)  # (B, 3, 224, 224)

        # 5. DenseNet121 features from RGB + Mask + Edges (already at 224x224)
        combined_input = torch.cat([image_small, mask_small, edges], dim=1)  # (B, 9, 224, 224)
        densenet_features = self.densenet(combined_input)  # (B, 1024)

        # 6. Multi-scale feature fusion
        multiscale_features = self.multiscale_fusion(decoder_features)  # (B, 512)

        # 7. Fuse all features
        fused_features = torch.cat([
            densenet_features,
            multiscale_features,
            clinical_features
        ], dim=1)  # (B, 1541)

        # 8. Classification
        logits = self.classifier(fused_features)  # (B, 1)

        return logits


# ============================================================================
# Dataset
# ============================================================================

class GlaucomaDataset(Dataset):
    """Dataset for glaucoma classification"""
    def __init__(self, rg_dir, nrg_dir, transform=None):
        self.transform = transform

        self.rg_images = [
            os.path.join(rg_dir, f) for f in os.listdir(rg_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.nrg_images = [
            os.path.join(nrg_dir, f) for f in os.listdir(nrg_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.image_paths = self.rg_images + self.nrg_images
        self.labels = [1] * len(self.rg_images) + [0] * len(self.nrg_images)

        print(f"Loaded {len(self.rg_images)} RG, {len(self.nrg_images)} NRG images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ============================================================================
# Loss Function
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for binary classification"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha >= 0:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss
        return loss.mean()


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs)

    return epoch_loss, epoch_acc, epoch_auc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs)

    return epoch_loss, epoch_acc, epoch_auc, all_preds, all_labels, all_probs


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("=" * 80)
    print("Model 4: U-Net V6 + CED + DenseNet121 Enhanced")
    print("=" * 80)

    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

    # Datasets
    print("\nLoading datasets...")
    train_dataset = GlaucomaDataset(Config.TRAIN_RG, Config.TRAIN_NRG, train_transform)
    val_dataset = GlaucomaDataset(Config.VAL_RG, Config.VAL_NRG, val_transform)
    test_dataset = GlaucomaDataset(Config.TEST_RG, Config.TEST_NRG, val_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    # Model
    print("\nInitializing Model 4 Enhanced...")
    model = Model4_Enhanced(
        unet_weights_path=Config.UNET_WEIGHTS,
        dropout_rates=Config.DROPOUT_RATES
    ).to(device)

    print(f"\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (U-Net): {total_params - trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler()

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 80)

        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc, val_auc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"[BEST] Saved model with Val AUC: {val_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        scheduler.step()

    # Test evaluation
    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    test_loss, test_acc, test_auc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_cm = confusion_matrix(test_labels, test_preds)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(test_cm)

    # Save results
    results = {
        'model': 'Model 4: U-Net V6 + CED + DenseNet121 Enhanced',
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_loss': float(test_loss),
        'confusion_matrix': test_cm.tolist(),
        'best_val_auc': float(best_val_auc),
        'config': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'num_epochs': Config.NUM_EPOCHS,
            'dropout_rates': Config.DROPOUT_RATES,
            'features': '1024 DenseNet + 512 Multi-scale + 5 Clinical = 1541-dim (optimized)'
        }
    }

    with open(Config.RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {Config.RESULTS_PATH}")
    print("Training complete!")


if __name__ == '__main__':
    main()
