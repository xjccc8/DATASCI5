"""
Model 1c-V6: U-Net V6 + Canny Edge Detection + ResNet-50 Classifier
================================================================================

Architecture:
    Fundus Image (512×512)
    → U-Net V6 (frozen, 95.98% Dice) [UPGRADED FROM V5.1]
    → Segmentation Mask (512×512×3)
    → Canny Edge Detection
    → Concatenate [Mask + Edges] (512×512×6)
    → Resize to 256×256×6
    → ResNet-50 Classifier (modified for 6 channels)
    → Glaucoma Probability [0, 1]

Purpose:
- Test pipeline approach with high-quality V6 masks (95.98% Dice)
- Previous: Model 1c with V5.1 (62.5% Dice): 83.51% acc, 0.9251 AUC
- Expected: 92-95% accuracy with V6 masks (+9-12% improvement)
- Goal: Match or exceed Model 4 (94.80% acc, 0.9818 AUC)

Key Features:
- U-Net V6 (frozen, 95.98% Dice accuracy) - ATTENTION UNET + DEEP SUPERVISION
- Canny edge detection on high-quality segmentation masks
- ResNet-50 classifier (6-channel input: 3 mask + 3 edges)
- Mixed precision training (FP16)
- Expected: 92-95% accuracy (near-perfect masks should enable excellent classification)

Comparison with related models:
- Model 1 (U-Net + DenseNet121): 78.70% acc, 0.8733 AUC
- Model 1b (U-Net + ResNet50): 76.49% acc, 0.8424 AUC
- Model 2 (U-Net + Canny + DenseNet121): 76.23% acc, 0.8457 AUC
- Model 1c (This model): U-Net + Canny + ResNet50 (TBD)

Created: 2025-11-11
Data Source: processed_gpu/train/, processed_gpu/test/, processed_gpu/validation/
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


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Model 1c: U-Net + Canny + ResNet-50 Configuration"""

    # Paths (relative to project root)
    DATA_ROOT = '../processed_gpu'
    TRAIN_RG = os.path.join(DATA_ROOT, 'train', 'RG')
    TRAIN_NRG = os.path.join(DATA_ROOT, 'train', 'NRG')
    VAL_RG = os.path.join(DATA_ROOT, 'validation', 'RG')
    VAL_NRG = os.path.join(DATA_ROOT, 'validation', 'NRG')
    TEST_RG = os.path.join(DATA_ROOT, 'test', 'RG')
    TEST_NRG = os.path.join(DATA_ROOT, 'test', 'NRG')

    # U-Net weights (V6 - 95.98% Dice, upgraded from V5.1)
    UNET_WEIGHTS = '../02_scripts/best_segmentation_model_gpu_v6.pth'

    # Output
    MODEL_SAVE_PATH = 'model1c_best.pth'
    RESULTS_PATH = 'model1c_results.json'

    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 15

    # Dropout rates for ResNet-50 classifier
    DROPOUT_RATES = [0.6, 0.35, 0.25]

    # Focal Loss parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Architecture
    UNET_INPUT_SIZE = 512
    RESNET50_INPUT_SIZE = 256  # Standard for ResNet-50
    NUM_CLASSES = 1  # Binary classification

    # Canny edge detection parameters
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# U-Net Segmentation Model (Frozen) - V5.1 Compatible Architecture
# ============================================================================

class ResNetEncoder(nn.Module):
    """ResNet34 pretrained encoder (from V5.1)"""
    def __init__(self):
        super().__init__()
        import torchvision.models as models_tv
        resnet = models_tv.resnet34(pretrained=True)

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


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 (from V5.1)"""
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


class UNet_V51(nn.Module):
    """U-Net with ResNet34 encoder (V5.1 pretrained weights) - FROZEN"""

    def __init__(self, weights_path=None, n_classes=3):
        super(UNet_V51, self).__init__()
        self.n_classes = n_classes

        # Pretrained encoder (from V5.1)
        self.encoder = ResNetEncoder()

        # Decoder (from V5.1)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # Final upsample to reach 512x512
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(64, 64)

        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # Load V5.1 pretrained weights
        if weights_path and os.path.exists(weights_path):
            print(f"Loading V5.1 U-Net weights from {weights_path}...")
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                dice = checkpoint.get('dice_accuracy', 'N/A')
                print(f"[OK] Loaded V5.1 weights (Dice: {dice}%)")
            elif 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
                print("[OK] Loaded V5.1 weights")
            else:
                self.load_state_dict(checkpoint)
                print("[OK] Loaded V5.1 weights")
        else:
            print(f"[WARN] V5.1 weights not found at {weights_path}, using random initialization")

    def forward(self, x):
        """Forward pass - standard U-Net"""
        # Encoder
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

        x = self.up5(x)
        x = self.conv5(x)

        logits = self.outc(x)
        return logits

    def generate_mask(self, x):
        """Generate segmentation mask (no sigmoid, raw logits)"""
        with torch.no_grad():
            logits = self.forward(x)
            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)
            return probs


# ============================================================================
# Canny Edge Detection Module
# ============================================================================

class CannyEdgeDetector(nn.Module):
    """Canny edge detection on GPU tensors"""

    def __init__(self, threshold1=50, threshold2=150):
        super(CannyEdgeDetector, self).__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def forward(self, mask):
        """
        Apply Canny edge detection to segmentation mask

        Args:
            mask: (B, 3, H, W) tensor in [0, 1] range

        Returns:
            edges: (B, 3, H, W) tensor with detected edges
        """
        B, C, H, W = mask.shape
        edges = torch.zeros_like(mask)

        # Convert to numpy for OpenCV processing (per batch item)
        for i in range(B):
            mask_np = mask[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
            mask_np = (mask_np * 255).astype(np.uint8)

            # Apply Canny per channel
            edges_np = np.zeros_like(mask_np)
            for c in range(C):
                edges_np[:, :, c] = cv2.Canny(
                    mask_np[:, :, c],
                    self.threshold1,
                    self.threshold2
                )

            # Convert back to tensor
            edges_tensor = torch.from_numpy(edges_np).float() / 255.0
            edges[i] = edges_tensor.permute(2, 0, 1)  # (3, H, W)

        return edges.to(mask.device)


# ============================================================================
# ResNet-50 Classifier (6-channel input)
# ============================================================================

class ResNet50Classifier_6Channel(nn.Module):
    """
    ResNet-50 classifier modified for 6-channel input

    Input: 6 channels (3 for mask + 3 for Canny edges)
    Output: 1 logit for binary classification
    """

    def __init__(self, num_classes=1, dropout_rates=[0.6, 0.35, 0.25]):
        super(ResNet50Classifier_6Channel, self).__init__()

        # Load pretrained ResNet-50
        resnet50 = models.resnet50(pretrained=True)

        # Modify first conv layer for 6-channel input
        self.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new conv1 with pretrained weights
        # Copy pretrained weights for first 3 channels, duplicate for channels 4-6
        with torch.no_grad():
            pretrained_weight = resnet50.conv1.weight.clone()
            self.conv1.weight[:, :3, :, :] = pretrained_weight
            self.conv1.weight[:, 3:, :, :] = pretrained_weight  # Duplicate for edges

        # Use rest of ResNet-50 backbone
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (B, 6, H, W) tensor with mask + edges

        Returns:
            logits: (B, 1) classification logits
        """
        # ResNet-50 backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # Classification head
        logits = self.classifier(x)
        return logits


# ============================================================================
# Complete Model 1c
# ============================================================================

class Model1c_UNet_Canny_ResNet50(nn.Module):
    """
    Model 1c: Three-stage glaucoma classification with Canny edge detection

    Stage 1 (Frozen): U-Net generates segmentation mask
    Stage 2 (CPU): Canny edge detection extracts boundaries
    Stage 3 (Trainable): ResNet-50 classifies mask + edges

    Architecture:
        Input (512×512×3)
        → U-Net [FROZEN]
        → Mask (512×512×3)
        → Canny Detection
        → Concatenate [Mask, Edges] (512×512×6)
        → Resize (256×256×6)
        → ResNet-50 [TRAINABLE]
        → Logit (1)
    """

    def __init__(self, unet_weights_path=None, freeze_unet=True,
                 canny_threshold1=50, canny_threshold2=150):
        super(Model1c_UNet_Canny_ResNet50, self).__init__()

        # Stage 1: U-Net for segmentation (frozen)
        # Load V6 directly from checkpoint without architecture definition
        # (V6 is self-contained in the checkpoint)
        from pathlib import Path
        if unet_weights_path and Path(unet_weights_path).exists():
            print(f"\nLoading U-Net from {unet_weights_path}...")
            checkpoint = torch.load(unet_weights_path, map_location='cpu', weights_only=False)
            dice_acc = checkpoint.get('dice_accuracy', 'N/A')
            print(f"[OK] U-Net Dice Accuracy: {dice_acc}%")

            # Import V6 architecture from training script
            import sys
            sys.path.insert(0, str(Path(unet_weights_path).parent))
            try:
                from gpu_optimized_mask_generation_v6 import AttentionUNet_V6
                self.unet = AttentionUNet_V6(n_classes=3, use_attention=True, deep_supervision=True)
                self.unet.load_state_dict(checkpoint['model_state_dict'])
                print(f"[OK] Loaded V6 Attention U-Net architecture")
            except ImportError:
                print("[ERROR] Could not import V6 architecture from gpu_optimized_mask_generation_v6.py")
                print("[FALLBACK] Using V5.1 architecture")
                self.unet = UNet_V51(weights_path=unet_weights_path)
        else:
            print(f"[WARN] U-Net weights not found at {unet_weights_path}")
            self.unet = UNet_V51(weights_path=None)

        # Stage 2: Canny edge detection
        self.canny = CannyEdgeDetector(
            threshold1=canny_threshold1,
            threshold2=canny_threshold2
        )

        # Stage 3: ResNet-50 for classification (trainable)
        self.classifier = ResNet50Classifier_6Channel(num_classes=1)

        # Ensure U-Net is frozen
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()

    def forward(self, x, return_mask=False, return_edges=False):
        """
        Forward pass through three-stage architecture

        Args:
            x: Fundus image (B, 3, 512, 512)
            return_mask: If True, also return segmentation mask
            return_edges: If True, also return edge map

        Returns:
            logits: (B, 1) classification logits
            (optional) mask: (B, 3, 512, 512) segmentation mask
            (optional) edges: (B, 3, 512, 512) edge map
        """
        # Stage 1: U-Net segmentation (frozen, no gradient)
        with torch.no_grad():
            # V6 returns (output, aux_outputs) during training, just output during eval
            # V5.1 has generate_mask() method
            if hasattr(self.unet, 'generate_mask'):
                mask = self.unet.generate_mask(x)  # (B, 3, 512, 512) in [0, 1]
            else:
                # V6 forward pass
                output = self.unet(x)
                if isinstance(output, tuple):  # Deep supervision during training
                    mask_logits = output[0]
                else:
                    mask_logits = output
                # Apply softmax to get probabilities
                mask = torch.softmax(mask_logits, dim=1)  # (B, 3, 512, 512) in [0, 1]

        # Stage 2: Canny edge detection (on CPU/GPU)
        edges = self.canny(mask)  # (B, 3, 512, 512) in [0, 1]

        # Concatenate mask and edges
        mask_edges = torch.cat([mask, edges], dim=1)  # (B, 6, 512, 512)

        # Resize to ResNet-50 input size (256×256)
        mask_edges_resized = F.interpolate(
            mask_edges,
            size=(Config.RESNET50_INPUT_SIZE, Config.RESNET50_INPUT_SIZE),
            mode='bilinear',
            align_corners=False
        )

        # Stage 3: ResNet-50 classification
        logits = self.classifier(mask_edges_resized)  # (B, 1)

        # Return outputs
        if return_mask and return_edges:
            return logits, mask, edges
        elif return_mask:
            return logits, mask
        elif return_edges:
            return logits, edges
        else:
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
    """Dataset for fundus images (no masks needed)"""

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
    print("Model 1c: U-Net + Canny Edge Detection + ResNet-50")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Data Source: {Config.DATA_ROOT}/")
    print(f"U-Net Weights: {Config.UNET_WEIGHTS}")
    print(f"Canny Thresholds: ({Config.CANNY_THRESHOLD1}, {Config.CANNY_THRESHOLD2})")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print("=" * 80)
    print()

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )

    # Create model
    print("\nCreating Model 1c...")
    model = Model1c_UNet_Canny_ResNet50(
        unet_weights_path=Config.UNET_WEIGHTS,
        freeze_unet=True,
        canny_threshold1=Config.CANNY_THRESHOLD1,
        canny_threshold2=Config.CANNY_THRESHOLD2
    ).to(Config.DEVICE)

    # Print parameter counts
    param_counts = model.count_parameters()
    print("\nModel Parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen (U-Net): {param_counts['frozen']:,}")

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
    print("Starting Model 1c Training: U-Net + Canny + ResNet-50")
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
    print("Testing Best Model")
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
        'model': 'Model 1c: U-Net + Canny Edge Detection + ResNet-50',
        'architecture': {
            'stage1': 'U-Net V5.1 (frozen, 62.5% Dice)',
            'stage2': 'Canny edge detection',
            'stage3': 'ResNet-50 classifier (6-channel input)',
            'input': 'Segmentation mask (3ch) + Canny edges (3ch)',
            'approach': 'Two-stage pipeline with edge enhancement'
        },
        'canny_parameters': {
            'threshold1': Config.CANNY_THRESHOLD1,
            'threshold2': Config.CANNY_THRESHOLD2
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
        'model_parameters': param_counts
    }

    with open(Config.RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {Config.RESULTS_PATH}")
    print(f"Best model saved to {Config.MODEL_SAVE_PATH}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
