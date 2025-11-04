"""
Model 1 IMPROVED: U-Net + DenseNet-121 Glaucoma Classifier
================================================================================

IMPROVEMENTS APPLIED:
Phase 1:
- Class-weighted loss function for better sensitivity
- Increased dropout (0.5 -> 0.6) for better regularization
- Weight decay (L2 regularization) added
- Increased early stopping patience (10 -> 15)

Phase 2:
- Focal Loss for hard example mining
- Enhanced data augmentation (vertical flip, crop, affine)
- CosineAnnealingWarmRestarts scheduler
- Increased max epochs (50 -> 75)

Expected Improvements:
- Baseline: 78.7% accuracy, 0.8733 AUC, 75.58% sensitivity
- Target: 85-87% accuracy, 0.92-0.94 AUC, 85-88% sensitivity

Created: 2025-11-04 (Improved Version)
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
    """Model 1 IMPROVED Configuration"""

    # Paths - ONLY use processed_gpu folders
    DATA_ROOT = 'processed_gpu'
    TRAIN_RG = os.path.join(DATA_ROOT, 'train', 'RG')
    TRAIN_NRG = os.path.join(DATA_ROOT, 'train', 'NRG')
    VAL_RG = os.path.join(DATA_ROOT, 'validation', 'RG')
    VAL_NRG = os.path.join(DATA_ROOT, 'validation', 'NRG')
    TEST_RG = os.path.join(DATA_ROOT, 'test', 'RG')
    TEST_NRG = os.path.join(DATA_ROOT, 'test', 'NRG')

    # V5.1 U-Net weights
    UNET_WEIGHTS = 'best_segmentation_model_gpu_v51.pth'

    # Output paths
    MODEL_SAVE_PATH = 'model1_best_improved.pth'
    RESULTS_PATH = 'model1_results_improved.json'

    # IMPROVED Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 75  # Increased from 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01  # Added L2 regularization (Phase 1)
    EARLY_STOPPING_PATIENCE = 15  # Increased from 10 (Phase 1)

    # IMPROVED Dropout rates (Phase 1)
    DROPOUT_RATES = [0.6, 0.35, 0.25]  # Increased from [0.5, 0.3, 0.2]

    # Focal Loss parameters (Phase 2)
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    USE_FOCAL_LOSS = True

    # Architecture
    UNET_INPUT_SIZE = 512  # U-Net expects 512×512
    DENSENET_INPUT_SIZE = 224  # DenseNet-121 expects 224×224
    NUM_CLASSES = 1  # Binary classification

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4  # Increased for better data loading
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
    """U-Net with ResNet34 encoder (V5.1 pretrained weights)"""

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
            print(f"[WARN] V5.1 weights not found at {weights_path}")

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()  # Set to evaluation mode permanently

    def forward(self, x):
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

        # Final upsample to 512x512
        x = self.up5(x)
        x = self.conv5(x)

        return self.outc(x)

    @torch.no_grad()  # Never compute gradients for U-Net
    def generate_mask(self, x):
        logits = self.forward(x)
        mask = F.softmax(logits, dim=1)  # Convert to probabilities
        return mask


# ============================================================================
# Focal Loss (Phase 2 Improvement)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples

    Focuses learning on hard misclassified examples by down-weighting
    easy examples. Particularly effective for medical imaging where
    subtle features distinguish positive/negative cases.

    Paper: Lin et al. "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (B, 1)
            targets: Labels (B,) in range [0, 1]

        Returns:
            loss: Focal loss value
        """
        # Binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs.squeeze(),
            targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Compute pt (probability of true class)
        pt = torch.exp(-bce_loss)

        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Focal loss: -alpha * (1 - pt)^gamma * log(pt)
        focal_loss = self.alpha * focal_term * bce_loss

        return focal_loss.mean()


# ============================================================================
# DenseNet-121 Classifier (IMPROVED)
# ============================================================================

class DenseNet121Classifier(nn.Module):
    """DenseNet-121 classifier with IMPROVED dropout regularization"""

    def __init__(self, num_classes=1, dropout_rates=[0.6, 0.35, 0.25]):
        super(DenseNet121Classifier, self).__init__()

        # Load pretrained DenseNet-121
        densenet = models.densenet121(pretrained=True)

        # Extract features (everything except classifier)
        self.features = densenet.features
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        self.features.add_module('avgpool_final', nn.AdaptiveAvgPool2d((1, 1)))

        # Custom classification head with IMPROVED dropout
        num_features = densenet.classifier.in_features  # 1024
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),  # 0.6 (was 0.5)
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),  # 0.35 (was 0.3)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[2]),  # 0.25 (was 0.2)
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        logits = self.classifier(features)
        return logits


# ============================================================================
# Model 1 IMPROVED: U-Net + DenseNet-121
# ============================================================================

class Model1_UNet_DenseNet(nn.Module):
    """Model 1 IMPROVED: Two-stage glaucoma classification"""

    def __init__(self, unet_weights_path=None, freeze_unet=True):
        super(Model1_UNet_DenseNet, self).__init__()

        # Stage 1: U-Net for segmentation (frozen)
        self.unet = UNet_V51(weights_path=unet_weights_path)

        # Stage 2: DenseNet-121 with IMPROVED dropout
        self.classifier = DenseNet121Classifier(
            num_classes=1,
            dropout_rates=Config.DROPOUT_RATES
        )

        # Ensure U-Net is frozen
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()

    def forward(self, x, return_mask=False):
        # Stage 1: Generate segmentation mask (frozen, no gradients)
        with torch.no_grad():
            mask = self.unet.generate_mask(x)  # (B, 3, 512, 512)

        # Resize mask for DenseNet-121 (512×512 → 224×224)
        mask_resized = F.interpolate(
            mask,
            size=(Config.DENSENET_INPUT_SIZE, Config.DENSENET_INPUT_SIZE),
            mode='bilinear',
            align_corners=False
        )  # (B, 3, 224, 224)

        # Stage 2: Classify segmentation mask
        logits = self.classifier(mask_resized)  # (B, 1)

        if return_mask:
            return logits, mask
        return logits


# ============================================================================
# Dataset with ENHANCED Augmentation (Phase 2)
# ============================================================================

class GlaucomaDataset(Dataset):
    """Dataset with ENHANCED data augmentation"""

    def __init__(self, rg_dir, nrg_dir, transform=None, augment=False):
        self.image_paths = []
        self.labels = []

        # Load RG (glaucoma) images
        if os.path.exists(rg_dir):
            rg_files = [f for f in os.listdir(rg_dir) if f.endswith('.jpg')]
            self.image_paths.extend([os.path.join(rg_dir, f) for f in rg_files])
            self.labels.extend([1] * len(rg_files))

        # Load NRG (normal) images
        if os.path.exists(nrg_dir):
            nrg_files = [f for f in os.listdir(nrg_dir) if f.endswith('.jpg')]
            self.image_paths.extend([os.path.join(nrg_dir, f) for f in nrg_files])
            self.labels.extend([0] * len(nrg_files))

        self.transform = transform
        self.augment = augment

        print(f"Dataset: {len(self.image_paths)} images ({sum(self.labels)} RG, {len(self.labels)-sum(self.labels)} NRG)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 512×512 (U-Net input size)
        image = cv2.resize(image, (Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE))

        # ENHANCED data augmentation (Phase 2)
        if self.augment:
            image = self._augment_enhanced(image)

        # Convert to PIL for transforms
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    def _augment_enhanced(self, image):
        """
        ENHANCED data augmentation (Phase 2)

        Additions:
        - Vertical flip
        - Random crop + resize
        - Affine transformations (rotation + translation)
        - Color jitter (brightness, contrast)
        """
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)

        # Random vertical flip (30% chance) - NEW
        if np.random.random() > 0.7:
            image = cv2.flip(image, 0)

        # Random crop + resize (NEW)
        if np.random.random() > 0.5:
            h, w = image.shape[:2]
            scale = np.random.uniform(0.85, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)

            # Random crop
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            image = image[top:top+new_h, left:left+new_w]

            # Resize back
            image = cv2.resize(image, (w, h))

        # Random rotation + translation (NEW)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            tx = np.random.uniform(-0.1, 0.1) * image.shape[1]
            ty = np.random.uniform(-0.1, 0.1) * image.shape[0]

            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty

            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Random brightness + contrast (ENHANCED)
        if np.random.random() > 0.5:
            # Brightness
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if np.random.random() > 0.5:
            # Contrast
            mean = image.mean()
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        return image


# ============================================================================
# Training Functions
# ============================================================================

def create_data_loaders():
    """Create train, validation, and test data loaders"""

    # ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

    # Training dataset (with ENHANCED augmentation)
    train_dataset = GlaucomaDataset(
        rg_dir=Config.TRAIN_RG,
        nrg_dir=Config.TRAIN_NRG,
        transform=transform,
        augment=True
    )

    # Validation dataset (no augmentation)
    val_dataset = GlaucomaDataset(
        rg_dir=Config.VAL_RG,
        nrg_dir=Config.VAL_NRG,
        transform=transform,
        augment=False
    )

    # Test dataset (no augmentation)
    test_dataset = GlaucomaDataset(
        rg_dir=Config.TEST_RG,
        nrg_dir=Config.TEST_NRG,
        transform=transform,
        augment=False
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, train_dataset


def calculate_metrics(y_true, y_pred, y_scores):
    """Calculate classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Recall
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    }

    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['specificity'] = 0.0

    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()

    # Keep U-Net in eval mode (it's frozen)
    model.unet.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Predictions
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs > 0.5).float()

        # Accumulate
        running_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Metrics
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_scores)

    return epoch_loss, metrics


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            # Predictions
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).float()

            # Accumulate
            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    # Metrics
    epoch_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_scores)

    return epoch_loss, metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    """Train model with IMPROVED settings"""

    scaler = GradScaler()

    best_auc = 0.0
    best_model_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }

    print("\n" + "="*80)
    print("Starting Training (IMPROVED Model 1)")
    print("="*80)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler (CosineAnnealingWarmRestarts)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        # Print results
        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  Val Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Check for improvement
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  [OK] Best model saved! (AUC: {best_auc:.4f})")

            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics
            }, Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")

            # Early stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n[OK] Early stopping triggered at epoch {epoch+1}")
                break

        # GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  GPU Memory: {memory_used:.2f} GB")

    return history, best_model_state


# ============================================================================
# Main Training
# ============================================================================

def main():
    print("="*80)
    print("Model 1 IMPROVED: U-Net + DenseNet-121 Glaucoma Classifier")
    print("="*80)

    print("\nIMPROVEMENTS APPLIED:")
    print("  Phase 1: Class weights, Dropout 0.6, Weight decay, Early stopping patience 15")
    print("  Phase 2: Focal Loss, Enhanced augmentation, Cosine scheduler, Max epochs 75")
    print(f"\nExpected: 85-87% accuracy, 0.92-0.94 AUC, 85-88% sensitivity")
    print(f"Baseline: 78.7% accuracy, 0.8733 AUC, 75.58% sensitivity")

    # Device
    device = Config.DEVICE
    print(f"\nDevice: {device}")

    # Data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders()

    # Calculate class weights for Focal Loss (Phase 1)
    num_rg = sum(train_dataset.labels)
    num_nrg = len(train_dataset.labels) - num_rg
    pos_weight = torch.tensor([num_nrg / num_rg]).to(device)
    print(f"\nClass distribution: {num_rg} RG, {num_nrg} NRG")
    print(f"Positive class weight: {pos_weight.item():.4f}")

    # Model
    print("\nInitializing IMPROVED Model 1...")
    model = Model1_UNet_DenseNet(
        unet_weights_path=Config.UNET_WEIGHTS,
        freeze_unet=True
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (U-Net): {frozen_params:,}")

    # Loss function: Focal Loss (Phase 2)
    if Config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=Config.FOCAL_ALPHA,
            gamma=Config.FOCAL_GAMMA,
            pos_weight=pos_weight
        )
        print(f"\nLoss: Focal Loss (alpha={Config.FOCAL_ALPHA}, gamma={Config.FOCAL_GAMMA})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"\nLoss: BCEWithLogitsLoss (weighted)")

    # Optimizer: AdamW with weight decay (Phase 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    print(f"Optimizer: AdamW (lr={Config.LEARNING_RATE}, weight_decay={Config.WEIGHT_DECAY})")

    # Scheduler: CosineAnnealingWarmRestarts (Phase 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=10, eta_min=1e-6)")

    # Validation: Test forward pass
    print("\n" + "="*80)
    print("Validation: Testing forward pass...")
    print("="*80)

    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(train_loader))
        test_images = test_images.to(device)
        test_logits = model(test_images)
        test_loss = criterion(test_logits, test_labels.to(device))

        print(f"[OK] Forward pass successful!")
        print(f"  Batch size: {test_images.shape[0]}")
        print(f"  Input shape: {test_images.shape}")
        print(f"  Output shape: {test_logits.shape}")
        print(f"  Loss: {test_loss.item():.4f}")

        if test_loss.item() > 10:
            print(f"\n[WARN] Loss is very high ({test_loss.item():.4f})")
            print(f"[WARN] This might indicate a problem with the model or data")
            user_input = input("Continue training? (y/n): ")
            if user_input.lower() != 'y':
                print("Training aborted.")
                return

    # Train
    print("\n" + "="*80)
    print("Starting Full Training...")
    print("="*80)

    history, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS
    )

    # Load best model
    model.load_state_dict(best_model_state)

    # Test
    print("\n" + "="*80)
    print("Testing Best Model...")
    print("="*80)

    _, test_metrics = validate(model, test_loader, criterion, device)

    print("\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f} ({test_metrics['sensitivity']*100:.2f}%)")
    print(f"  Specificity: {test_metrics['specificity']:.4f} ({test_metrics['specificity']*100:.2f}%)")
    print(f"  Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")

    # Confusion matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(f"  TN: {tn:4d} | FP: {fp:4d}")
    print(f"  FN: {fn:4d} | TP: {tp:4d}")

    # Save results
    results = {
        'model': 'Model 1 IMPROVED: U-Net + DenseNet-121',
        'improvements': {
            'phase1': 'Class weights, Dropout 0.6, Weight decay 0.01, Early stopping patience 15',
            'phase2': 'Focal Loss, Enhanced augmentation, Cosine scheduler, Max epochs 75'
        },
        'baseline': {
            'accuracy': 0.7870,
            'auc': 0.8733,
            'sensitivity': 0.7558
        },
        'architecture': {
            'stage1': 'U-Net (ResNet34 encoder, V5.1 weights, frozen)',
            'stage2': 'DenseNet-121 (ImageNet pretrained, trainable, dropout 0.6)'
        },
        'data_source': 'processed_gpu',
        'training': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'weight_decay': Config.WEIGHT_DECAY,
            'epochs_trained': len(history['train_loss']),
            'best_val_auc': float(max(history['val_auc']))
        },
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'sensitivity': float(test_metrics['sensitivity']),
            'specificity': float(test_metrics['specificity']),
            'precision': float(test_metrics['precision']),
            'f1': float(test_metrics['f1']),
            'auc': float(test_metrics['auc'])
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'train_auc': [float(x) for x in history['train_auc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']],
            'val_auc': [float(x) for x in history['val_auc']]
        }
    }

    with open(Config.RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {Config.RESULTS_PATH}")
    print(f"[OK] Model saved to {Config.MODEL_SAVE_PATH}")

    # Compare with baseline
    print("\n" + "="*80)
    print("IMPROVEMENT vs BASELINE:")
    print("="*80)

    acc_improvement = (test_metrics['accuracy'] - 0.7870) * 100
    auc_improvement = (test_metrics['auc'] - 0.8733)
    sens_improvement = (test_metrics['sensitivity'] - 0.7558) * 100

    print(f"Accuracy: {test_metrics['accuracy']*100:.2f}% (baseline: 78.70%, +{acc_improvement:.2f}%)")
    print(f"AUC: {test_metrics['auc']:.4f} (baseline: 0.8733, +{auc_improvement:.4f})")
    print(f"Sensitivity: {test_metrics['sensitivity']*100:.2f}% (baseline: 75.58%, +{sens_improvement:.2f}%)")

    print("\n" + "="*80)
    print("Model 1 IMPROVED Training Complete!")
    print("="*80)
    print(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
