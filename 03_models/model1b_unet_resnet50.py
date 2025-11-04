"""
Model 1b: U-Net + ResNet-50 Classifier (Compare with Model 1 DenseNet-121)
================================================================================

Architecture:
    Fundus Image (512×512)
    → U-Net (frozen, V5.1 weights, 62.5% Dice)
    → Segmentation Mask (512×512×3)
    → Resize to 256×256×3
    → ResNet-50 Classifier (pretrained, trainable)
    → Glaucoma Probability [0, 1]

Purpose:
- Compare ResNet-50 vs DenseNet-121 as classifier for U-Net masks
- Test if ResNet-50's larger capacity helps with noisy U-Net masks
- Model 1 (DenseNet-121): 79.48% acc, 0.8838 AUC
- Model 3 (ResNet-50 direct): 94.03% acc, 0.9795 AUC
- Target: See if ResNet-50 classifier improves U-Net pipeline

Key Differences from Model 1:
- Classifier: ResNet-50 (23.5M params) instead of DenseNet-121 (7.6M params)
- Input size: 256×256 (ResNet-50 standard) instead of 224×224 (DenseNet-121)
- Same U-Net V5.1 (frozen)
- Same training hyperparameters

Expected: 85-88% accuracy (better than Model 1, but below Model 3)

Created: 2025-11-04
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
    """Model 1b: U-Net + ResNet-50 Configuration"""

    # Paths
    DATA_ROOT = 'processed_gpu'
    TRAIN_RG = os.path.join(DATA_ROOT, 'train', 'RG')
    TRAIN_NRG = os.path.join(DATA_ROOT, 'train', 'NRG')
    VAL_RG = os.path.join(DATA_ROOT, 'validation', 'RG')
    VAL_NRG = os.path.join(DATA_ROOT, 'validation', 'NRG')
    TEST_RG = os.path.join(DATA_ROOT, 'test', 'RG')
    TEST_NRG = os.path.join(DATA_ROOT, 'test', 'NRG')

    # U-Net weights (V5.1)
    UNET_WEIGHTS = 'best_segmentation_model_gpu_v51.pth'

    # Output
    MODEL_SAVE_PATH = 'model1b_best.pth'
    RESULTS_PATH = 'model1b_results.json'

    # Training hyperparameters (same as Model 1)
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

        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def generate_mask(self, x):
        """Generate segmentation mask with softmax"""
        logits = self.forward(x)
        mask = F.softmax(logits, dim=1)
        return mask


# ============================================================================
# ResNet-50 Classifier (Modified for 3-channel masks)
# ============================================================================

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 classifier for 3-channel segmentation masks

    Input: Segmentation mask (B, 3, 256, 256)
    Output: Binary classification logits (B, 1)

    Architecture:
    - ResNet-50 backbone (pretrained on ImageNet)
    - Modified first conv for 3-channel masks
    - Custom classification head with dropout
    """

    def __init__(self, num_classes=1, dropout_rates=[0.6, 0.35, 0.25]):
        super(ResNet50Classifier, self).__init__()

        # Load pretrained ResNet-50
        resnet50 = models.resnet50(pretrained=True)

        # Use ResNet-50 backbone as-is (already accepts 3 channels)
        self.features = nn.Sequential(
            resnet50.conv1,      # 3 → 64 channels
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,     # 64 → 256
            resnet50.layer2,     # 256 → 512
            resnet50.layer3,     # 512 → 1024
            resnet50.layer4,     # 1024 → 2048
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Custom classification head
        num_features = 2048  # ResNet-50 output channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Classify 3-channel mask input

        Args:
            x: Mask tensor (B, 3, 256, 256)

        Returns:
            logits: Classification logits (B, 1)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples"""

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============================================================================
# Model 1b: U-Net + ResNet-50 (Two-Stage Pipeline)
# ============================================================================

class Model1b_UNet_ResNet50(nn.Module):
    """
    Model 1b: Two-stage glaucoma classification with ResNet-50 classifier

    Stage 1 (Frozen): U-Net generates segmentation mask from fundus image
    Stage 2 (Trainable): ResNet-50 classifies mask

    Architecture:
        Input (512×512×3)
        → U-Net [FROZEN]
        → Mask (512×512×3)
        → Resize (256×256×3)
        → ResNet-50 [TRAINABLE]
        → Logit (1)
    """

    def __init__(self, unet_weights_path=None, freeze_unet=True):
        super(Model1b_UNet_ResNet50, self).__init__()

        # Stage 1: U-Net for segmentation (frozen)
        self.unet = UNet_V51(weights_path=unet_weights_path)

        # Stage 2: ResNet-50 for classification (trainable)
        self.classifier = ResNet50Classifier(num_classes=1)

        # Ensure U-Net is frozen
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()

    def forward(self, x, return_mask=False):
        """
        Forward pass through two-stage architecture

        Args:
            x: Fundus image (B, 3, 512, 512)
            return_mask: If True, also return segmentation mask

        Returns:
            logits: Classification logits (B, 1)
            mask (optional): Segmentation mask (B, 3, 512, 512)
        """
        # Stage 1: Generate segmentation mask (frozen)
        with torch.no_grad():
            mask = self.unet.generate_mask(x)  # (B, 3, 512, 512)

        # Resize for ResNet-50 (512×512 → 256×256)
        mask_resized = F.interpolate(
            mask,
            size=(Config.RESNET50_INPUT_SIZE, Config.RESNET50_INPUT_SIZE),
            mode='bilinear',
            align_corners=False
        )  # (B, 3, 256, 256)

        # Stage 2: Classify (trainable)
        logits = self.classifier(mask_resized)  # (B, 1)

        if return_mask:
            return logits, mask
        return logits


# ============================================================================
# Dataset (Same as Model 1)
# ============================================================================

class GlaucomaDataset(Dataset):
    """Glaucoma classification dataset from processed_gpu"""

    def __init__(self, rg_dir, nrg_dir, transform=None, augment=False):
        self.images = []
        self.labels = []
        self.transform = transform
        self.augment = augment

        # Load RG images (glaucoma = 1)
        if os.path.exists(rg_dir):
            for img_name in os.listdir(rg_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(rg_dir, img_name))
                    self.labels.append(1)

        # Load NRG images (non-glaucoma = 0)
        if os.path.exists(nrg_dir):
            for img_name in os.listdir(nrg_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(nrg_dir, img_name))
                    self.labels.append(0)

        print(f"Dataset: {len(self.images)} images ({sum(self.labels)} RG, {len(self.labels)-sum(self.labels)} NRG)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 512×512 for U-Net
        image = cv2.resize(image, (Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE))

        # Convert to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Apply ImageNet normalization
        mean = torch.tensor(Config.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(Config.IMAGENET_STD).view(3, 1, 1)
        image = (image - mean) / std

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    # Keep U-Net in eval mode
    model.unet.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

    return epoch_loss, epoch_acc, epoch_auc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

    return epoch_loss, epoch_acc, epoch_auc


def evaluate(model, dataloader, device):
    """Comprehensive evaluation"""
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Testing', leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, pos_label=1)
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }

    return metrics


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("="*80)
    print("Model 1b: U-Net + ResNet-50 Classifier Training")
    print("="*80)
    print(f"\nComparing with:")
    print(f"  Model 1 (U-Net + DenseNet-121): 79.48% acc, 0.8838 AUC")
    print(f"  Model 3 (ResNet-50 direct): 94.03% acc, 0.9795 AUC")
    print(f"\nDevice: {Config.DEVICE}")
    print(f"U-Net: V5.1 (62.5% Dice, frozen)")
    print(f"Classifier: ResNet-50 (23.5M params, trainable)")
    print("="*80)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = GlaucomaDataset(Config.TRAIN_RG, Config.TRAIN_NRG)
    val_dataset = GlaucomaDataset(Config.VAL_RG, Config.VAL_NRG)
    test_dataset = GlaucomaDataset(Config.TEST_RG, Config.TEST_NRG)

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
    print("\nCreating Model 1b...")
    model = Model1b_UNet_ResNet50(
        unet_weights_path=Config.UNET_WEIGHTS,
        freeze_unet=True
    ).to(Config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable (ResNet-50): {trainable_params:,}")
    print(f"  Frozen (U-Net): {frozen_params:,}")

    # Loss and optimizer
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )
    scaler = GradScaler()

    # Training loop
    print(f"\n{'='*80}")
    print("Starting Model 1b Training")
    print(f"{'='*80}\n")

    best_val_auc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 80)

        # Train
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )

        # Validate
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"[BEST] Saved model (Val AUC: {val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered (patience={Config.EARLY_STOPPING_PATIENCE})")
            break

        scheduler.step()

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")

    # Load best model and evaluate on test set
    print(f"\n{'='*80}")
    print("Testing Model 1b on Test Set")
    print(f"{'='*80}\n")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    test_metrics = evaluate(model, test_loader, Config.DEVICE)

    print("\nTest Set Results:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f} ({test_metrics['sensitivity']*100:.2f}%)")
    print(f"  Specificity: {test_metrics['specificity']:.4f} ({test_metrics['specificity']*100:.2f}%)")
    print(f"  Precision:   {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"  F1-Score:    {test_metrics['f1']:.4f}")
    print(f"  AUC:         {test_metrics['auc']:.4f}")

    cm = np.array(test_metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")

    # Save results
    results = {
        'model': 'Model 1b: U-Net + ResNet-50',
        'comparison': {
            'model1_densenet121': {'accuracy': 0.7948, 'auc': 0.8838},
            'model3_resnet50_direct': {'accuracy': 0.9403, 'auc': 0.9795}
        },
        'architecture': {
            'stage1': 'U-Net (ResNet34 encoder, V5.1 weights, frozen)',
            'stage2': 'ResNet-50 (ImageNet pretrained, trainable)'
        },
        'training': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'epochs_trained': epoch + 1,
            'best_val_auc': best_val_auc,
            'training_time_minutes': training_time / 60
        },
        'test_metrics': test_metrics,
        'history': history
    }

    with open(Config.RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {Config.RESULTS_PATH}")
    print(f"✓ Model saved to {Config.MODEL_SAVE_PATH}")

    print(f"\n{'='*80}")
    print("Model 1b Training Complete!")
    print(f"{'='*80}")
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print(f"\nComparison:")
    print(f"  Model 1 (DenseNet-121): 79.48% acc, 0.8838 AUC")
    print(f"  Model 1b (ResNet-50):   {test_metrics['accuracy']*100:.2f}% acc, {test_metrics['auc']:.4f} AUC")
    print(f"  Model 3 (Direct):       94.03% acc, 0.9795 AUC")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
