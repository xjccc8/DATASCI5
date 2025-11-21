"""
Model 0: U-Net V6 + ResNet50 Baseline (No Enhancements)
================================================================================

Architecture:
    Fundus Image (512×512)
    → U-Net V6 (frozen, 95.98% Dice, Attention Gates + Deep Supervision)
    → Segmentation Mask (512×512×3)
    → Concatenate [RGB + Mask] → 6 channels
    → Resize to 256×256
    → ResNet-50 Classifier
    → Glaucoma Probability [0, 1]

Purpose:
- Clean baseline for pipeline approach with U-Net V6
- Compare with Model 2 (same + enhancements) to measure enhancement benefit
- NO Canny edges, NO multi-scale features, NO clinical features
- Just simple: Image + Mask → ResNet50 → Classification

Key Features:
- U-Net V6 (frozen, 95.98% Dice) with ResNet50 encoder + Attention Gates
- ResNet-50 classifier with 6-channel input (RGB + 3-channel mask)
- Simple concatenation - no fancy feature fusion
- Expected: ~92-93% accuracy (baseline for enhanced models)

Created: 2025-11-17
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
    """Model 0 Configuration"""

    # Paths
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
    MODEL_SAVE_PATH = 'model0_best.pth'
    RESULTS_PATH = 'model0_results.json'

    # Training hyperparameters
    BATCH_SIZE = 24
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 15

    # Dropout rates
    DROPOUT_RATES = [0.5, 0.3]

    # Architecture
    UNET_INPUT_SIZE = 512
    RESNET50_INPUT_SIZE = 256
    NUM_CLASSES = 1

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 6
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


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
        # Initial conv
        x1 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x2 = self.maxpool(x1)  # (B, 64, H/4, W/4)

        # ResNet blocks
        x2 = self.layer1(x2)  # (B, 256, H/4, W/4)
        x3 = self.layer2(x2)  # (B, 512, H/8, W/8)
        x4 = self.layer3(x3)  # (B, 1024, H/16, W/16)
        x5 = self.layer4(x4)  # (B, 2048, H/32, W/32)

        return x1, x2, x3, x4, x5


class DecoderBlock(nn.Module):
    """Decoder block with attention gate and double convolution"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()

        self.use_attention = use_attention

        # Attention gate
        if use_attention:
            self.attention = AttentionGate(
                F_g=out_channels,
                F_l=skip_channels,
                F_int=out_channels
            )

        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        # Double convolution
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


class AttentionUNet_V6(nn.Module):
    """U-Net V6 with ResNet50 Encoder, Attention Gates and Deep Supervision"""
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
        # Encoder
        e1, e2, e3, e4, e5 = self.encoder(x)

        # Decoder with skip connections
        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        # Final output
        d0 = self.final_upsample(d1)
        final = self.final_conv(d0)

        # Deep supervision (only during training)
        if self.deep_supervision and self.training:
            aux3 = self.aux_head3(d4)
            aux2 = self.aux_head2(d3)
            aux1 = self.aux_head1(d2)
            return final, [aux3, aux2, aux1]
        else:
            return final


# ============================================================================
# ResNet-50 Classifier (6-Channel Input)
# ============================================================================

class ResNet50_6Channel(nn.Module):
    """ResNet-50 modified for 6-channel input (RGB + 3-channel mask)"""

    def __init__(self, pretrained=True):
        super(ResNet50_6Channel, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Modify first conv layer for 6-channel input
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize new conv layer weights
        if pretrained:
            # Copy RGB weights and initialize mask weights
            with torch.no_grad():
                self.conv1.weight[:, :3, :, :] = resnet.conv1.weight
                self.conv1.weight[:, 3:, :, :] = resnet.conv1.weight.mean(dim=1, keepdim=True)

        # Copy remaining ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Feature dimension
        self.num_features = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


# ============================================================================
# Model 0: U-Net V6 + ResNet50 Baseline
# ============================================================================

class Model0_UNet_V6_ResNet50(nn.Module):
    """
    Model 0: Simple U-Net V6 + ResNet50 baseline

    Pipeline:
    1. Generate masks with U-Net V6 (frozen, 95.98% Dice)
    2. Concatenate RGB image + mask → 6 channels
    3. Extract features with ResNet50 (6-channel input)
    4. Classification head with dropout
    """
    def __init__(self, unet_weights_path, dropout_rates=[0.5, 0.3]):
        super(Model0_UNet_V6_ResNet50, self).__init__()

        # 1. U-Net V6 (frozen)
        self.unet = AttentionUNet_V6(n_classes=3, use_attention=True, deep_supervision=True)
        checkpoint = torch.load(unet_weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.unet.load_state_dict(checkpoint)
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        # 2. ResNet50 feature extractor (6 channels)
        self.resnet = ResNet50_6Channel(pretrained=True)
        resnet_features = self.resnet.num_features  # 2048 for ResNet50

        # 3. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(128, 1)
        )

    def forward(self, image):
        """
        Args:
            image: (B, 3, 512, 512) RGB fundus image

        Returns:
            logits: (B, 1) glaucoma classification logits
        """
        B = image.size(0)

        # 1. Generate masks with U-Net V6 (frozen)
        with torch.no_grad():
            mask_output = self.unet(image)
            # U-Net may return tuple (final, aux) during training, but we're in eval mode
            if isinstance(mask_output, tuple):
                mask = mask_output[0]
            else:
                mask = mask_output
            mask = torch.sigmoid(mask)  # (B, 3, 512, 512)

        # 2. Concatenate image + mask → 6 channels
        combined = torch.cat([image, mask], dim=1)  # (B, 6, 512, 512)

        # Resize to ResNet50 input size (256×256)
        combined = F.interpolate(combined, size=(256, 256), mode='bilinear', align_corners=False)

        # 3. Extract ResNet50 features
        features = self.resnet(combined)  # (B, 2048)

        # 4. Classification
        logits = self.classifier(features)  # (B, 1)

        return logits


# ============================================================================
# Dataset
# ============================================================================

class GlaucomaDataset(Dataset):
    """Glaucoma fundus image dataset"""

    def __init__(self, rg_dir, nrg_dir, transform=None, augment=False):
        self.rg_images = [os.path.join(rg_dir, f) for f in os.listdir(rg_dir) if f.endswith(('.jpg', '.png'))]
        self.nrg_images = [os.path.join(nrg_dir, f) for f in os.listdir(nrg_dir) if f.endswith(('.jpg', '.png'))]

        self.images = self.rg_images + self.nrg_images
        self.labels = [1] * len(self.rg_images) + [0] * len(self.nrg_images)

        self.transform = transform
        self.augment = augment

        print(f"Dataset: {len(self.images)} images ({len(self.rg_images)} RG, {len(self.nrg_images)} NRG)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 512x512
        image = cv2.resize(image, (Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE))

        # Augmentation
        if self.augment:
            image = self.apply_augmentation(image)

        # Convert to tensor and normalize
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def apply_augmentation(self, image):
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)

        # Random vertical flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)

        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.randint(-15, 15)
            center = (Config.UNET_INPUT_SIZE // 2, Config.UNET_INPUT_SIZE // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE))

        # Random brightness
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

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

    # Training dataset (with augmentation)
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
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    return train_loader, val_loader, test_loader


def calculate_metrics(y_true, y_pred, y_scores):
    """Calculate classification metrics"""

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    }

    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = metrics['recall']

    return metrics


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    pbar = tqdm(train_loader, desc='Training')
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

        # Get predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.cpu().detach().numpy())

        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )
    metrics['loss'] = epoch_loss

    return metrics


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
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
            all_scores.extend(probs.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )
    metrics['loss'] = epoch_loss

    return metrics


def train_model(model, train_loader, val_loader, device):
    """Complete training loop with early stopping"""

    print("\n" + "="*80)
    print("Starting Model 0 Training: U-Net V6 + ResNet50 Baseline")
    print("="*80)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    best_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Results ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  Val Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Check for improvement
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ Best model saved! (AUC: {best_auc:.4f})")

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
                print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                break

        # GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  GPU Memory: {memory_used:.2f} GB")

    return history, best_model_state


def test_model(model, test_loader, device):
    """Test the model"""
    print("\n" + "="*80)
    print("Testing Model 0")
    print("="*80)

    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []
    test_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            test_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )

    print("\nTest Set Results:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f} ({test_metrics['sensitivity']*100:.2f}%)")
    print(f"  Specificity: {test_metrics['specificity']:.4f} ({test_metrics['specificity']*100:.2f}%)")
    print(f"  Precision:   {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"  F1-Score:    {test_metrics['f1']:.4f}")
    print(f"  AUC:         {test_metrics['auc']:.4f}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten()
    ).ravel()

    print("\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")

    test_metrics['loss'] = test_loss
    test_metrics['confusion_matrix'] = [[int(tn), int(fp)], [int(fn), int(tp)]]

    return test_metrics


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training pipeline"""

    print("\n" + "="*80)
    print("Model 0: U-Net V6 + ResNet50 Baseline (No Enhancements)")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Data Source: {Config.DATA_ROOT}/")
    print(f"U-Net: V6 (95.98% Dice)")
    print(f"Classifier: ResNet-50 (6-channel input)")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print("="*80)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders()

    # Create model
    print("\nCreating Model 0...")
    model = Model0_UNet_V6_ResNet50(
        unet_weights_path=Config.UNET_WEIGHTS,
        dropout_rates=Config.DROPOUT_RATES
    )
    model = model.to(Config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (U-Net V6): {frozen_params:,}")

    # Train
    history, best_model_state = train_model(model, train_loader, val_loader, Config.DEVICE)

    # Test
    print("\nLoading best model for testing...")
    model.load_state_dict(best_model_state)
    test_metrics = test_model(model, test_loader, Config.DEVICE)

    # Save results
    results = {
        'model': 'Model 0: U-Net V6 + ResNet50 Baseline',
        'architecture': {
            'segmentation': 'U-Net V6 (frozen, 95.98% Dice)',
            'classifier': 'ResNet-50 (6-channel input)',
            'approach': 'Pipeline: Segmentation → Classification',
            'enhancements': 'None (baseline)'
        },
        'data_source': Config.DATA_ROOT,
        'training': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'epochs_trained': len(history['train_loss']),
            'best_val_auc': max(history['val_auc'])
        },
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'sensitivity': float(test_metrics['sensitivity']),
            'specificity': float(test_metrics['specificity']),
            'precision': float(test_metrics['precision']),
            'f1': float(test_metrics['f1']),
            'auc': float(test_metrics['auc']),
            'loss': float(test_metrics['loss']),
            'confusion_matrix': test_metrics['confusion_matrix']
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'train_auc': [float(x) for x in history['train_auc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']],
            'val_auc': [float(x) for x in history['val_auc']]
        },
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    }

    with open(Config.RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {Config.RESULTS_PATH}")
    print(f"✓ Model saved to {Config.MODEL_SAVE_PATH}")

    print("\n" + "="*80)
    print("Model 0 Training Complete!")
    print("="*80)
    print(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
