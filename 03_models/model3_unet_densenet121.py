"""
Model 3: U-Net V6 + DenseNet121 (No Canny)
================================================================================

Architecture:
    Fundus Image (512×512)
    → U-Net V6 (frozen, 95.98% Dice) with Attention Gates
    → Segmentation Mask (512×512×3)
    → DenseNet121 Classifier (6 channels: RGB + Mask)
    → Glaucoma Probability [0, 1]

Purpose:
- Compare DenseNet121 vs ResNet50 as backbone
- Baseline without Canny edge detection
- Direct comparison to Model 1 (U-Net + ResNet50)

Expected Performance:
- Accuracy: 85-87%
- AUC: 0.93-0.94

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


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Model 3 Configuration"""

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
    MODEL_SAVE_PATH = 'model3_best.pth'
    RESULTS_PATH = 'model3_results.json'

    # Training hyperparameters
    BATCH_SIZE = 24
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 15

    # Dropout rates
    DROPOUT_RATES = [0.5, 0.3]

    # Focal Loss parameters
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Architecture
    UNET_INPUT_SIZE = 512
    DENSENET_INPUT_SIZE = 224
    NUM_CLASSES = 1

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
# DenseNet121 6-Channel Classifier
# ============================================================================

class DenseNet121_6Channel(nn.Module):
    """DenseNet121 modified for 6-channel input (RGB + 3-channel mask)"""
    def __init__(self, pretrained=True):
        super(DenseNet121_6Channel, self).__init__()

        # Load pretrained DenseNet121
        densenet = models.densenet121(pretrained=pretrained)

        # Modify first conv layer: 3 channels → 6 channels
        original_conv = densenet.features.conv0
        self.conv0 = nn.Conv2d(
            6, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new conv layer
        with torch.no_grad():
            # Copy RGB weights
            self.conv0.weight[:, :3, :, :] = original_conv.weight
            # Initialize mask channels with small random values
            self.conv0.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) * 0.1

        # Copy rest of features
        self.features = nn.Sequential(
            self.conv0,
            *list(densenet.features.children())[1:]
        )

        # Get feature dimension
        self.num_features = densenet.classifier.in_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


# ============================================================================
# Model 3: U-Net V6 + DenseNet121
# ============================================================================

class Model3_UNet_DenseNet121(nn.Module):
    """
    Model 3: U-Net V6 Segmentation + DenseNet121 Classification

    Pipeline:
    1. Generate masks with frozen U-Net V6
    2. Concatenate RGB image + mask → 6 channels
    3. Extract features with DenseNet121
    4. Classification head with dropout
    """
    def __init__(self, unet_weights_path, dropout_rates=[0.5, 0.3]):
        super(Model3_UNet_DenseNet121, self).__init__()

        # 1. U-Net V6 (frozen)
        self.unet = AttentionUNet_V6(n_classes=3, use_attention=True, deep_supervision=True)
        checkpoint = torch.load(unet_weights_path, map_location='cpu')
        # Extract model state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.unet.load_state_dict(checkpoint)
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        # 2. DenseNet121 feature extractor (6 channels)
        self.densenet = DenseNet121_6Channel(pretrained=True)
        densenet_features = self.densenet.num_features  # 1024 for DenseNet121

        # 3. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(densenet_features, 512),
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
            image: (B, 3, 512, 512) RGB fundus images

        Returns:
            logits: (B, 1) glaucoma classification logits
        """
        B = image.size(0)

        # 1. Generate masks with U-Net V6 (frozen)
        with torch.no_grad():
            mask_output = self.unet(image)
            # U-Net may return tuple (final, aux) during training, but we're in eval mode
            if isinstance(mask_output, tuple):
                mask = mask_output[0]  # Get final output only
            else:
                mask = mask_output
            mask = torch.sigmoid(mask)  # (B, 3, 512, 512)

        # 2. Concatenate image + mask → 6 channels
        combined = torch.cat([image, mask], dim=1)  # (B, 6, 512, 512)

        # Resize to DenseNet input size (224×224)
        combined = F.interpolate(combined, size=(224, 224), mode='bilinear', align_corners=False)

        # 3. Extract DenseNet features
        features = self.densenet(combined)  # (B, 1024)

        # 4. Classification
        logits = self.classifier(features)  # (B, 1)

        return logits


# ============================================================================
# Dataset
# ============================================================================

class GlaucomaDataset(Dataset):
    """Dataset for glaucoma classification"""
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
        self.image_paths = self.rg_images + self.nrg_images
        self.labels = [1] * len(self.rg_images) + [0] * len(self.nrg_images)

        print(f"Loaded {len(self.rg_images)} RG, {len(self.nrg_images)} NRG images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
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

        # Metrics
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
    print("Model 3: U-Net V6 + DenseNet121 (No Canny)")
    print("=" * 80)

    # Device
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Data transforms (simplified for speed)
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
    print("\nInitializing Model 3...")
    model = Model3_UNet_DenseNet121(
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

        # Train
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_acc, val_auc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"[BEST] Saved model with Val AUC: {val_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        # Scheduler step
        scheduler.step()

    # Test evaluation
    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    test_loss, test_acc, test_auc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    # Compute additional metrics
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
        'model': 'Model 3: U-Net V6 + DenseNet121',
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
            'dropout_rates': Config.DROPOUT_RATES
        }
    }

    with open(Config.RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {Config.RESULTS_PATH}")
    print("Training complete!")


if __name__ == '__main__':
    main()
