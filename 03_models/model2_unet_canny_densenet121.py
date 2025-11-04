"""
Model 2: U-Net + Canny Edge Detection + DenseNet-121 Glaucoma Classifier
================================================================================

Architecture:
    Fundus Image (512×512)
    → U-Net (frozen, V5.1 weights)
    → Segmentation Mask (512×512×3)
    → Canny Edge Detection
    → Concatenate [Mask + Edges] (512×512×6)
    → Resize to 224×224×6
    → DenseNet-121 (modified for 6 channels)
    → Glaucoma Probability [0, 1]

Key Features:
- Uses V5.1 pretrained U-Net (62.5% Dice) - FROZEN
- On-the-fly mask generation + Canny edge detection
- DenseNet-121 classifier with 6-channel input (mask + edges)
- Data from processed_gpu/ folders ONLY
- Mixed precision training (FP16)
- Expected: 89-91% accuracy, 0.92-0.94 AUC (+2% from Model 1 due to Canny)

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
    """Model 2 Configuration"""

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
    MODEL_SAVE_PATH = 'model2_best.pth'
    RESULTS_PATH = 'model2_results.json'

    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Architecture
    UNET_INPUT_SIZE = 512  # U-Net expects 512×512
    DENSENET_INPUT_SIZE = 224  # DenseNet-121 expects 224×224
    NUM_CLASSES = 1  # Binary classification

    # Canny edge detection parameters
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0  # Windows compatibility
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
# Canny Edge Detection Module
# ============================================================================

class CannyEdgeDetector(nn.Module):
    """
    Canny edge detection for segmentation masks

    Applies Canny edge detection to each channel of the segmentation mask
    to extract structural information about optic disc and cup boundaries.
    """

    def __init__(self, threshold1=50, threshold2=150):
        super(CannyEdgeDetector, self).__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def forward(self, mask):
        """
        Apply Canny edge detection to mask

        Args:
            mask: Segmentation mask tensor (B, 3, H, W) on GPU

        Returns:
            edges: Edge map tensor (B, 3, H, W) on GPU
        """
        batch_size, channels, height, width = mask.shape
        edges = torch.zeros_like(mask)

        # Process each image in batch
        for b in range(batch_size):
            # Process each channel separately
            for c in range(channels):
                # Convert to numpy for OpenCV
                channel = mask[b, c].cpu().numpy()
                channel_uint8 = (channel * 255).astype(np.uint8)

                # Apply Canny edge detection
                edge = cv2.Canny(
                    channel_uint8,
                    self.threshold1,
                    self.threshold2
                )

                # Normalize back to [0, 1] and convert to tensor
                edge_normalized = edge.astype(np.float32) / 255.0
                edges[b, c] = torch.from_numpy(edge_normalized)

        # Move back to GPU
        edges = edges.to(mask.device)
        return edges


# ============================================================================
# DenseNet-121 Classifier (Modified for 6 channels)
# ============================================================================

class DenseNet121Classifier_6Channel(nn.Module):
    """
    DenseNet-121 classifier modified to accept 6-channel input

    Input channels:
    - Channels 0-2: Segmentation mask (background, disc, cup)
    - Channels 3-5: Canny edges (background, disc, cup)
    """

    def __init__(self, num_classes=1, dropout_rates=[0.5, 0.3, 0.2]):
        super(DenseNet121Classifier_6Channel, self).__init__()

        # Load pretrained DenseNet-121
        densenet = models.densenet121(pretrained=True)

        # Modify first convolution to accept 6 channels
        original_conv = densenet.features.conv0
        self.features = densenet.features

        # Create new first conv layer (6 channels → 64 filters)
        self.features.conv0 = nn.Conv2d(
            in_channels=6,  # Modified for mask + edges
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize new conv weights
        # Copy pretrained weights for first 3 channels, duplicate for channels 4-6
        with torch.no_grad():
            # First 3 channels: use pretrained weights
            self.features.conv0.weight[:, :3, :, :] = original_conv.weight
            # Last 3 channels: duplicate pretrained weights
            self.features.conv0.weight[:, 3:, :, :] = original_conv.weight

        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        self.features.add_module('avgpool_final', nn.AdaptiveAvgPool2d((1, 1)))

        # Custom classification head
        num_features = densenet.classifier.in_features  # 1024
        self.classifier = nn.Sequential(
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
        Classify 6-channel input (mask + edges)

        Args:
            x: 6-channel tensor (B, 6, 224, 224)

        Returns:
            logits: Classification logits (B, 1)
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


# ============================================================================
# Model 2: U-Net + Canny + DenseNet-121 (Two-Stage with Edge Detection)
# ============================================================================

class Model2_UNet_Canny_DenseNet(nn.Module):
    """
    Model 2: Two-stage glaucoma classification with Canny edge detection

    Stage 1 (Frozen): U-Net generates segmentation mask
    Stage 2 (CPU): Canny edge detection extracts boundaries
    Stage 3 (Trainable): DenseNet-121 classifies mask + edges

    Architecture:
        Input (512×512×3)
        → U-Net [FROZEN]
        → Mask (512×512×3)
        → Canny Detection
        → Concatenate [Mask, Edges] (512×512×6)
        → Resize (224×224×6)
        → DenseNet-121 [TRAINABLE]
        → Logit (1)
    """

    def __init__(self, unet_weights_path=None, freeze_unet=True,
                 canny_threshold1=50, canny_threshold2=150):
        super(Model2_UNet_Canny_DenseNet, self).__init__()

        # Stage 1: U-Net for segmentation (frozen)
        self.unet = UNet_V51(weights_path=unet_weights_path)

        # Stage 2: Canny edge detection
        self.canny = CannyEdgeDetector(
            threshold1=canny_threshold1,
            threshold2=canny_threshold2
        )

        # Stage 3: DenseNet-121 for classification (trainable)
        self.classifier = DenseNet121Classifier_6Channel(num_classes=1)

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
            logits: Classification logits (B, 1)
            mask (optional): Segmentation mask (B, 3, 512, 512)
            edges (optional): Edge map (B, 3, 512, 512)
        """
        # Stage 1: Generate segmentation mask (frozen)
        with torch.no_grad():
            mask = self.unet.generate_mask(x)  # (B, 3, 512, 512)

        # Stage 2: Apply Canny edge detection
        edges = self.canny(mask)  # (B, 3, 512, 512)

        # Concatenate mask and edges
        mask_with_edges = torch.cat([mask, edges], dim=1)  # (B, 6, 512, 512)

        # Resize for DenseNet-121 (512×512 → 224×224)
        mask_with_edges_resized = F.interpolate(
            mask_with_edges,
            size=(Config.DENSENET_INPUT_SIZE, Config.DENSENET_INPUT_SIZE),
            mode='bilinear',
            align_corners=False
        )  # (B, 6, 224, 224)

        # Stage 3: Classify (trainable)
        logits = self.classifier(mask_with_edges_resized)  # (B, 1)

        if return_mask and return_edges:
            return logits, mask, edges
        elif return_mask:
            return logits, mask
        elif return_edges:
            return logits, edges
        return logits


# ============================================================================
# Dataset (Same as Model 1)
# ============================================================================

class GlaucomaDataset(Dataset):
    """Dataset for glaucoma classification from fundus images"""

    def __init__(self, rg_dir, nrg_dir, transform=None, augment=False):
        self.image_paths = []
        self.labels = []

        # Load RG (glaucoma)
        if os.path.exists(rg_dir):
            rg_files = [f for f in os.listdir(rg_dir) if f.endswith('.jpg')]
            self.image_paths.extend([os.path.join(rg_dir, f) for f in rg_files])
            self.labels.extend([1] * len(rg_files))

        # Load NRG (normal)
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
        # Load and preprocess image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (Config.UNET_INPUT_SIZE, Config.UNET_INPUT_SIZE))

        if self.augment:
            image = self._augment(image)

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    def _augment(self, image):
        """Simple data augmentation"""
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return image


# ============================================================================
# Training Functions (Same structure as Model 1)
# ============================================================================

def create_data_loaders():
    """Create train, validation, and test data loaders"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

    train_dataset = GlaucomaDataset(
        rg_dir=Config.TRAIN_RG, nrg_dir=Config.TRAIN_NRG,
        transform=transform, augment=True
    )

    val_dataset = GlaucomaDataset(
        rg_dir=Config.VAL_RG, nrg_dir=Config.VAL_NRG,
        transform=transform, augment=False
    )

    test_dataset = GlaucomaDataset(
        rg_dir=Config.TEST_RG, nrg_dir=Config.TEST_NRG,
        transform=transform, augment=False
    )

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

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = metrics['recall']

    return metrics


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    model.unet.eval()  # Keep U-Net frozen

    running_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    pbar = tqdm(train_loader, desc='Training', leave=False)
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
        all_scores.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )
    metrics['loss'] = epoch_loss

    return metrics


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()

    running_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
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

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )
    metrics['loss'] = epoch_loss

    return metrics


def train_model(model, train_loader, val_loader, device):
    """Complete training loop"""

    print("\n" + "="*80)
    print("Starting Model 2 Training: U-Net + Canny + DenseNet-121")
    print("="*80)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    scaler = GradScaler()

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

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Results ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  Val Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ Best model saved! (AUC: {best_auc:.4f})")

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

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                break

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  GPU Memory: {memory_used:.2f} GB")

    return history, best_model_state


def test_model(model, test_loader, device):
    """Evaluate model on test set"""

    print("\n" + "="*80)
    print("Testing Model 2 on Test Set")
    print("="*80)

    model.eval()

    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with autocast():
                logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

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

    tn, fp, fn, tp = confusion_matrix(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten()
    ).ravel()

    print("\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")

    return test_metrics


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training pipeline"""

    print("\n" + "="*80)
    print("Model 2: U-Net + Canny Edge Detection + DenseNet-121")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Data Source: {Config.DATA_ROOT}/")
    print(f"U-Net Weights: {Config.UNET_WEIGHTS}")
    print(f"Canny Thresholds: ({Config.CANNY_THRESHOLD1}, {Config.CANNY_THRESHOLD2})")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print("="*80)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders()

    # Create model
    print("\nCreating Model 2...")
    model = Model2_UNet_Canny_DenseNet(
        unet_weights_path=Config.UNET_WEIGHTS,
        freeze_unet=True,
        canny_threshold1=Config.CANNY_THRESHOLD1,
        canny_threshold2=Config.CANNY_THRESHOLD2
    )
    model = model.to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (U-Net): {total_params - trainable_params:,}")

    # Train
    history, best_model_state = train_model(model, train_loader, val_loader, Config.DEVICE)

    # Test
    print("\nLoading best model for testing...")
    model.load_state_dict(best_model_state)
    test_metrics = test_model(model, test_loader, Config.DEVICE)

    # Save results
    results = {
        'model': 'Model 2: U-Net + Canny + DenseNet-121',
        'architecture': {
            'stage1': 'U-Net (ResNet34 encoder, V5.1 weights, frozen)',
            'stage2': 'Canny Edge Detection (50, 150)',
            'stage3': 'DenseNet-121 (6-channel input, trainable)'
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

    print(f"\n✓ Results saved to {Config.RESULTS_PATH}")
    print(f"✓ Model saved to {Config.MODEL_SAVE_PATH}")

    print("\n" + "="*80)
    print("Model 2 Training Complete!")
    print("="*80)
    print(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
