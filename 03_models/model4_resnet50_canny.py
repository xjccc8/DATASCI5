"""
Model 4: ResNet-50 + Canny Edge Detection Direct Classifier
================================================================================

Architecture:
    Fundus Image (256×256)
    → Canny Edge Detection
    → Concatenate [Image + Edges] (256×256×6)
    → ResNet-50 (modified for 6 channels)
    → Glaucoma Probability [0, 1]

Key Features:
- Direct end-to-end classification with Canny edge enhancement
- ResNet-50 modified to accept 6-channel input (RGB + edges)
- Tests if Canny edges improve ResNet-50 benchmark performance
- Input size: 256×256
- Data from processed_gpu/ folders ONLY
- Mixed precision training (FP16)
- Expected: 90-92% accuracy, 0.93-0.95 AUC (highest of all 4 models)

Purpose:
- Test if Canny edge detection improves direct classification
- Compare with Model 3 (no Canny) to measure Canny benefit
- Potentially beat Jisy Nj et al. benchmark (90.91%)

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
    """Model 4 Configuration"""

    # Paths - ONLY use processed_gpu folders
    DATA_ROOT = 'processed_gpu'
    TRAIN_RG = os.path.join(DATA_ROOT, 'train', 'RG')
    TRAIN_NRG = os.path.join(DATA_ROOT, 'train', 'NRG')
    VAL_RG = os.path.join(DATA_ROOT, 'validation', 'RG')
    VAL_NRG = os.path.join(DATA_ROOT, 'validation', 'NRG')
    TEST_RG = os.path.join(DATA_ROOT, 'test', 'RG')
    TEST_NRG = os.path.join(DATA_ROOT, 'test', 'NRG')

    # Output paths
    MODEL_SAVE_PATH = 'model4_best.pth'
    RESULTS_PATH = 'model4_results.json'

    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Architecture
    INPUT_SIZE = 256  # Standard ResNet-50 input size
    NUM_CLASSES = 1  # Binary classification

    # Canny edge detection parameters
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0  # Windows compatibility
    PIN_MEMORY = True

    # ImageNet normalization (for RGB channels)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# ResNet-50 Classifier (Modified for 6 channels)
# ============================================================================

class ResNet50Classifier_6Channel(nn.Module):
    """
    ResNet-50 modified to accept 6-channel input

    Input channels:
    - Channels 0-2: RGB fundus image
    - Channels 3-5: Canny edges (R, G, B channels separately)
    """

    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ResNet50Classifier_6Channel, self).__init__()

        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=True)

        # Modify first convolution to accept 6 channels
        original_conv = self.resnet.conv1

        # Create new first conv layer
        self.resnet.conv1 = nn.Conv2d(
            in_channels=6,  # RGB + Canny edges
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize new conv weights
        with torch.no_grad():
            # First 3 channels: use pretrained weights (RGB)
            self.resnet.conv1.weight[:, :3, :, :] = original_conv.weight
            # Last 3 channels: duplicate pretrained weights (for edges)
            self.resnet.conv1.weight[:, 3:, :, :] = original_conv.weight

        # Replace final FC layer with custom classifier
        num_features = self.resnet.fc.in_features  # 2048

        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Direct classification from 6-channel input (image + edges)

        Args:
            x: 6-channel tensor (B, 6, 256, 256)

        Returns:
            logits: Classification logits (B, 1)
        """
        return self.resnet(x)


# ============================================================================
# Dataset with On-the-fly Canny Edge Detection
# ============================================================================

class GlaucomaDatasetWithCanny(Dataset):
    """
    Dataset with on-the-fly Canny edge detection

    Loads fundus images, applies Canny edge detection, and returns
    6-channel tensors (RGB + edges) for ResNet-50 classification.
    """

    def __init__(self, rg_dir, nrg_dir, transform=None, augment=False,
                 canny_threshold1=50, canny_threshold2=150):
        self.image_paths = []
        self.labels = []
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

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
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 256×256
        image = cv2.resize(image, (Config.INPUT_SIZE, Config.INPUT_SIZE))

        # Data augmentation (before edge detection)
        if self.augment:
            image = self._augment(image)

        # Apply Canny edge detection to each channel
        edges = self._apply_canny(image)

        # Convert to float32 and normalize [0, 1]
        image_float = image.astype(np.float32) / 255.0
        edges_float = edges.astype(np.float32) / 255.0

        # Concatenate: [R, G, B, Edge_R, Edge_G, Edge_B]
        image_with_edges = np.concatenate([image_float, edges_float], axis=2)  # (256, 256, 6)

        # Transpose to (C, H, W) for PyTorch
        image_with_edges = np.transpose(image_with_edges, (2, 0, 1))  # (6, 256, 256)

        # Convert to tensor
        image_tensor = torch.from_numpy(image_with_edges).float()

        # Apply normalization (only to first 3 channels - RGB)
        if self.transform:
            # Normalize RGB channels
            mean = torch.tensor(Config.IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(Config.IMAGENET_STD).view(3, 1, 1)
            image_tensor[:3, :, :] = (image_tensor[:3, :, :] - mean) / std
            # Edge channels (3-5) remain unnormalized [0, 1]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image_tensor, label

    def _apply_canny(self, image):
        """
        Apply Canny edge detection to each RGB channel separately

        Args:
            image: RGB image (H, W, 3) uint8

        Returns:
            edges: Edge map (H, W, 3) uint8
        """
        edges = np.zeros_like(image)

        for c in range(3):  # Process R, G, B separately
            channel = image[:, :, c]
            edge = cv2.Canny(
                channel,
                self.canny_threshold1,
                self.canny_threshold2
            )
            edges[:, :, c] = edge

        return edges

    def _augment(self, image):
        """Data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)

        # Random vertical flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)

        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.randint(-15, 15)
            center = (Config.INPUT_SIZE // 2, Config.INPUT_SIZE // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (Config.INPUT_SIZE, Config.INPUT_SIZE))

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

    # Note: Normalization handled in dataset __getitem__
    transform = True  # Flag to apply normalization

    # Training dataset (with augmentation)
    train_dataset = GlaucomaDatasetWithCanny(
        rg_dir=Config.TRAIN_RG,
        nrg_dir=Config.TRAIN_NRG,
        transform=transform,
        augment=True,
        canny_threshold1=Config.CANNY_THRESHOLD1,
        canny_threshold2=Config.CANNY_THRESHOLD2
    )

    # Validation dataset (no augmentation)
    val_dataset = GlaucomaDatasetWithCanny(
        rg_dir=Config.VAL_RG,
        nrg_dir=Config.VAL_NRG,
        transform=transform,
        augment=False,
        canny_threshold1=Config.CANNY_THRESHOLD1,
        canny_threshold2=Config.CANNY_THRESHOLD2
    )

    # Test dataset (no augmentation)
    test_dataset = GlaucomaDatasetWithCanny(
        rg_dir=Config.TEST_RG,
        nrg_dir=Config.TEST_NRG,
        transform=transform,
        augment=False,
        canny_threshold1=Config.CANNY_THRESHOLD1,
        canny_threshold2=Config.CANNY_THRESHOLD2
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


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""

    model.train()

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
    print("Starting Model 4 Training: ResNet-50 + Canny Edge Detection")
    print("="*80)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    print("Testing Model 4 on Test Set")
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
    print("Model 4: ResNet-50 + Canny Edge Detection")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Data Source: {Config.DATA_ROOT}/")
    print(f"Canny Thresholds: ({Config.CANNY_THRESHOLD1}, {Config.CANNY_THRESHOLD2})")
    print(f"Input: 6 channels (RGB + Canny edges)")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print("="*80)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders()

    # Create model
    print("\nCreating Model 4...")
    model = ResNet50Classifier_6Channel(num_classes=1, dropout_rate=0.5)
    model = model.to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Train
    history, best_model_state = train_model(model, train_loader, val_loader, Config.DEVICE)

    # Test
    print("\nLoading best model for testing...")
    model.load_state_dict(best_model_state)
    test_metrics = test_model(model, test_loader, Config.DEVICE)

    # Save results
    results = {
        'model': 'Model 4: ResNet-50 + Canny Edge Detection',
        'architecture': {
            'backbone': 'ResNet-50 (6-channel input, ImageNet pretrained)',
            'input': 'RGB fundus image + Canny edges (6 channels)',
            'classifier': 'Custom FC head with dropout',
            'approach': 'Direct classification with edge enhancement'
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
    print("Model 4 Training Complete!")
    print("="*80)
    print(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print(f"Benchmark (Jisy Nj et al. 2024): 90.91% accuracy")
    print("="*80)


if __name__ == '__main__':
    main()
