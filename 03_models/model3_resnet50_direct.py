"""
Model 3: ResNet-50 Direct Classifier (Benchmark)
================================================================================

Architecture:
    Fundus Image (256×256)
    → ResNet-50 (pretrained on ImageNet)
    → Glaucoma Probability [0, 1]

Key Features:
- Direct end-to-end classification (no segmentation)
- ResNet-50 pretrained on ImageNet
- Benchmark approach (like Jisy Nj et al. 2024 - 90.91%)
- Input size: 256×256 (standard for ResNet-50)
- Data from processed_gpu/ folders ONLY
- Mixed precision training (FP16)
- Expected: 88-90% accuracy, 0.91-0.93 AUC

Purpose:
- Baseline comparison for U-Net segmentation approach
- Test if segmentation-based approach (Models 1-2) outperforms direct classification

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
    """Model 3 Configuration"""

    # Paths - ONLY use processed_gpu folders
    DATA_ROOT = 'processed_gpu'
    TRAIN_RG = os.path.join(DATA_ROOT, 'train', 'RG')
    TRAIN_NRG = os.path.join(DATA_ROOT, 'train', 'NRG')
    VAL_RG = os.path.join(DATA_ROOT, 'validation', 'RG')
    VAL_NRG = os.path.join(DATA_ROOT, 'validation', 'NRG')
    TEST_RG = os.path.join(DATA_ROOT, 'test', 'RG')
    TEST_NRG = os.path.join(DATA_ROOT, 'test', 'NRG')

    # Output paths
    MODEL_SAVE_PATH = 'model3_best.pth'
    RESULTS_PATH = 'model3_results.json'

    # Training hyperparameters
    BATCH_SIZE = 32  # Larger batch for direct classification
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Architecture
    INPUT_SIZE = 256  # Standard ResNet-50 input size
    NUM_CLASSES = 1  # Binary classification

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0  # Windows compatibility
    PIN_MEMORY = True

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# ResNet-50 Direct Classifier
# ============================================================================

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 for direct glaucoma classification

    Direct end-to-end classification from fundus images without
    intermediate segmentation step. This is the benchmark approach
    used by Jisy Nj et al. (2024) which achieved 90.91% accuracy.
    """

    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ResNet50Classifier, self).__init__()

        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=True)

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
        Direct classification from fundus image

        Args:
            x: Fundus image (B, 3, 256, 256)

        Returns:
            logits: Classification logits (B, 1)
        """
        return self.resnet(x)


# ============================================================================
# Dataset
# ============================================================================

class GlaucomaDataset(Dataset):
    """
    Dataset for glaucoma classification from fundus images

    Loads 256×256 RGB images for direct ResNet-50 classification.
    """

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
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 256×256 (ResNet-50 standard)
        image = cv2.resize(image, (Config.INPUT_SIZE, Config.INPUT_SIZE))

        # Data augmentation (only for training)
        if self.augment:
            image = self._augment(image)

        # Convert to PIL for transforms
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

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
    all_preds = []
    all_labels = []
    all_scores = []

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate epoch metrics
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
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Accumulate metrics
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate metrics
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
    print("Starting Model 3 Training: ResNet-50 Direct Classifier")
    print("="*80)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),  # Train all parameters
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize AUC
        factor=0.5,
        patience=3,
        verbose=True
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

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 80)

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
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
    """Evaluate model on test set"""

    print("\n" + "="*80)
    print("Testing Model 3 on Test Set")
    print("="*80)

    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            with autocast():
                logits = model(images)

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    # Calculate test metrics
    test_metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )

    # Print results
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

    return test_metrics


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training pipeline"""

    print("\n" + "="*80)
    print("Model 3: ResNet-50 Direct Classifier (Benchmark)")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Data Source: {Config.DATA_ROOT}/")
    print(f"Input Size: {Config.INPUT_SIZE}×{Config.INPUT_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print("="*80)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders()

    # Create model
    print("\nCreating Model 3...")
    model = ResNet50Classifier(num_classes=1, dropout_rate=0.5)
    model = model.to(Config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Train model
    history, best_model_state = train_model(model, train_loader, val_loader, Config.DEVICE)

    # Load best model for testing
    print("\nLoading best model for testing...")
    model.load_state_dict(best_model_state)

    # Test model
    test_metrics = test_model(model, test_loader, Config.DEVICE)

    # Save results
    results = {
        'model': 'Model 3: ResNet-50 Direct Classifier (Benchmark)',
        'architecture': {
            'backbone': 'ResNet-50 (ImageNet pretrained)',
            'classifier': 'Custom FC head with dropout',
            'approach': 'Direct end-to-end classification (no segmentation)'
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
    print("Model 3 Training Complete!")
    print("="*80)
    print(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print(f"Benchmark (Jisy Nj et al. 2024): 90.91% accuracy")
    print("="*80)


if __name__ == '__main__':
    main()
