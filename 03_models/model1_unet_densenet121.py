"""
Model 1: U-Net Segmentation + DenseNet-121 Classifier
Glaucoma Classification from Fundus Images

Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: U-Net (Pretrained from V5.1)
  Input: Fundus image (variable size)
  Output: Segmentation mask (512×512, 3 classes)

Stage 2: DenseNet-121 Classifier
  Input: Segmentation mask (resized to 224×224)
  Output: Glaucoma probability [0, 1]

Training Data:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- EyePACS: 10,340 images with generated masks
- Labels: RG (glaucoma) vs NRG (non-glaucoma)
- Split: 80% train, 10% val, 10% test

Expected Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Accuracy: 85-88%
- Sensitivity: 83-86%
- Specificity: 86-89%
- AUC: 0.90-0.93
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL 1: U-NET + DENSENET-121 GLAUCOMA CLASSIFIER")
print("="*80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nModel Architecture:")
print("  Stage 1: U-Net Segmentation (pretrained from V5.1)")
print("  Stage 2: DenseNet-121 Classification")
print("\nExpected: 85-88% accuracy")
print("="*80)

# Setup PyTorch
def setup_pytorch():
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available!")
        sys.exit(1)

    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)

    print(f"\n[OK] PyTorch Setup Complete")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    return device

device = setup_pytorch()


# ==============================================================================
# STAGE 1: U-NET SEGMENTATION (LOAD PRETRAINED)
# ==============================================================================

class ResNetEncoder(nn.Module):
    """ResNet34 encoder (same as V5.1)"""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet34(pretrained=True)

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


class UNet(nn.Module):
    """U-Net segmentation model (same as V5.1)"""
    def __init__(self, n_classes=3):
        super().__init__()
        self.encoder = ResNetEncoder()

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

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

        return self.outc(x)


# ==============================================================================
# STAGE 2: DENSENET-121 CLASSIFIER
# ==============================================================================

class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 classifier for glaucoma detection

    Input: Segmentation mask (224×224×3)
    Output: Glaucoma probability [0, 1]
    """
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        import torchvision.models as models

        # Load pretrained DenseNet-121
        self.densenet = models.densenet121(pretrained=pretrained)

        # Get number of features from last layer
        num_features = self.densenet.classifier.in_features

        # Replace classifier
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        logits = self.densenet(x)
        return logits.squeeze(1)  # (batch,)


# ==============================================================================
# COMPLETE MODEL 1: UNET + DENSENET-121
# ==============================================================================

class Model1_UNet_DenseNet(nn.Module):
    """
    Complete Model 1: Two-stage classification

    Stage 1: U-Net generates segmentation mask
    Stage 2: DenseNet-121 classifies from mask
    """
    def __init__(self, unet_weights_path=None, freeze_unet=True):
        super().__init__()

        # Stage 1: U-Net segmentation
        self.unet = UNet(n_classes=3)

        # Load pretrained U-Net weights from V5.1
        if unet_weights_path and os.path.exists(unet_weights_path):
            checkpoint = torch.load(unet_weights_path, map_location='cpu')
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Loaded U-Net weights from {unet_weights_path}")
        else:
            print(f"[WARNING] U-Net weights not found at {unet_weights_path}")
            print("         U-Net will be randomly initialized!")

        # Freeze U-Net if specified (usually True for classification)
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()
            print("[OK] U-Net frozen (inference mode)")

        # Stage 2: DenseNet-121 classifier
        self.classifier = DenseNet121Classifier(pretrained=True, num_classes=1)

    def forward(self, x, return_mask=False):
        """
        Forward pass

        Args:
            x: Input fundus image (batch, 3, H, W)
            return_mask: If True, return (prediction, mask)

        Returns:
            prediction: Glaucoma logit (batch,)
            mask (optional): Segmentation mask (batch, 3, 512, 512)
        """
        # Stage 1: Generate segmentation mask
        with torch.no_grad() if not self.training else torch.enable_grad():
            mask_logits = self.unet(x)  # (batch, 3, 512, 512)
            mask = F.softmax(mask_logits, dim=1)  # (batch, 3, 512, 512)

        # Resize mask to 224×224 for DenseNet-121
        mask_resized = F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)

        # Stage 2: Classify from mask
        prediction = self.classifier(mask_resized)  # (batch,)

        if return_mask:
            return prediction, mask
        return prediction


# ==============================================================================
# DATASET FOR MODEL 1
# ==============================================================================

class GlaucomaDataset(Dataset):
    """
    Dataset for glaucoma classification

    Loads fundus images and labels (RG vs NRG)
    """
    def __init__(self, data_dirs, transform=None):
        """
        Args:
            data_dirs: List of (image_dir, label) tuples
                       e.g., [('train/eyepac/RG', 1), ('train/eyepac/NRG', 0)]
            transform: Optional transforms
        """
        self.samples = []
        self.transform = transform

        for img_dir, label in data_dirs:
            img_path = Path(img_dir)
            if not img_path.exists():
                print(f"[WARNING] Directory not found: {img_dir}")
                continue

            # Get all images
            image_files = []
            for ext in ['.jpg', '.png', '.jpeg']:
                image_files.extend(list(img_path.glob(f'*{ext}')))

            for img_file in image_files:
                self.samples.append((str(img_file), label))

            print(f"  Loaded {len(image_files)} images from {img_dir} (label={label})")

        print(f"\n[OK] Total samples: {len(self.samples)}")

        # Calculate class distribution
        labels = [label for _, label in self.samples]
        num_glaucoma = sum(labels)
        num_normal = len(labels) - num_glaucoma
        print(f"  Glaucoma (RG): {num_glaucoma} ({num_glaucoma/len(labels)*100:.1f}%)")
        print(f"  Normal (NRG): {num_normal} ({num_normal/len(labels)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 512×512 (U-Net input)
        image = cv2.resize(image, (512, 512))

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: normalize to [0, 1] and convert to tensor
            image = image.astype(np.float32) / 255.0
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = torch.from_numpy(image).float()

        label = torch.tensor(label, dtype=torch.float32)

        return image, label


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()

    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Metrics
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

    avg_loss = total_loss / len(val_loader)

    # Calculate metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
    }

    return metrics


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Complete training loop"""
    print("\n" + "="*80)
    print("TRAINING MODEL 1: U-NET + DENSENET-121")
    print("="*80)

    model = model.to(device)

    # Loss function (Binary Cross Entropy)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer (only train classifier, U-Net is frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_auc = 0
    best_epoch = 0
    patience = 10
    no_improve_count = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'learning_rate': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_metrics['auc'])

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Print metrics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val AUC:    {val_metrics['auc']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch + 1
            no_improve_count = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'history': history
            }, 'model1_best.pth')

            print(f"  [OK] Best model saved! (AUC: {best_auc:.4f})")
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"\n[EARLY STOPPING] No improvement for {patience} epochs")
            break

        # GPU memory stats
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"\n[OK] Training complete!")
    print(f"  Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    return history


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main training pipeline"""

    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    # Define data directories
    train_dirs = [
        ('train/eyepac/RG', 1),      # Glaucoma
        ('train/eyepac/NRG', 0),     # Normal
    ]

    val_dirs = [
        ('validation/eyepac/RG', 1),
        ('validation/eyepac/NRG', 0),
    ]

    test_dirs = [
        ('test/eyepac/RG', 1),
        ('test/eyepac/NRG', 0),
    ]

    # Create datasets
    print("\nTrain Dataset:")
    train_dataset = GlaucomaDataset(train_dirs)

    print("\nValidation Dataset:")
    val_dataset = GlaucomaDataset(val_dirs)

    print("\nTest Dataset:")
    test_dataset = GlaucomaDataset(test_dirs)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\n[OK] Data loaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "="*80)
    print("STEP 2: CREATING MODEL 1")
    print("="*80)

    # Path to pretrained U-Net from V5.1
    unet_weights = 'best_segmentation_model_gpu_v51.pth'

    model = Model1_UNet_DenseNet(
        unet_weights_path=unet_weights,
        freeze_unet=True  # Freeze U-Net, only train DenseNet-121
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[OK] Model 1 created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # Train model
    print("\n" + "="*80)
    print("STEP 3: TRAINING")
    print("="*80)

    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=50,
        device=device
    )

    # Evaluate on test set
    print("\n" + "="*80)
    print("STEP 4: FINAL EVALUATION ON TEST SET")
    print("="*80)

    # Load best model
    checkpoint = torch.load('model1_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.BCEWithLogitsLoss()
    test_metrics = validate(model, test_loader, criterion, device)

    print("\nFinal Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity): {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")

    # Save results
    results = {
        'model': 'Model 1: U-Net + DenseNet-121',
        'timestamp': datetime.now().isoformat(),
        'test_metrics': test_metrics,
        'training_history': history,
        'best_epoch': checkpoint['epoch'] + 1
    }

    with open('model1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n[OK] Results saved to model1_results.json")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("MODEL 1 TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
