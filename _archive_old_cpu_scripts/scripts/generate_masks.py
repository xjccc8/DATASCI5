"""
Mask Generation Script for Glaucoma Detection
Generates optic disc and cup segmentation masks for all images
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
warnings.filterwarnings('ignore')

print("="*80)
print("MASK GENERATION PIPELINE")
print("="*80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nInitializing...")

# ==============================================================================
# U-NET ARCHITECTURE FOR SEGMENTATION
# ==============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetSegmentation(nn.Module):
    """U-Net for optic disc/cup segmentation"""

    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

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

        return self.outc(x)


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class FundusSegmentationDataset(Dataset):
    """Dataset for fundus image segmentation"""

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Get image files
        self.image_files = []
        for ext in ['.jpg', '.png', '.jpeg']:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (if exists)
        mask_path = self.masks_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Resize
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask, str(img_path)


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for segmentation"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_segmentation_model(model, train_loader, num_epochs=50, lr=0.001, device='cuda'):
    """Train U-Net segmentation model"""
    print("\n" + "="*80)
    print("TRAINING SEGMENTATION MODEL")
    print("="*80)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = dice_loss

    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_segmentation_model.pth')
            print(f'  [OK] Saved best model (loss: {best_loss:.4f})')

    print(f"\n[OK] Training complete! Best loss: {best_loss:.4f}")
    return model


# ==============================================================================
# MASK GENERATION
# ==============================================================================

def generate_masks_for_directory(model, input_dir, output_dir, device='cuda'):
    """Generate masks for all images in directory"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = []
    for ext in ['.jpg', '.png', '.jpeg']:
        image_files.extend(list(input_path.glob(f'*{ext}')))

    if len(image_files) == 0:
        print(f"  [SKIP] No images found in {input_dir}")
        return 0

    model = model.to(device)
    model.eval()

    generated_count = 0

    with torch.no_grad():
        for img_path in tqdm(image_files, desc=f"Generating masks for {input_dir}"):
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                original_size = image.shape[:2]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (512, 512))
                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float().unsqueeze(0).to(device)

                # Generate mask
                output = model(image)
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                # Resize back to original size
                pred_mask = cv2.resize(pred_mask.astype(np.uint8),
                                      (original_size[1], original_size[0]),
                                      interpolation=cv2.INTER_NEAREST)

                # Save mask
                mask_path = output_path / img_path.name
                cv2.imwrite(str(mask_path), pred_mask)

                generated_count += 1

            except Exception as e:
                print(f"  [ERROR] Failed to process {img_path.name}: {e}")
                continue

    return generated_count


def generate_all_masks(model, device='cuda'):
    """Generate masks for all datasets"""
    print("\n" + "="*80)
    print("GENERATING MASKS FOR ALL IMAGES")
    print("="*80)

    directories_to_process = [
        # EyePACS
        ('train/eyepac/NRG', 'train/eyepac/NRG_masks'),
        ('train/eyepac/RG', 'train/eyepac/RG_masks'),
        ('validation/eyepac/NRG', 'validation/eyepac/NRG_masks'),
        ('validation/eyepac/RG', 'validation/eyepac/RG_masks'),
        ('test/eyepac/NRG', 'test/eyepac/NRG_masks'),
        ('test/eyepac/RG', 'test/eyepac/RG_masks'),

        # REFUGE2 (train and test - validation already has masks)
        ('train/refuge2/images', 'train/refuge2/generated_masks'),
        ('test/refuge2/images', 'test/refuge2/generated_masks'),
    ]

    total_generated = 0
    stats = {}

    for input_dir, output_dir in directories_to_process:
        print(f"\nProcessing: {input_dir}")
        count = generate_masks_for_directory(model, input_dir, output_dir, device)
        stats[input_dir] = count
        total_generated += count
        print(f"  [OK] Generated {count} masks")

    print("\n" + "="*80)
    print(f"TOTAL MASKS GENERATED: {total_generated}")
    print("="*80)

    # Save statistics
    with open('mask_generation_stats.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_generated': total_generated,
            'details': stats
        }, f, indent=2)

    print("\n[OK] Statistics saved to: mask_generation_stats.json")

    return total_generated


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cpu':
        print("[WARNING] Running on CPU - this will be slow!")
        print("Consider using GPU for faster processing")

    # Check if training data exists
    refuge_val_images = Path('validation/refuge2/images')
    refuge_val_masks = Path('validation/refuge2/mask')

    if not refuge_val_images.exists() or not refuge_val_masks.exists():
        print("\n[ERROR] REFUGE2 validation data not found!")
        print("Expected:")
        print("  - validation/refuge2/images/")
        print("  - validation/refuge2/mask/")
        return

    # Step 1: Create dataset and dataloader
    print("\n" + "="*80)
    print("STEP 1: Loading REFUGE2 Validation Dataset")
    print("="*80)

    dataset = FundusSegmentationDataset(
        images_dir=refuge_val_images,
        masks_dir=refuge_val_masks
    )

    print(f"Found {len(dataset)} images with masks")

    if len(dataset) == 0:
        print("[ERROR] No training data found!")
        return

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )

    # Step 2: Train model
    print("\n" + "="*80)
    print("STEP 2: Training Segmentation Model")
    print("="*80)
    print("This may take 30-60 minutes depending on your hardware...")

    model = UNetSegmentation(n_channels=3, n_classes=3)
    model = train_segmentation_model(
        model,
        train_loader,
        num_epochs=30,  # Reduced for faster training
        lr=0.001,
        device=device
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_segmentation_model.pth'))
    print("[OK] Best model loaded")

    # Step 3: Generate masks for all images
    total_generated = generate_all_masks(model, device)

    # Step 4: Summary
    print("\n" + "="*80)
    print("MASK GENERATION COMPLETE!")
    print("="*80)
    print(f"\nTotal masks generated: {total_generated}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated directories:")
    print("  - train/eyepac/NRG_masks/")
    print("  - train/eyepac/RG_masks/")
    print("  - validation/eyepac/NRG_masks/")
    print("  - validation/eyepac/RG_masks/")
    print("  - test/eyepac/NRG_masks/")
    print("  - test/eyepac/RG_masks/")
    print("  - train/refuge2/generated_masks/")
    print("  - test/refuge2/generated_masks/")
    print("\nNext steps:")
    print("  1. Verify mask quality (check sample masks)")
    print("  2. Use masks for attention-guided learning")
    print("  3. Implement multi-task learning")
    print("  4. Calculate CDR features")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Mask generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
