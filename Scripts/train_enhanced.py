"""
Enhanced Segmentation Training Script for Duality AI Hackathon
Trains a segmentation head on top of DINOv2 backbone with:
- Data augmentation (critical for generalization to unseen desert)
- Class-weighted loss (handles class imbalance)
- Learning rate scheduling (CosineAnnealing)
- Mixed precision training (AMP) for GPU speedup
- Gradient accumulation for larger effective batch size
- Best model checkpointing
- Configurable backbone size (small/base/large)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
import argparse
import json
import time
from tqdm import tqdm
from collections import Counter

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Configuration
# ============================================================================

# Mapping from raw pixel values to new class IDs
# FIXED: Added class 600 (Flowers) that was missing in baseline!
VALUE_MAP = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers  ← WAS MISSING IN BASELINE
    700: 7,      # Logs
    800: 8,      # Rocks
    7100: 9,     # Landscape
    10000: 10    # Sky
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

N_CLASSES = len(VALUE_MAP)  # 11


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in VALUE_MAP.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Paired Augmentation (same spatial transform for image + mask)
# ============================================================================

class PairedTransform:
    """Apply identical spatial transforms to both image and mask."""
    
    def __init__(self, img_size, augment=True):
        self.img_size = img_size  # (h, w)
        self.augment = augment
        
        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, image, mask):
        # Resize both to target size
        image = TF.resize(image, self.img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=transforms.InterpolationMode.NEAREST)
        
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip (less common but useful)
            if random.random() > 0.8:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation (-15 to 15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)
            
            # Random color jitter (only image, NOT mask)
            if random.random() > 0.3:
                image = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                )(image)
            
            # Random Gaussian blur (simulates dust/atmosphere)
            if random.random() > 0.7:
                image = TF.gaussian_blur(image, kernel_size=5, sigma=(0.1, 2.0))
        
        # Convert to tensors
        image = TF.to_tensor(image)
        image = self.img_normalize(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, paired_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.paired_transform = paired_transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.paired_transform:
            image, mask = self.paired_transform(image, mask)
        
        return image, mask


# ============================================================================
# Model: Enhanced Segmentation Head
# ============================================================================

class SegmentationHead(nn.Module):
    """Enhanced segmentation head with deeper architecture and batch norm."""
    
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.decoder = nn.Sequential(
            # Stage 1: Project from embedding dim
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            # Stage 2: Spatial processing
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            # Stage 3: Depthwise separable conv
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Stage 4: Another spatial processing block
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.decoder(x)
        return self.classifier(x)


class SegmentationHeadConvNeXt(nn.Module):
    """Original ConvNeXt-style head (for compatibility with baseline)."""
    
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=N_CLASSES, ignore_index=255):
    """Compute IoU for each class and return mean IoU + per-class IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=N_CLASSES, smooth=1e-6):
    """Compute Dice coefficient per class."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=N_CLASSES):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            if labels.dim() == 4:
                labels = labels.squeeze(dim=1)
            labels = labels.long()

            iou, class_iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice, _ = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)

    model.train()
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    return np.nanmean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies), avg_class_iou


# ============================================================================
# Class Weight Computation
# ============================================================================

def compute_class_weights(data_dir, num_classes=N_CLASSES, max_samples=200):
    """Compute inverse-frequency class weights from training masks."""
    print("Computing class weights from training data...")
    masks_dir = os.path.join(data_dir, 'Segmentation')
    mask_files = sorted(os.listdir(masks_dir))
    
    # Sample a subset for speed
    if len(mask_files) > max_samples:
        mask_files = random.sample(mask_files, max_samples)
    
    pixel_counts = np.zeros(num_classes, dtype=np.float64)
    
    for mf in tqdm(mask_files, desc="Counting class pixels", leave=False):
        mask = Image.open(os.path.join(masks_dir, mf))
        mask_arr = np.array(convert_mask(mask))
        for c in range(num_classes):
            pixel_counts[c] += (mask_arr == c).sum()
    
    # Inverse frequency weighting with smoothing
    total = pixel_counts.sum()
    weights = total / (num_classes * pixel_counts + 1e-6)
    
    # Clip extreme weights
    weights = np.clip(weights, 0.1, 10.0)
    
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    
    print(f"Class weights: {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in weights]))}")
    return torch.FloatTensor(weights)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Loss vs Epoch', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_iou'], label='Val', linewidth=2)
    axes[0, 1].set_title('Mean IoU vs Epoch', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Dice
    axes[1, 0].plot(history['train_dice'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_dice'], label='Val', linewidth=2)
    axes[1, 0].set_title('Dice Score vs Epoch', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Pixel Accuracy
    axes[1, 1].plot(history['train_pixel_acc'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_pixel_acc'], label='Val', linewidth=2)
    axes[1, 1].set_title('Pixel Accuracy vs Epoch', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150)
    plt.close()

    # Per-class IoU bar chart
    if 'final_class_iou' in history:
        fig, ax = plt.subplots(figsize=(12, 6))
        class_iou = history['final_class_iou']
        valid_iou = [x if not np.isnan(x) else 0 for x in class_iou]
        colors = plt.cm.Set3(np.linspace(0, 1, N_CLASSES))
        bars = ax.bar(range(N_CLASSES), valid_iou, color=colors, edgecolor='black')
        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_ylabel('IoU')
        ax.set_title(f'Per-Class IoU (Mean: {np.nanmean(class_iou):.4f})', fontsize=14)
        ax.set_ylim(0, 1)
        ax.axhline(y=np.nanmean(class_iou), color='red', linestyle='--', label='Mean IoU')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, valid_iou):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
        plt.close()

    # Learning rate plot
    if 'lr' in history:
        plt.figure(figsize=(10, 4))
        plt.plot(history['lr'], linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_schedule.png'), dpi=150)
        plt.close()

    print(f"Saved all plots to '{output_dir}'")


def save_history_to_file(history, output_dir):
    """Save training history to text and JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON for programmatic access
    json_path = os.path.join(output_dir, 'training_history.json')
    json_history = {}
    for k, v in history.items():
        if isinstance(v, list):
            json_history[k] = [float(x) if not (isinstance(x, float) and np.isnan(x)) else None for x in v]
        else:
            json_history[k] = v
    with open(json_path, 'w') as f:
        json.dump(json_history, f, indent=2)
    
    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("ENHANCED TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Configuration:\n")
        if 'config' in history:
            for k, v in history['config'].items():
                f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Best Results:\n")
        best_val_iou_idx = int(np.argmax(history['val_iou']))
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {best_val_iou_idx + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {int(np.argmax(history['val_dice'])) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {int(np.argmax(history['val_pixel_acc'])) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {int(np.argmin(history['val_loss'])) + 1})\n")
        f.write("=" * 60 + "\n\n")

        if 'final_class_iou' in history:
            f.write("Per-Class IoU (Best Model):\n")
            f.write("-" * 40 + "\n")
            for name, iou in zip(CLASS_NAMES, history['final_class_iou']):
                iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A (not present)"
                f.write(f"  {name:<20}: {iou_str}\n")
            f.write("\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i], history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced Segmentation Training')
    parser.add_argument('--backbone', type=str, default='small', choices=['small', 'base', 'large'],
                        help='DINOv2 backbone size')
    parser.add_argument('--head', type=str, default='enhanced', choices=['enhanced', 'convnext'],
                        help='Segmentation head architecture')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--img_scale', type=float, default=0.5, help='Image scale factor (0.5 = half res)')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no_class_weights', action='store_true', help='Disable class-weighted loss')
    parser.add_argument('--data_dir', type=str, default=None, help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=None, help='Validation data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.data_dir is None:
        # Try multiple possible paths
        candidates = [
            os.path.join(script_dir, '..', 'Dataset', 'Offroad_Segmentation_Training_Dataset', 'train'),
            os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train'),
        ]
        for c in candidates:
            if os.path.isdir(c):
                args.data_dir = c
                break
        if args.data_dir is None:
            raise FileNotFoundError(f"Training data not found. Tried: {candidates}. Use --data_dir to specify.")
    
    if args.val_dir is None:
        args.val_dir = args.data_dir.replace('train', 'val')
    
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'runs', f'{args.backbone}_{args.head}_ep{args.epochs}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()  # Only use AMP with CUDA
    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")
    
    # Image dimensions (must be divisible by 14 for DINOv2 ViT)
    w = int(((960 * args.img_scale) // 14) * 14)
    h = int(((540 * args.img_scale) // 14) * 14)
    print(f"Image size: {w}x{h}")

    # Transforms
    train_transform = PairedTransform((h, w), augment=not args.no_augment)
    val_transform = PairedTransform((h, w), augment=False)

    # Datasets
    print(f"Loading training data from: {args.data_dir}")
    print(f"Loading validation data from: {args.val_dir}")
    
    trainset = MaskDataset(data_dir=args.data_dir, paired_transform=train_transform)
    valset = MaskDataset(data_dir=args.val_dir, paired_transform=val_transform)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Load DINOv2 backbone
    print(f"\nLoading DINOv2 backbone ({args.backbone})...")
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
    }
    backbone_arch = backbone_archs[args.backbone]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded!")

    # Get embedding dimension
    with torch.no_grad():
        sample_img = trainset[0][0].unsqueeze(0).to(device)
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
        n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens: {output.shape[1]} ({h//14}x{w//14})")

    # Create segmentation head
    HeadClass = SegmentationHead if args.head == 'enhanced' else SegmentationHeadConvNeXt
    classifier = HeadClass(
        in_channels=n_embedding,
        out_channels=N_CLASSES,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier = classifier.to(device)

    # Class-weighted loss
    if not args.no_class_weights:
        class_weights = compute_class_weights(args.data_dir, num_classes=N_CLASSES)
        class_weights = class_weights.to(device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fct = nn.CrossEntropyLoss()

    # Optimizer with weight decay
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'lr': [],
        'config': {
            'backbone': args.backbone,
            'head': args.head,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'img_size': f'{w}x{h}',
            'augmentation': not args.no_augment,
            'class_weights': not args.no_class_weights,
            'accum_steps': args.accum_steps,
            'n_classes': N_CLASSES,
        }
    }

    best_val_iou = 0
    best_epoch = 0

    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting training: {args.epochs} epochs, backbone={args.backbone}, head={args.head}")
    print(f"Effective batch size: {args.batch_size * args.accum_steps}")
    print(f"{'='*80}\n")

    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # ---- Training Phase ----
        classifier.train()
        train_losses = []
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for step, (imgs, labels) in enumerate(train_pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                features = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            # Forward pass (with or without AMP)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = classifier(features)
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                    loss = loss_fct(outputs, labels) / args.accum_steps
                scaler.scale(loss).backward()
            else:
                logits = classifier(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                loss = loss_fct(outputs, labels) / args.accum_steps
                loss.backward()

            # Gradient accumulation
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * args.accum_steps)
            train_pbar.set_postfix(loss=f"{train_losses[-1]:.4f}")

        # ---- Validation Phase ----
        classifier.eval()
        val_losses = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)

                features = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                loss = loss_fct(outputs, labels)
                val_losses.append(loss.item())

        # ---- Metrics ----
        train_iou, train_dice, train_pixel_acc, _ = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=N_CLASSES)
        val_iou, val_dice, val_pixel_acc, val_class_iou = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=N_CLASSES)

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(float(train_iou))
        history['val_iou'].append(float(val_iou))
        history['train_dice'].append(float(train_dice))
        history['val_dice'].append(float(val_dice))
        history['train_pixel_acc'].append(float(train_pixel_acc))
        history['val_pixel_acc'].append(float(val_pixel_acc))
        history['lr'].append(current_lr)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s) | "
              f"Loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f} | "
              f"IoU: {train_iou:.4f}/{val_iou:.4f} | "
              f"Dice: {train_dice:.4f}/{val_dice:.4f} | "
              f"Acc: {train_pixel_acc:.4f}/{val_pixel_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            best_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': history['config'],
            }, best_path)
            print(f"  ★ New best model! Val IoU: {val_iou:.4f} (saved to {best_path})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_ep{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)

    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"Training complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best Val IoU: {best_val_iou:.4f} (Epoch {best_epoch})")
    print(f"{'='*80}")

    # Final per-class IoU with best model
    best_ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device)
    classifier.load_state_dict(best_ckpt['model_state_dict'])
    _, _, _, final_class_iou = evaluate_metrics(classifier, backbone_model, val_loader, device, num_classes=N_CLASSES)
    history['final_class_iou'] = [float(x) for x in final_class_iou]

    print("\nPer-Class IoU (Best Model):")
    for name, iou in zip(CLASS_NAMES, final_class_iou):
        print(f"  {name:<20}: {iou:.4f}" if not np.isnan(iou) else f"  {name:<20}: N/A")

    # Save everything
    save_training_plots(history, args.output_dir)
    save_history_to_file(history, args.output_dir)

    # Also save model weights separately for easy loading in test script
    weights_path = os.path.join(args.output_dir, 'segmentation_head.pth')
    torch.save(classifier.state_dict(), weights_path)
    print(f"Saved final weights to '{weights_path}'")


if __name__ == "__main__":
    main()
