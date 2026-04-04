"""
Enhanced Segmentation Test/Inference Script for Duality AI Hackathon
Evaluates trained model on test images and generates:
- Predicted segmentation masks
- Colored visualization masks
- Side-by-side comparisons (if ground truth available)
- Per-class IoU analysis
- Confusion matrix
- Failure case analysis
- Inference timing
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import argparse
import json
import time
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Configuration (must match training)
# ============================================================================

VALUE_MAP = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers
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

# High-contrast color palette for visualization
COLOR_PALETTE = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [255, 105, 180],  # Flowers - hot pink
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


# ============================================================================
# Utility Functions
# ============================================================================

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in VALUE_MAP.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(N_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask


# ============================================================================
# Datasets
# ============================================================================

class TestDatasetWithGT(Dataset):
    """Dataset with both RGB images AND ground truth masks."""
    def __init__(self, data_dir, img_size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.img_size = img_size
        self.has_masks = os.path.isdir(self.masks_dir)
        
        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (W, H)
        
        # Resize and normalize
        image = TF.resize(image, self.img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        image = TF.to_tensor(image)
        image = self.img_normalize(image)
        
        # Load mask if available
        mask = None
        if self.has_masks:
            mask_path = os.path.join(self.masks_dir, data_id)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = convert_mask(mask)
                mask = TF.resize(mask, self.img_size, interpolation=transforms.InterpolationMode.NEAREST)
                mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask, data_id, original_size


# ============================================================================
# Models (must match training)
# ============================================================================

class SegmentationHead(nn.Module):
    """Enhanced segmentation head."""
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
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
    """Original ConvNeXt-style head."""
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

def compute_iou(pred, target, num_classes=N_CLASSES):
    """Compute IoU for each class."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_confusion_matrix(pred, target, num_classes=N_CLASSES):
    """Compute confusion matrix."""
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target, pred):
        if t < num_classes and p < num_classes:
            cm[t][p] += 1
    return cm


# ============================================================================
# Visualization
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id, class_names=CLASS_NAMES):
    """Save side-by-side comparison."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))
    
    if gt_mask is not None:
        gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=12)
        axes[0].axis('off')
        axes[1].imshow(gt_color)
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].axis('off')
        axes[2].imshow(pred_color)
        axes[2].set_title('Prediction', fontsize=12)
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=12)
        axes[0].axis('off')
        axes[1].imshow(pred_color)
        axes[1].set_title('Prediction', fontsize=12)
        axes[1].axis('off')

    plt.suptitle(f'{data_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix_plot(cm, output_path, class_names=CLASS_NAMES):
    """Save confusion matrix as heatmap."""
    # Normalize by row (true class)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Enhanced Segmentation Test/Inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model weights (best_model.pth or segmentation_head.pth)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to test dataset (must contain Color_Images/ folder)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(script_dir, 'predictions'),
                        help='Directory to save predictions')
    parser.add_argument('--backbone', type=str, default='small', choices=['small', 'base', 'large'],
                        help='DINOv2 backbone size (must match training)')
    parser.add_argument('--head', type=str, default='enhanced', choices=['enhanced', 'convnext'],
                        help='Segmentation head type (must match training)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--img_scale', type=float, default=0.5, help='Image scale (must match training)')
    parser.add_argument('--num_vis', type=int, default=20, help='Number of comparison visualizations')
    args = parser.parse_args()

    # Auto-detect paths
    if args.data_dir is None:
        candidates = [
            os.path.join(script_dir, '..', 'TestImages', 'Offroad_Segmentation_testImages'),
            os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'),
        ]
        for c in candidates:
            if os.path.isdir(c):
                args.data_dir = c
                break
        if args.data_dir is None:
            raise FileNotFoundError(f"Test data not found. Tried: {candidates}. Use --data_dir.")
    
    if args.model_path is None:
        # Look for best model in runs/ directory
        candidates = [
            os.path.join(script_dir, 'runs', f'{args.backbone}_{args.head}_ep30', 'best_model.pth'),
            os.path.join(script_dir, 'runs', f'{args.backbone}_{args.head}_ep30', 'segmentation_head.pth'),
            os.path.join(script_dir, 'segmentation_head.pth'),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.model_path = c
                break
        if args.model_path is None:
            raise FileNotFoundError(f"Model not found. Tried: {candidates}. Use --model_path.")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Image dimensions
    w = int(((960 * args.img_scale) // 14) * 14)
    h = int(((540 * args.img_scale) // 14) * 14)
    print(f"Image size: {w}x{h}")

    # Dataset
    print(f"Loading test data from: {args.data_dir}")
    dataset = TestDatasetWithGT(args.data_dir, img_size=(h, w))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")
    has_gt = dataset.has_masks
    print(f"Ground truth available: {has_gt}")

    # Load backbone
    print(f"\nLoading DINOv2 backbone ({args.backbone})...")
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
    }
    backbone_name = f"dinov2_{backbone_archs[args.backbone]}"
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)

    # Get embedding dim
    sample_img = dataset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]

    # Load classifier
    print(f"Loading model from {args.model_path}...")
    HeadClass = SegmentationHead if args.head == 'enhanced' else SegmentationHeadConvNeXt
    classifier = HeadClass(
        in_channels=n_embedding,
        out_channels=N_CLASSES,
        tokenW=w // 14,
        tokenH=h // 14
    )
    
    # Handle both checkpoint formats
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', '?')}, val_iou={checkpoint.get('val_iou', '?')})")
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded!")

    # Output subdirectories
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run inference
    print(f"\nRunning inference on {len(dataset)} images...")
    
    all_iou = []
    all_class_iou = []
    confusion_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    inference_times = []
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (imgs, masks, data_ids, orig_sizes) in enumerate(tqdm(loader, desc="Inference")):
            imgs = imgs.to(device)
            
            # Time inference
            t_start = time.time()
            features = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(features)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            predicted = torch.argmax(outputs, dim=1)
            t_end = time.time()
            
            inference_times.append((t_end - t_start) / imgs.shape[0])  # per image

            # Compute metrics if GT available
            if has_gt and masks[0] is not None:
                gt_masks = torch.stack([m for m in masks]).to(device)
                iou, class_iou = compute_iou(outputs, gt_masks, num_classes=N_CLASSES)
                all_iou.append(iou)
                all_class_iou.append(class_iou)
                
                # Accumulate confusion matrix
                cm = compute_confusion_matrix(predicted, gt_masks, num_classes=N_CLASSES)
                confusion_matrix += cm

            # Save predictions for every image
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                pred_mask = predicted[i].cpu().numpy().astype(np.uint8)
                
                # Save raw mask
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Save colored mask
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save comparison visualization
                if vis_count < args.num_vis:
                    gt = masks[i] if has_gt and masks[i] is not None else None
                    save_prediction_comparison(
                        imgs[i], gt, predicted[i],
                        os.path.join(comparisons_dir, f'sample_{vis_count:04d}.png'),
                        data_id
                    )
                    vis_count += 1

    # ---- Results Summary ----
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Avg Inference Time: {avg_inference_time:.1f} ms/image")
    
    results = {
        'avg_inference_time_ms': avg_inference_time,
        'num_images': len(dataset),
    }

    if has_gt and len(all_iou) > 0:
        mean_iou = np.nanmean(all_iou)
        avg_class_iou = np.nanmean(all_class_iou, axis=0)
        
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"\nPer-Class IoU:")
        for name, iou in zip(CLASS_NAMES, avg_class_iou):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            print(f"  {name:<20}: {iou_str}")
        
        results['mean_iou'] = float(mean_iou)
        results['class_iou'] = {name: float(iou) if not np.isnan(iou) else None 
                                for name, iou in zip(CLASS_NAMES, avg_class_iou)}
        
        # Save confusion matrix
        save_confusion_matrix_plot(confusion_matrix, 
                                   os.path.join(args.output_dir, 'confusion_matrix.png'))
        print(f"\nSaved confusion matrix to '{args.output_dir}/confusion_matrix.png'")

        # Failure analysis: find worst-performing classes
        print(f"\n--- Failure Analysis ---")
        sorted_classes = sorted(enumerate(avg_class_iou), key=lambda x: x[1] if not np.isnan(x[1]) else 1.0)
        print("Worst performing classes:")
        for class_id, iou in sorted_classes[:5]:
            if not np.isnan(iou):
                # Find most common misclassification
                row = confusion_matrix[class_id]
                if row.sum() > 0:
                    row_norm = row / row.sum()
                    row_norm[class_id] = 0  # exclude correct predictions
                    worst_confusion = np.argmax(row_norm)
                    print(f"  {CLASS_NAMES[class_id]} (IoU={iou:.4f}): "
                          f"most confused with {CLASS_NAMES[worst_confusion]} ({row_norm[worst_confusion]:.1%})")
    else:
        print("No ground truth available - metrics not computed.")
        print("Predictions saved to output directory.")

    print(f"{'='*60}")

    # Save results JSON
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save text summary
    summary_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(summary_path, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Avg Inference Time: {avg_inference_time:.1f} ms/image\n")
        f.write(f"Total Images: {len(dataset)}\n")
        if 'mean_iou' in results:
            f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
            f.write("\nPer-Class IoU:\n")
            for name, iou in results['class_iou'].items():
                f.write(f"  {name:<20}: {iou:.4f}\n" if iou is not None else f"  {name:<20}: N/A\n")
        f.write("=" * 60 + "\n")

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print(f"  masks/        : Raw prediction masks")
    print(f"  masks_color/  : Colored prediction masks")
    print(f"  comparisons/  : Side-by-side visualizations ({vis_count} samples)")
    if has_gt:
        print(f"  confusion_matrix.png")
    print(f"  results.json")
    print(f"  evaluation_metrics.txt")


if __name__ == "__main__":
    main()
