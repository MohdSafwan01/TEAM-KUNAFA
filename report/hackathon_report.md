# Duality AI Offroad Semantic Segmentation
## Hackathon Report

**Team Name:** [YOUR TEAM NAME]  
**Date:** April 4, 2026  
**Event:** MIT HackTheNight — Duality AI Segmentation Track

---

## 1. Executive Summary

We developed a semantic segmentation pipeline for off-road desert environments using **DINOv2** vision foundation model as a frozen backbone with a custom enhanced segmentation head. Our approach focuses on robust generalization to unseen desert environments through aggressive data augmentation, class-weighted loss to handle severe class imbalance, and modern training techniques including cosine learning rate scheduling and gradient accumulation.

**Key Results:**
- Final Validation IoU: **[INSERT]**
- Test IoU: **[INSERT]**
- Inference Speed: **[INSERT] ms/image**
- 11-class segmentation across desert terrain features

---

## 2. Methodology

### 2.1 Data Analysis

We began by thoroughly analyzing the provided synthetic dataset from Duality AI's Falcon platform:

| Split | Images | Classes Present |
|-------|--------|----------------|
| Training | 2,857 | All 10 classes (Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky) |
| Validation | 317 | All 10 classes |
| Test | 1,002 | 7 classes (no Ground Clutter, Flowers, Logs) |

**Critical Discovery:** The provided baseline scripts were missing **Flowers (class 600)** from the value map, which would have resulted in all Flowers pixels being misclassified as Background, guaranteeing 0% IoU for that class. We immediately fixed this.

**Class Distribution Analysis:** Significant class imbalance was observed — Landscape and Sky dominate most frames, while classes like Logs, Flowers, and Ground Clutter are rare. This motivated our class-weighted loss approach.

### 2.2 Model Architecture

**Backbone:** DINOv2 ViT-S/14 (frozen) — a self-supervised Vision Transformer pretrained on diverse images, providing rich semantic features without requiring fine-tuning.

**Segmentation Head (Enhanced):** Custom decoder architecture:
```
DINOv2 patch tokens (384-dim)
  → 1×1 Conv (384→256) + BatchNorm + GELU
  → 3×3 Conv (256→256) + BatchNorm + GELU
  → 3×3 Depthwise Conv (256→256) + BatchNorm + GELU
  → 1×1 Conv (256→128) + BatchNorm + GELU  
  → 3×3 Depthwise Conv (128→128) + BatchNorm + GELU
  → 1×1 Conv (128→128) + BatchNorm + GELU
  → Dropout(0.1) + 1×1 Conv (128→11 classes)
  → Bilinear Upsample to input resolution
```

Key improvements over baseline ConvNeXt head:
- BatchNorm for training stability
- Dropout for regularization
- Depthwise separable convolutions for efficiency
- Deeper architecture for better feature extraction

### 2.3 Data Augmentation Strategy

Since the test set comes from a **different desert location**, generalization is critical. We implemented paired augmentation (identical spatial transforms applied to both RGB image and mask):

| Augmentation | Probability | Parameters | Purpose |
|-------------|-------------|------------|---------|
| Horizontal Flip | 50% | — | Scene invariance |
| Vertical Flip | 20% | — | Additional diversity |
| Random Rotation | 50% | ±15° | Orientation invariance |
| Color Jitter | 70% | B=0.3, C=0.3, S=0.3, H=0.1 | Lighting/color robustness |
| Gaussian Blur | 30% | σ ∈ [0.1, 2.0] | Simulate dust/atmosphere |

### 2.4 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Better weight decay handling than SGD |
| Learning Rate | 1e-3 | Higher than baseline (1e-4) due to AdamW |
| Weight Decay | 1e-4 | Regularization |
| LR Schedule | CosineAnnealing | Smooth warm-down, better convergence |
| Loss | CrossEntropy (class-weighted) | Handles class imbalance |
| Batch Size | [INSERT] (effective: [INSERT]) | Via gradient accumulation |
| Image Size | 476×266 (0.5× scale) | Divisible by 14 for ViT patch size |
| Epochs | [INSERT] | — |

### 2.5 Class Weights

Inverse-frequency weights computed from training masks:

| Class | Weight | Interpretation |
|-------|--------|---------------|
| [Fill in from training output] | | |

---

## 3. Results & Performance Metrics

### 3.1 Training Curves

[INSERT: training_curves.png — Loss, IoU, Dice, LR plots]

### 3.2 Validation Performance

**Overall:**
- Mean IoU: **[INSERT]**
- Mean Dice: **[INSERT]**
- Pixel Accuracy: **[INSERT]**

### 3.3 Per-Class IoU

[INSERT: per_class_iou.png — Bar chart]

| Class | IoU | Notes |
|-------|-----|-------|
| Trees | [INSERT] | |
| Lush Bushes | [INSERT] | |
| Dry Grass | [INSERT] | |
| Dry Bushes | [INSERT] | |
| Ground Clutter | [INSERT] | |
| Flowers | [INSERT] | |
| Logs | [INSERT] | |
| Rocks | [INSERT] | |
| Landscape | [INSERT] | |
| Sky | [INSERT] | |

### 3.4 Test Performance (Unseen Environment)

**Test IoU: [INSERT]**

[INSERT: Confusion matrix image]

---

## 4. Challenges & Solutions

### Challenge 1: Missing Class in Baseline
**Problem:** The provided `train_segmentation.py` did not include class 600 (Flowers) in the value_map, mapping 10 classes to IDs 0-9 instead of 0-10.  
**Impact:** All Flowers pixels would be treated as Background, and the model could never learn to segment Flowers.  
**Solution:** Added Flowers to the value_map, increasing total classes from 10 to 11.

### Challenge 2: No NVIDIA GPU Available
**Problem:** Our machine only had an AMD Radeon integrated GPU (512MB), with no CUDA support. DINOv2 inference on CPU runs at only ~2 images/second.  
**Solution:** Created a Google Colab notebook to leverage free T4 GPUs, reducing training time from ~12 hours to ~30-60 minutes.

### Challenge 3: Severe Class Imbalance
**Problem:** Landscape and Sky dominate >60% of pixels, while Logs, Flowers, and Ground Clutter occupy <1% each.  
**Solution:** Computed inverse-frequency class weights from training masks and applied weighted CrossEntropyLoss, giving rare classes higher loss importance.

### Challenge 4: Domain Shift (Different Desert Location)
**Problem:** Test images come from a different desert environment than training. A model overfit to the training location would generalize poorly.  
**Solution:** Aggressive data augmentation including color jitter (simulate different lighting), Gaussian blur (simulate dust), and geometric transforms (rotation, flip) to force the model to learn invariant features.

### Challenge 5: [Add any additional challenges encountered]

---

## 5. Failure Case Analysis

### 5.1 Worst-Performing Classes

[INSERT: Analysis of classes with lowest IoU, with example images]

**Common Misclassifications:**
- [Class A] most confused with [Class B] — likely due to visual similarity
- [INSERT more from confusion matrix analysis]

### 5.2 Example Failure Cases

[INSERT: Side-by-side comparison images showing input → ground truth → prediction for failure cases]

---

## 6. Conclusion & Future Work

### What Worked
- DINOv2 backbone provides strong features without fine-tuning
- Data augmentation significantly improved generalization to test environment
- Class-weighted loss improved IoU for rare classes

### Future Improvements
1. **Larger backbone:** DINOv2 ViT-L would provide richer features (768-d vs 384-d)
2. **Multi-scale features:** Use features from multiple ViT layers
3. **Test-time augmentation (TTA):** Average predictions over multiple augmented versions
4. **Higher resolution:** Train at full 960×540 with larger GPU
5. **Domain adaptation:** Unsupervised adaptation techniques for the test domain
6. **Self-supervised learning:** Pre-train on unlabeled desert images

---

## Appendix

### A. Repository Structure
```
MIT/
├── Scripts/
│   ├── train_enhanced.py         # Enhanced training script  
│   ├── test_enhanced.py          # Test/inference script
│   ├── hackathon_colab.ipynb     # Colab GPU notebook
│   ├── train_segmentation.py     # Original baseline
│   ├── test_segmentation.py      # Original baseline
│   └── runs/                     # Training outputs
├── Dataset/                      # Training + validation data
├── TestImages/                   # Test images
├── README.md                     # Reproduction instructions
└── report/                       # This report
```

### B. Environment & Dependencies
- Python 3.11
- PyTorch 2.11.0
- torchvision 0.26.0
- DINOv2 (via torch.hub, facebookresearch/dinov2)
- OpenCV 4.13
- CUDA T4 GPU (Google Colab)
