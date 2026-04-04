
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from train_segmentation_optimized import MaskDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Configuration - Adjusted to 518 to match DINOv2 requirements
IMG_SIZE = 518
BATCH_SIZE = 4 # Reduced batch size slightly as resolution is higher
EPOCHS = 15
LR = 2e-5

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device).long().squeeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

def validate(model, loader, criterion, device):
    model.eval()
    total_iou = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Evaluating'):
            images, masks = images.to(device), masks.to(device).long().squeeze(1)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for p, t in zip(preds, masks):
                intersection = (p & t).float().sum()
                union = (p | t).float().sum()
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou.append(iou.item())
    avg_iou = np.mean(total_iou)
    print(f'Validation IoU: {avg_iou:.4f}')
    return {'iou': avg_iou}

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

def create_enhanced_model(num_classes=10):
    backbone = timm.create_model('vit_small_patch14_dinov2', pretrained=True, num_classes=0)
    for param in backbone.parameters():
        param.requires_grad = True

    class DecoderBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
            )
        def forward(self, x): return self.conv(x)

    class EnhancedModel(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec1 = DecoderBlock(384, 128)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec2 = DecoderBlock(128, 64)
            self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
            
        def forward(self, x):
            features = self.backbone.forward_features(x)
            grid_size = int((features.shape[1] - 1)**0.5)
            x = features[:, 1:, :].transpose(1, 2).reshape(-1, 384, grid_size, grid_size)
            x = self.final_conv(self.dec2(self.up2(self.dec1(self.up1(x)))))
            return F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            
    return EnhancedModel(backbone, num_classes)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_enhanced_model().to(device)
    train_ds = MaskDataset('/content/Offroad_Segmentation_Training_Dataset/train', h=IMG_SIZE, w=IMG_SIZE, is_train=True)
    val_ds = MaskDataset('/content/Offroad_Segmentation_Training_Dataset/train', h=IMG_SIZE, w=IMG_SIZE, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_iou = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            torch.save(model.state_dict(), "enhanced_model_best.pth")
            print(f"New Best IoU: {best_iou:.4f}")
