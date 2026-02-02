"""
Train a lightweight MobileNetV3-Small classifier to select the best enhancement for each image.
Uses PolypDatasetWithAugmentImage which reads ground truth (0=normal, 1=denoise) from a JSON file.
Returns softmax logits for 2-class classification.
"""

import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision.models import mobilenet_v3_small
from utils.dataloader import PolypDatasetWithAugmentImage

NUM_CLASSES = 2  # 0=normal, 1=denoise


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Small classifier for 2-class enhancement selection (normal vs denoise)."""
    try:
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    except TypeError:
        model = mobilenet_v3_small(pretrained=pretrained)
    last_layer = model.classifier[-1]
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            _, pred = logits.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train enhancement selection classifier (normal vs denoise)')
    parser.add_argument('--data_path', type=str, default='./dataset/TrainDataset/',
                        help='Path to dataset root (with images/ and masks/ subdirs)')
    parser.add_argument('--pseudo_label_json', type=str, required=True,
                        help='Path to pseudo label JSON: {image_name: 0|1} where 0=normal, 1=denoise')
    parser.add_argument('--model_save', type=str, default='./model_pth/EnhanceSelector/',
                        help='Folder to save model checkpoints')
    parser.add_argument('--model_pth', type=str, default=None,
                        help='Optional: path to checkpoint to resume training')
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--trainsize', type=int, default=224)
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction of data for validation')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    image_root = os.path.join(opt.data_path, 'images')
    gt_root = os.path.join(opt.data_path, 'masks')
    if not os.path.isdir(image_root) or not os.path.isdir(gt_root):
        raise FileNotFoundError(f"Dataset must have images/ and masks/ subdirs under {opt.data_path}")
    if not os.path.isfile(opt.pseudo_label_json):
        raise FileNotFoundError(f"Pseudo label JSON not found: {opt.pseudo_label_json}")

    dataset = PolypDatasetWithAugmentImage(
        image_root=image_root + '/',
        gt_root=gt_root + '/',
        trainsize=opt.trainsize,
        augmentations='True',
        pseudo_label_json_path=opt.pseudo_label_json,
    )
    n_val = max(1, int(len(dataset) * opt.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    start_epoch = 1

    if opt.model_pth and os.path.isfile(opt.model_pth):
        ckpt = torch.load(opt.model_pth, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 1) + 1
        print(f"Resumed from {opt.model_pth}, epoch {start_epoch}")

    os.makedirs(opt.model_save, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(start_epoch, opt.epoch + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:3d}/{opt.epoch} | train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(opt.model_save, 'best.pth'))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        if epoch % opt.save_every == 0:
            torch.save(ckpt, os.path.join(opt.model_save, f'epoch_{epoch}.pth'))

    print(f"Training done. Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
