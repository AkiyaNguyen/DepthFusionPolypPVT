"""
Train a lightweight MobileNetV3-Small classifier to select the best enhancement for each image.
Reads pseudo labels from JSON (e.g. TrainDataset.json produced by enhancement_pseudo_label.py).
"""

import os
import json
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

NUM_CLASSES = 4  # 0=normal, 1=denoise, 2=clahe, 3=unsharp
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_canonical_name(path: str) -> str:
    """Map image filename to canonical .png form (matches enhancement_pseudo_label JSON keys)."""
    name = os.path.basename(path)
    if name.lower().endswith('.jpg'):
        return name[:-4] + '.png'
    return name


class EnhanceSelectionDataset(Dataset):
    """Dataset for enhancement selection: image -> enhancement class (0-3)."""

    def __init__(self, image_root: str, pseudo_labels: dict, img_size: int = 224, augment: bool = False):
        self.image_root = image_root.rstrip('/') + '/'
        self.pseudo_labels = pseudo_labels
        self.img_size = img_size
        self.samples = []

        # Build list of (path, label) for images that have a label
        all_files = [f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png'))]
        for f in sorted(all_files):
            canonical = get_canonical_name(f)
            if canonical in pseudo_labels:
                path = os.path.join(self.image_root, f)
                self.samples.append((path, pseudo_labels[canonical]))

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Small classifier for 4-class enhancement selection."""
    weights = 'IMAGENET1K_V1' if pretrained else None
    try:
        model = mobilenet_v3_small(weights=weights)
    except TypeError:
        model = mobilenet_v3_small(pretrained=pretrained)
    # Replace final Linear (index 3 in Sequential, or use -1 for last layer)
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
    parser = argparse.ArgumentParser(description='Train enhancement selection classifier')
    parser.add_argument('--train_dataset', type=str, default='./dataset/TrainDataset/',
                        help='Path to train dataset root (images/ subdir or direct image folder)')
    parser.add_argument('--output_folder', type=str, default='./model_pth/EnhanceSelector/',
                        help='Folder to save checkpoints')
    parser.add_argument('--pseudo_label_json', '--pseudo_label_path', type=str, default=None, dest='pseudo_label_json',
                        help='Path to pseudo label JSON file. Can be any location (not tied to dataset dir). '
                             'If not set, falls back to {train_dataset}/TrainDataset.json or pseudo_label_enhancement.json')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction of data for validation')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    image_root = opt.train_dataset
    if os.path.isdir(os.path.join(image_root, 'images')):
        image_root = os.path.join(image_root, 'images')
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image folder not found: {image_root}")

    json_path = opt.pseudo_label_json
    if json_path is None:
        dataset_name = os.path.basename(os.path.normpath(os.path.dirname(image_root)))
        if dataset_name == 'images':
            dataset_name = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(image_root))))
        json_path = os.path.join(os.path.dirname(image_root), f'{dataset_name}.json')
        if not os.path.isfile(json_path):
            json_path = os.path.join(os.path.dirname(image_root), 'pseudo_label_enhancement.json')
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Pseudo label JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        pseudo_labels = json.load(f)
    # JSON may store int or str values
    pseudo_labels = {k: int(v) for k, v in pseudo_labels.items()}

    print(f"Loaded {len(pseudo_labels)} pseudo labels from {json_path}")
    dataset = EnhanceSelectionDataset(image_root, pseudo_labels, img_size=opt.img_size, augment=True)
    n_val = max(1, int(len(dataset) * opt.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    os.makedirs(opt.output_folder, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, opt.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:3d}/{opt.epochs} | train loss: {train_loss:.4f} acc: {train_acc:.4f} | val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(opt.output_folder, 'best.pth'))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        if epoch % opt.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(opt.output_folder, f'epoch_{epoch}.pth'))

    print(f"Training done. Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
