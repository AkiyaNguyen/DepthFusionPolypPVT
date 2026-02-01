"""
Create pseudo labels for image enhancement selection.
For each training image: label = the enhancement that yields the highest Dice score.
Saves {image_name: enhance_option} where 0=normal, 1=denoise, 2=clahe, 3=unsharp.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm

from lib.pvt import DepthFusePolypPVT
from utils.dataloader import rgb_loader, binary_loader

# Enhancement option constants
ENHANCE_NORMAL = 0
ENHANCE_DENOISE = 1
ENHANCE_CLAHE = 2
ENHANCE_UNSHARP = 3

# ImageNet normalization (same as model expects)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def enhance_image(image_np: np.ndarray, option: int) -> np.ndarray:
    """
    Apply enhancement to RGB image (H, W, 3) in uint8 [0, 255].
    Returns enhanced image as numpy array.
    Aligned with trigger.ipynb: denoise=Gaussian blur, clahe=histogram equalize, unsharp=unsharp mask.
    """
    if option == ENHANCE_NORMAL:
        return image_np.copy()

    if option == ENHANCE_DENOISE:
        # Match trigger.ipynb denoise_tensor: TF.gaussian_blur(kernel_size=(3,3), sigma=(0.5,0.5))
        return cv2.GaussianBlur(image_np, (3, 3), 0.5)

    if option == ENHANCE_CLAHE:
        # Match trigger.ipynb apply_clahe_tensor: TF.equalize (histogram equalization)
        result = np.zeros_like(image_np)
        for c in range(3):
            result[..., c] = cv2.equalizeHist(image_np[..., c])
        return result

    if option == ENHANCE_UNSHARP:
        # Match trigger.ipynb unsharp_mask_tensor: gaussian_blur(5,5) sigma=1, strength=1.5
        blurred = cv2.GaussianBlur(image_np.astype(np.float32), (5, 5), 1.0)
        sharpened = image_np.astype(np.float32) + (image_np.astype(np.float32) - blurred) * 1.5
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown enhancement option: {option}")


def image_to_tensor(image_np: np.ndarray, testsize: int, device: torch.device) -> torch.Tensor:
    """Convert numpy RGB image to normalized tensor (1, 3, H, W)."""
    img = Image.fromarray(image_np.astype(np.uint8))
    img = img.resize((testsize, testsize), Image.BILINEAR)
    img_np = np.asarray(img, np.float32) / 255.0
    for c in range(3):
        img_np[..., c] = (img_np[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def compute_dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-8) -> float:
    """Compute Dice score between prediction and ground truth (both in [0, 1])."""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    return (2.0 * intersection + smooth) / (np.sum(pred_flat) + np.sum(gt_flat) + smooth)


def get_image_name(path: str) -> str:
    """Extract filename and ensure .png extension for consistency."""
    name = os.path.basename(path)
    if name.lower().endswith('.jpg'):
        name = name[:-4] + '.png'
    return name


def main():
    parser = argparse.ArgumentParser(description='Create enhancement pseudo labels for training dataset')
    parser.add_argument('--train_dataset', type=str, default='./dataset/TrainDataset/',
                        help='Path to train dataset root (with images/, masks/, depths/ subdirs)')
    parser.add_argument('--output_folder', type=str, default='./dataset/TrainDataset/',
                        help='Folder to save pseudo label JSON files')
    parser.add_argument('--model_pth', type=str, default='./model_pth/DepthFusion/depthFusion.pth',
                        help='Path to DepthFusePolypPVT model weights')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--method_name', type=str, default='DepthFusion',
                        help='Method name for logging')
    parser.add_argument('--testsize', type=int, default=352, help='Input size for model')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name for output JSON (default: extracted from train_dataset path)')
    parser.add_argument('--pseudo_label_output', type=str, default=None,
                        help='Full path to save pseudo label JSON. Overrides output_folder+dataset_name when set.')
    opt = parser.parse_args()

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    image_root = os.path.join(opt.train_dataset, 'images')
    gt_root = os.path.join(opt.train_dataset, 'masks')
    depth_root = os.path.join(opt.train_dataset, 'depths')

    if not os.path.isdir(image_root) or not os.path.isdir(gt_root) or not os.path.isdir(depth_root):
        raise FileNotFoundError(
            f"Train dataset must have images/, masks/, depths/ subdirs. "
            f"Found: images={os.path.isdir(image_root)}, masks={os.path.isdir(gt_root)}, depths={os.path.isdir(depth_root)}"
        )

    images = sorted([f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png'))])
    gts = sorted([f for f in os.listdir(gt_root) if f.lower().endswith('.png')])
    depths = sorted([f for f in os.listdir(depth_root) if f.lower().endswith('.png')])

    # Align by base name (handle .jpg vs .png)
    def base_name(p):
        return os.path.splitext(p)[0]

    img_bases = {base_name(p): p for p in images}
    gt_bases = {base_name(p): p for p in gts}
    depth_bases = {base_name(p): p for p in depths}
    common = set(img_bases) & set(gt_bases) & set(depth_bases)
    if not common:
        raise ValueError("No common image/gt/depth files found. Check naming alignment.")

    pairs = []
    for b in sorted(common):
        img_path = os.path.join(image_root, img_bases[b])
        gt_path = os.path.join(gt_root, gt_bases[b])
        depth_path = os.path.join(depth_root, depth_bases[b])
        pairs.append((img_path, gt_path, depth_path))

    print(f"Loading {opt.method_name} from {opt.model_pth}")
    model = DepthFusePolypPVT()
    model.load_state_dict(torch.load(opt.model_pth, map_location=device))
    model.to(device)
    model.eval()

    output_name = opt.dataset_name or os.path.basename(os.path.normpath(opt.train_dataset))
    if not output_name:
        output_name = 'TrainDataset'
    pseudo_labels = {}

    for img_path, gt_path, depth_path in tqdm(pairs, desc='Creating pseudo labels'):
        image_pil = rgb_loader(img_path)
        depth_pil = rgb_loader(depth_path)
        gt_pil = binary_loader(gt_path)

        image_np = np.asarray(image_pil)
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        depth_np = np.asarray(depth_pil)
        if depth_np.ndim == 2:
            depth_np = np.stack([depth_np] * 3, axis=-1)

        gt_np = np.asarray(gt_pil, np.float32)
        gt_np /= (gt_np.max() + 1e-8)

        best_dice = -1.0
        best_option = ENHANCE_NORMAL

        for option in [ENHANCE_NORMAL, ENHANCE_DENOISE, ENHANCE_CLAHE, ENHANCE_UNSHARP]:
            enhanced = enhance_image(image_np, option)
            img_tensor = image_to_tensor(enhanced, opt.testsize, device)
            depth_tensor = image_to_tensor(depth_np, opt.testsize, device)

            with torch.no_grad():
                P1, P2 = model(img_tensor, depth_tensor)
                res = F.upsample(P1 + P2, size=gt_np.shape[-2:], mode='bilinear', align_corners=False)
                res = res.sigmoid().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            dice = compute_dice(res, gt_np)
            if dice > best_dice:
                best_dice = dice
                best_option = option

        name = get_image_name(img_path)
        pseudo_labels[name] = int(best_option)

    if opt.pseudo_label_output:
        out_path = opt.pseudo_label_output
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    else:
        os.makedirs(opt.output_folder, exist_ok=True)
        out_path = os.path.join(opt.output_folder, f'{output_name}.json')
    with open(out_path, 'w') as f:
        json.dump(pseudo_labels, f, indent=2)

    print(f"Saved pseudo labels to {out_path} ({len(pseudo_labels)} images)")
    option_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for v in pseudo_labels.values():
        option_counts[v] = option_counts.get(v, 0) + 1
    print("Enhancement distribution: normal={}, denoise={}, clahe={}, unsharp={}".format(
        option_counts.get(0, 0), option_counts.get(1, 0), option_counts.get(2, 0), option_counts.get(3, 0)))


if __name__ == '__main__':
    main()
