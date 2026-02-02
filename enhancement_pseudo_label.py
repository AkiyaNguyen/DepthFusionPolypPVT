"""
Create pseudo labels for image enhancement selection.
For each training image: label = the enhancement that yields the highest Dice score.
Saves {image_name: enhance_option} where 0=normal, 1=denoise.
"""

import os
import json
import argparse
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
from lib.pvt import DepthFusePolypPVT
from utils.dataloader import test_depth_enhance_dataset, gt_to_normalized_numpy, process_output_for_inference

# Enhancement option constants (only 2 options)
ENHANCE_NORMAL = 0
ENHANCE_DENOISE = 1


def enhance_image(image_tensor: torch.Tensor, option: int) -> torch.Tensor:
    """
    Apply enhancement to image tensor (B, 3, H, W) or (3, H, W).
    Returns enhanced image as tensor, same shape.
    Options: 0=normal, 1=denoise (Gaussian blur, aligned with trigger.ipynb).
    """
    if option == ENHANCE_NORMAL:
        return image_tensor.clone()

    if option == ENHANCE_DENOISE:
        # Match trigger.ipynb denoise_tensor: TF.gaussian_blur(kernel_size=(3,3), sigma=(0.5,0.5))
        return gaussian_blur(image_tensor, kernel_size=(3, 3), sigma=(0.5, 0.5))

    raise ValueError(f"Unknown enhancement option: {option}")


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

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') ## either cuda or cpu
    image_root = os.path.join(opt.train_dataset, 'images/')
    gt_root = os.path.join(opt.train_dataset, 'masks/')
    depth_root = os.path.join(opt.train_dataset, 'depths/')

    model = DepthFusePolypPVT().to(device)
    model.load_state_dict(torch.load(opt.model_pth, map_location=device))
    model.eval()

    test_loader = test_depth_enhance_dataset(image_root, gt_root, depth_root, opt.testsize)
    better_dice_candidate_id = []
    for _ in tqdm(range(test_loader.size)):
        image, depth, gt, name = test_loader.load_data()
        image = image.to(device) ## already tensor
        depth = depth.to(device)
        gt = gt_to_normalized_numpy(gt)
        dice_scores = []
        for option in [ENHANCE_NORMAL, ENHANCE_DENOISE]:
            new_img = enhance_image(image, option)
            assert new_img.shape == image.shape
            depth_clone = depth.clone() 
            p1, p2 = model(new_img, depth_clone)
            result = process_output_for_inference(p1, p2, gt)
            dice = compute_dice(result, gt)
            dice_scores.append(dice)
        best_dice_index = np.argmax(dice_scores)
        better_dice_candidate_id.append((name, best_dice_index))
    
    print("Statistics of better dice candidate id:")
    normal_exceed, denoise_exceed = 0, 0
    for id in better_dice_candidate_id:
        if id[1] == ENHANCE_NORMAL:
            normal_exceed += 1
        else:
            denoise_exceed += 1
    print(f"Normal exceed: {normal_exceed}, Denoise exceed: {denoise_exceed}")
    print(f"Total: {len(better_dice_candidate_id)}")
    with open(opt.pseudo_label_output, 'w') as f:
        better_dice_candidate_id_dict = {id[0]: id[1] for id in better_dice_candidate_id}
        json.dump(better_dice_candidate_id_dict, f)

if __name__ == '__main__':
    main()
