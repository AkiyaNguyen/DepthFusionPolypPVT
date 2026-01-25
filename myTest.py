import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2

def compute_metrics(pred, gt, metric_names=['dice', 'iou']):
    """
    Currently handle dice, iou only
    """
    scores = {}
    if 'dice' in metric_names:
        intersection = np.sum(pred * gt)
        dice_score = (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)
        scores['dice'] = dice_score
    if 'iou' in metric_names:
        intersection = np.sum(pred * gt)
        iou_score = intersection / (np.sum(pred) + np.sum(gt) - intersection + 1e-8)
        scores['iou'] = iou_score
    return scores

## inference using cpu
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    parser.add_argument('--log_score_every', type=str, default='each', choices=['never', 'all', 'each'], 
                        help='logging score frequency: all (after each dataset), each (log score of each data with name), never (do not log)')
    parser.add_argument('--data_name', type=str, default='trigger', help='List of eval dataset, separated by comma')
    parser.add_argument('--log_to_file', type=str, default=None, help='If specified, log output to this file')
    opt = parser.parse_args()
    if opt.log_to_file is not None:
        import sys
        sys.stdout = open(opt.log_to_file, 'w')

    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path, map_location=torch.device('cpu')))
    # model.cuda()
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    data_names = [name.strip() for name in opt.data_name.split(',')]

    # for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

    for _data_name in data_names:
        ##### put data_path here #####
        data_path = './dataset/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/PolypPVT/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        name_scores = []
        for i in range(num1):
            image, gt, name = test_loader.load_data()

            image = image.to(device)
            P1,P2 = model(image)
            pred = P1 + P2

            res, gt = test_loader.process_for_inference(pred, gt)
            score = compute_metrics(res, gt, metric_names=['dice', 'iou'])
            name_scores.append((name, score))

            if opt.log_score_every == 'each':
                print(f"{_data_name} - {name}: ")
                for metric_name, metric_value in score.items():
                    print(f"    {metric_name}: {metric_value:.4f}")

            
            cv2.imwrite(save_path+name, res*255) ## scale back to [0,255] to write img
        if opt.log_score_every == 'all':
            # compute average score
            if len(name_scores) == 0:
                print(f"No scores to log for dataset {_data_name}.")
                continue
            avg_scores = {}
            for _, score in name_scores:
                for metric_name, metric_value in score.items():
                    if metric_name not in avg_scores:
                        avg_scores[metric_name] = 0.0
                    avg_scores[metric_name] += metric_value
            for metric_name in avg_scores:
                avg_scores[metric_name] /= num1
            print(f"{_data_name} - Average Scores :", "".join([f"{k}: {v:.4f}" for k, v in avg_scores.items()]))

        print(_data_name, 'Finish!')
