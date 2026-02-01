import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT, DepthFusePolypPVT
from utils.dataloader import test_depth_enhance_dataset
import cv2

## inference using cpu
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT.pth')
    parser.add_argument('--name', type=str, default='PolypPVT')
    opt = parser.parse_args()
    # model = PolypPVT
    model = DepthFusePolypPVT()
    model.load_state_dict(torch.load(opt.pth_path, map_location=torch.device('cpu')))
    # model.cuda()
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:

    # for _data_name in ['TrainDataset']: 
        ##### put data_path here #####
        data_path = './dataset/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/{}/{}/'.format(opt.name, _data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        depth_root = '{}/depths/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_depth_enhance_dataset(image_root, gt_root, depth_root, 352)
        for i in range(num1):
            image, depth, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            # image = image.cuda()
            image = image.to(device)
            depth = depth.to(device)
            P1,P2 = model(image, depth)
            res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')
