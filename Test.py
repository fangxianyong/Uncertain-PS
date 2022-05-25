import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import imageio
from lib.model import OurNet
from utils.my_dataloader import test_dataset

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/OurNet/OurNet.pth')

for _data_name in [ 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/OurNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = OurNet()
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.eval()

    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/image/'.format(data_path)
    gt_root = '{}/mask/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5 = model(image)
        res = res5
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)


        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, res)
