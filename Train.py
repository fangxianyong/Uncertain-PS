import torch
from torch.autograd import Variable
import os
import numpy as np
import argparse
from datetime import datetime
from lib.model import OurNet
from utils.my_dataloader import get_loader,test_dataset
from utils.util import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)    #权值为什么可以这么求,参考F3Net(显著性检测)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()



def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/image/'.format(data_path)
    gt_root = '{}/mask/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        out5,out4,out3,out2 = model(image)
        res = out2
        # eval Dice
        res = F.upsample(res , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

def train(train_loader, model, optimizer, epoch,test_path):
    model.train()
    global best
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record5, loss_record4, loss_record3, loss_record2 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            # images, gts
            images, gts= pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            out5, out4,\
            out3, out2 = model(images)

            # ---- loss function ---
            loss5 = structure_loss(out5, gts)
            loss4 = structure_loss(out4, gts)
            loss3 = structure_loss(out3, gts)
            loss2 = structure_loss(out2, gts)

            loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                        loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))


    #save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ ' OurNet.pth')

    # choose the best model
    global dict_plot

    test1path = './data/TestDataset/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
        meandice = test(model, test_path, 'test')
        dict_plot['test'].append(meandice)
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'OurNet.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'OurNet-best.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))


if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    model_name = 'OurNet'
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='./snapshots/' + model_name + '/')
    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset/' , help='path to testing dataset')


    opt = parser.parse_args()

    logging.basicConfig(filename=model_name+'_train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = OurNet()
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.cuda()

    best = 0

    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr,weight_decay=1e-4)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)


    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch,opt.test_path)


