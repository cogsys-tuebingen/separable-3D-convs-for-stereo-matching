from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA

from models import *

# from utils.flop_counter.flops_counter import *
import shutil
from torch.utils.tensorboard import SummaryWriter
from utils.pixel_error import calc_error


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/data/rahim/data/Kitti_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./checkpoint/FwSC/FwSC_PSMNet_sceneflow.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./checkpoint',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--trainbatchsize', type=int, default=12,
                    help='training batch size')

parser.add_argument('--testbatchsize', type=int, default=8,
                    help='test batch size')
parser.add_argument('--tensorboard_logs', type=str, default='runs/tmp', help="path to log for tensorboard")
parser.add_argument('--epoch_num', type=int, default=1, help="epoch number to start from")
parser.add_argument('--convolution_type', type=str, default='convolution', help='Type of convolution i.e. 3D Conv, FwSC or FDwSC')


args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls

print('===> Loading datasets')
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=args.trainbatchsize, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=args.testbatchsize, shuffle=False, num_workers=4, drop_last=False)

print('===> Building model')
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, args.convolution_type)
elif args.model == 'basic':
    model = basic(args.maxdisp, args.convolution_type)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    if os.path.isfile(args.loadmodel):
        print("=> loading checkpoint '{}'".format(args.loadmodel))
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'],strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(args.loadmodel))

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
writer = SummaryWriter(args.tensorboard_logs)
print(model)

def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                  size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss
    # return loss.data[0]


def test(imgL, imgR, disp_true,running_error_sum):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu()

    # *************** For 2,3,5 pixel errors
    error_dict = calc_error(pred_disp, disp_true, lb=0.001, ub=args.maxdisp)
    running_error_sum = {k: running_error_sum.get(k, 0) + error_dict.get(k, 0) for k in set(error_dict)}
    # ************************

    # computing 3-px error#
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
            disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
        index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0]))), running_error_sum


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc = 100
    max_epo = 0
    start_full_time = time.time()
    print_stats = True

    for epoch in range(args.epoch_num, args.epochs + 1):
        epoch_start = time.time()
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        train_len = len(TrainImgLoader)
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            # import copy
            # if print_stats:
            #     flops, params = get_model_complexity_info(copy.deepcopy(model),
            #                                               input_res=(3, imgL_crop.shape[-2], imgL_crop.shape[-1]),
            #                                               input_constructor=prepare_input,
            #                                               as_strings=True,
            #                                               print_per_layer_stat=False)  # make it true to see layer wise stats
            #     print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            #     print_stats = False

            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d / %d training loss = %.3f , time = %.2f' % (batch_idx, train_len, loss, time.time() - start_time))
            total_train_loss += loss

        writer.add_scalar('Avg. Training Loss', total_train_loss/ train_len, epoch)
        print('epoch %d total training loss = %.3f, epoch time= %.2f' % (
        epoch, total_train_loss / len(TrainImgLoader), time.time() - epoch_start))

        ## Test ##
        is_best = False

        error1 = torch.Tensor([0.]) # dictionary bulding
        error2 = torch.Tensor([0.])
        error3 = torch.Tensor([0.])
        error5 = torch.Tensor([0.])
        epe = torch.Tensor([0.])

        running_error_sum = {
            '1px': error1 * 100,
            '2px': error2 * 100,
            '3px': error3 * 100,
            '5px': error5 * 100,
            'epe': epe
        }

        test_len= len(TestImgLoader)
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss, running_error_sum = test(imgL, imgR, disp_L, running_error_sum)
            print('Iter %d / %d 3-px error in val = %.3f' % (batch_idx, test_len, test_loss * 100))
            total_test_loss += test_loss

        writer.add_scalar('Avg. Test Loss', test_loss/ test_len, epoch)
        tmp_dict = {k: running_error_sum.get(k, 0) / test_len for k in set(running_error_sum)}

        for key, val in tmp_dict.items():
            writer.add_scalar('{} error rate'.format(key), val, epoch)

        print("===> Test: Avg. Error dictionary: ({})".format(tmp_dict))

        print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_loss / len(TestImgLoader) * 100))
        if total_test_loss / len(TestImgLoader) * 100 < max_acc:
            max_acc = total_test_loss / len(TestImgLoader) * 100
            max_epo = epoch
            is_best = True
        print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        # SAVE
        if epoch % 50 == 0 or is_best:
            savefilename = args.savemodel + 'finetune_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                'test_loss': total_test_loss / len(TestImgLoader) * 100,
            }, savefilename)
            print(f"saving model: {savefilename}")
            if is_best:
                shutil.copyfile(savefilename, args.savemodel + 'finetune_best.tar')
                print(f" Current best epoch is: {epoch}")

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
    main()
