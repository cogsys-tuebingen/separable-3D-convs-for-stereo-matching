from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

from models import *


from dataloader import KITTILoader
from utils.pixel_error import calc_error
from torch.autograd import Variable
from utils.visualize import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/data/rahim/data/Kitti_2015/training/',
                    help='select model')
parser.add_argument('--loadmodel', default='./checkpoint/FwSC/FwSC_PSMNet_sceneflow_finetuned_kitti15.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity to consider')
parser.add_argument('--threshold', type=int, default=3.0, help='threshold of error rates')
parser.add_argument('--save_path', type=str, default='./evaluation-results', help='path to store results')
parser.add_argument('--print_GT', type=int, default=0, help="print colored GT folder or not")
parser.add_argument('--print_input_images', type=int, default=0, help="print input test images or not")
parser.add_argument('--max_test_images', type=int, default=30, help="number of images to evaluate/test")
parser.add_argument('--testbatchsize', type=int, default=1, help="")
parser.add_argument('--kitti2015', type=str, default='1', help="whether dataset is kitti or sceneflow")
parser.add_argument('--test_list', default='./lists/kitti2015_val.list',
                    help='select test data')
parser.add_argument('--convolution_type', type=str, default='convolution', help='Type of convolution i.e. 3D Conv, FwSC or FDwSC')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# if args.KITTI == '2015':
#    from dataloader import KITTIloader2015 as DA
# else:
#    from dataloader import KITTI_submission_loader2012 as DA

if args.kitti2015 == '1':
    from dataloader import KITTIloader2015 as DA
else:
    from dataloader import listflowfile as DA

from dataloader import SecenFlowLoader

print(f" ===> Loading dataset.......")
if args.kitti2015 == '1':
    print(f" ===> KITTI2015 dataset.......")
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = DA.dataloader(
        args.datapath)

    TestImgLoader = torch.utils.data.DataLoader(
        KITTILoader.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=1, shuffle=False, num_workers=1, drop_last=False)
else:
    print(f" ===> Sceneflow dataset.......")

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = DA.dataloader(
        args.datapath, args.test_list)

    TestImgLoader = torch.utils.data.DataLoader(
        SecenFlowLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.testbatchsize, shuffle=False, num_workers=1, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, args.convolution_type)
elif args.model == 'basic':
    model = basic(args.maxdisp, args.convolution_type)
else:
    print('no model')


model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print(f" ====> loading checkpoint {args.loadmodel}")
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# def test(imgL,imgR):
#     model.eval()
#
#     if args.cuda:
#         imgL = imgL.cuda()
#         imgR = imgR.cuda()
#
#     with torch.no_grad():
#         output = model(imgL,imgR)
#     output = torch.squeeze(output).data.cpu().numpy()
#     return output

def test(imgL, imgR, disp_true, running_error_sum):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    output = torch.squeeze(output3).data.cpu().numpy()

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

    return output, 1 - (float(torch.sum(correct)) / float(len(index[0]))), running_error_sum


from struct import unpack
import sys
import re
import skimage


def readPFM(file):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    return img, height, width


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])

    error1 = torch.Tensor([0.])  # dictionary bulding
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

    cummulative_inference_time = 0.0
    avg_error = 0
    avg_rate = 0
    avg_d1_error=0

    max_images = args.max_test_images
    total_images = np.minimum(len(test_left_img), max_images)
    print(f"Total images to process are: {total_images}")

    sttime = time.time()
    length, width = 0, 0

    model_name = args.loadmodel.split("/")[-1][:-4]
    save_path = args.save_path + "/{}".format(model_name)


    if not os.path.exists(save_path):
        os.makedirs(save_path)


    GT = args.print_GT
    if GT:
        GT_path = save_path + "/GT"
        if not os.path.exists(GT_path):
            os.makedirs(GT_path)
    else:
        print(">>>> Not saving GT coloured images....")

    print_input_images = args.print_input_images
    if print_input_images:
        input_images_path = save_path + "/input_images"
        if not os.path.exists(input_images_path):
            os.makedirs(input_images_path)
    else:
        print(">>>> Not saving input images....")

    if not os.path.exists(save_path+"/error-map"):
        os.makedirs(save_path+"/error-map")
    if not os.path.exists(save_path + "/predicted-colored"):
        os.makedirs(save_path + "/predicted-colored")
    if not os.path.exists(save_path + "/grouped"):
        os.makedirs(save_path + "/grouped")

    for inx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        if inx == total_images:
            length, width = imgL.shape[-2], imgL.shape[-1]
            break
        # print(f"====> processing image number {inx+1} \n at path: {test_left_img[inx+1]}")
        # imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        # imgR_o = Image.open(test_right_img[inx]).convert('RGB')
        # disp= Image.open(test_left_disp[inx])
        #
        # disp= torch.tensor(np.array(disp))
        #
        # imgL = infer_transform(imgL_o)
        # imgR = infer_transform(imgR_o)

        # pad to width and hight to 16 times
        if args.kitti2015 == '1':
            if imgL.shape[1] % 16 != 0:
                times = imgL.shape[1] // 16
                top_pad = (times + 1) * 16 - imgL.shape[1]
            else:
                top_pad = 0

            if imgL.shape[2] % 16 != 0:
                times = imgL.shape[2] // 16
                right_pad = (times + 1) * 16 - imgL.shape[2]
            else:
                right_pad = 0
        else:
            if imgL.shape[2] % 16 != 0:
                times = imgL.shape[2] // 16
                top_pad = (times + 1) * 16 - imgL.shape[2]
            else:
                top_pad = 0

            if imgL.shape[3] % 16 != 0:
                times = imgL.shape[3] // 16
                right_pad = (times + 1) * 16 - imgL.shape[3]
            else:
                right_pad = 0

        imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0))
        disp_L = F.pad(disp_L, (0, right_pad, top_pad, 0))

        start_time = time.time()
        disp_c = disp_L.numpy().squeeze().copy() # making copy of original disparity before sending to test() as its changed inside

        pred_disp, test_loss, running_error_sum = test(imgL, imgR, disp_L, running_error_sum)
        inference_time = time.time() - start_time
        # print('time = %.2f' %(inference_time))

        # Doing qauntitative evaluations
        cummulative_inference_time += inference_time

        # disp_c = disp_c # making this as a short solution, as rest of the code is using disp_c
        mask = np.logical_and(disp_c >= 0.001, disp_c <= args.max_disp)
        e= np.abs(pred_disp[mask] - disp_c[mask])
        error = np.mean(e)

        rate = np.sum( e > args.threshold) / np.sum(mask)

        d1_error= np.mean(np.logical_and(e > args.threshold,e / disp_c[mask] > 0.05))
        avg_error += error
        avg_rate += rate
        avg_d1_error+= d1_error

        print(
            "====>  Frame {}: ".format(inx + 1) + test_left_img[inx] + "\nEPE Error: {:.4f}, Error Rate: {:.4f}".format(
                error, rate))
        print(f"====> D1 error : {d1_error*100:.4f} %")
        print(f"Frame {inx + 1} processing time is: {inference_time:.4f} ")


        # for qualitative evaluations
        error_map = disp_err_to_color(pred_disp, disp_c)

        image_name = test_left_img[inx].split('/')[-1][:-4]

        imgL = imgL.numpy().squeeze()

        group_color(pred_disp, disp_c, map_to_range(imgL, 0, 255).transpose(1, 2, 0), None,
                    save_path + "/grouped/" + image_name + "-group.png")
        skimage.io.imsave(save_path + "/error-map/" + image_name + "-error-map.png", (error_map * 256).astype('uint16'))
        skimage.io.imsave(save_path + "/predicted-colored/" + image_name + "-pred.png",
                          (disp_to_color(pred_disp) * 256).astype('uint16'))

        if GT:
            skimage.io.imsave(save_path + "/GT/" + image_name + "-GT.png", (disp_to_color(disp_c) * 256).astype('uint16'))

        if print_input_images:
            skimage.io.imsave(save_path + "/input_images/" + image_name + "-input.png",
                              (map_to_range(imgL, 0, 255).transpose(1, 2, 0) * 256).astype('uint16'))

        skimage.io.imsave(save_path + "/" + image_name + '.png', (pred_disp * 256).astype('uint16'))

        # img=pred_disp
        #
        # img = (img*256).astype('uint16')
        # img = Image.fromarray(img)
        # img.save(test_left_img[inx+1].split('/')[-1])

    ettime = time.time()


    # *************** For 2,3,5 pixel errors
    tmp_dict = {k: running_error_sum.get(k, 0) / total_images for k in set(running_error_sum)}
    # ************************

    avg_error = avg_error / total_images
    avg_rate = avg_rate / total_images
    avg_d1_error= (avg_d1_error/ total_images)*100
    tmp_dict['D1_error'] = avg_d1_error

    print("===> Test: Avg. Error dictionary: ({})".format(tmp_dict))
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}".format(total_images, avg_error,
                                                                                          avg_rate))
    print(f"====>Average  D1 error : {avg_d1_error:.4f}")
    print(f'Inference Time per Image (model forward pass only) {cummulative_inference_time / total_images:.4f}')
    print('End-to-End Time Per Image of Size={}x{}={:.4f}sec'.format(length, width, (ettime - sttime) / (total_images)))


if __name__ == '__main__':
    main()
