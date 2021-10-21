from __future__ import print_function
import argparse
import numpy as np

import skimage
import skimage.io
import skimage.transform
from PIL import Image

import sys
import os
import re
from struct import unpack
import torch
import torch.nn.parallel
from torch.autograd import Variable


# evaluation settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--kitti', type=int, default=True, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--mideval', type=int, default=0, help='middlesburry? Default=False')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--test_list', type=str, required=True, help="test images list")
parser.add_argument('--save_path', type=str, default='./evaluate-result/', help="location to save result")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to test")
parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")

# ******************************************************************************************************
# parser.add_argument()
import time
from utils.visualize import *

parser.add_argument('--print_GT', type=int, default=0, help="print colored GT folder or not")
parser.add_argument('--print_input_images', type=bool, default=False, help="print input test images or not")
parser.add_argument('--max_test_images', type=int, default=30, help="number of images to evaluate/test")
parser.add_argument('--convolution_type', type=str, default='convolution', help='Type of convolution i.e. 3D Conv, FwSC or FDwSC')

from utils.pixel_error import calc_error

# ******************************************************************************************************
opt = parser.parse_args()

print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")

cuda = opt.cuda
# cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# torch.manual_seed(opt.seed)
# if cuda:
#    torch.cuda.manual_seed(opt.seed)
# print('===> Loading datasets')


print('===> Building model')
model = GANet(maxdisp=opt.max_disp, conv_type=opt.convolution_type)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


ndict={}
for k,val in checkpoint['state_dict'].items():
    if "depth_wise" in k :
        old_key=k.rsplit("depth_wise")
        prefix=old_key[0]
        postfix=old_key[1]
        print(old_key,prefix+f"conv.depth_wise"+postfix)
        ndict[prefix+f"conv.depth_wise"+postfix]=val
    elif "point_wise" in k :
        old_key=k.rsplit("point_wise")
        prefix=old_key[0]
        postfix=old_key[1]
        print(old_key,prefix+f"conv.point_wise"+postfix)
        ndict[prefix+f"conv.point_wise"+postfix]=val
    else:
        ndict[k]=val

ncheckpoint=checkpoint
ncheckpoint['state_dict']=ndict
torch.save(ncheckpoint,"updated_3DSeperable_GANet11_Sceneflow_epoch_20.pth")

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


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def test(leftname, rightname, savename):
    #  count=0

    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    model.eval()
    l_img= input1.numpy()
    r_img= input2.numpy()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        st= time.time()
        prediction = model(input1, input2)
        et=time.time()

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        l_img = l_img[0, :, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        r_img = r_img[0, :, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]

    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    return temp, l_img, r_img, et-st

if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    avg_error = 0
    avg_rate = 0
    avg_d1_error=0

    # *************** For 2,3,5 pixel errors
    error1 = torch.Tensor([0.])
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
    # ************************

    max_images = opt.max_test_images
    print(f">>>> Total images to process.... {max_images}")
    cummulative_inference_time=0.0
    sttime = time.time()

    model_name = opt.resume.split("/")[-1][:-4]
    save_path = opt.save_path + "{}".format(model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    GT = opt.print_GT
    if GT:
        GT_path = opt.save_path + "{}".format(model_name) + "/GT"
        if not os.path.exists(GT_path):
            os.makedirs(GT_path)
    else:
        print(">>>> Not saving GT coloured images....")

    print_input_images= opt.print_input_images
    if print_input_images:
        input_images_path = opt.save_path + "{}".format(model_name) + "/input_images"
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

    # print(filelist)
    images_processed = np.minimum(len(filelist), max_images)
    for index in range(images_processed):
        current_file = filelist[index]
        folder_name = file_path + "/" + current_file.strip()

        if opt.kitti2015:
            print('Testing on kitti 2015 Test Set')
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
            dispname = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
            savename = save_path + "/" + current_file.strip()
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0


        elif opt.kitti:
            print('Testing on kitti 2012 Test Set')
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
            dispname = file_path + 'disp_occ/' + current_file[0: len(current_file) - 1]
            savename = save_path + "/" + current_file.strip()
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0


        elif opt.mideval:
            print('Testing on MiddelBurry Test Set')
            leftname = folder_name + '/im0.png'
            rightname = folder_name + "/im1.png"

            tmp = current_file.split("/")
            image_name = tmp[0] + "-" + tmp[1].strip()

            resultfile = save_path + "/{}-disp.png".format(image_name)
            savename = resultfile

        else:
            leftname = opt.data_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = opt.data_path + 'frames_finalpass/' + current_file[
                                                              0: len(current_file) - 14] + 'right/' + current_file[len(
                current_file) - 9:len(current_file) - 1]
            dispname = opt.data_path + 'disparity/' + current_file[0: len(current_file) - 4] + 'pfm'
            savename = save_path + "/" + current_file.replace('/', '_').strip()
            disp, height, width = readPFM(dispname)


        print('Processing image number =', index + 1)
        # test(leftname, rightname, savename)
        prediction,l_img,r_img, inference_time = test(leftname, rightname, savename)
        cummulative_inference_time+=inference_time

        if opt.mideval:
            disp = np.zeros(prediction.shape)  # we are doing this just to incoporate mideval as other (we dont have GT for this data)
        mask = np.logical_and(disp >= 0.001, disp <= opt.max_disp)

        e= np.abs(prediction[mask] - disp[mask])
        error = np.mean(e)
        rate = np.sum(e > opt.threshold) / np.sum(mask)

        d1_error = np.mean(np.logical_and(e > opt.threshold, e / disp[mask] > 0.05))

        avg_error += error
        avg_rate += rate
        avg_d1_error+= d1_error

        print("===> Frame {}: ".format(index) + current_file[0:len(
            current_file) - 1] + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
        print(f"====> D1 error : {d1_error * 100:.4f} %")
        print(f"===> Frame {index} processing time is: {inference_time:.4f} ")

        # *************** For 2,3,5 pixel errors
        error_dict = calc_error(torch.from_numpy(prediction), torch.from_numpy(disp.astype(np.float32)), lb=0.001, ub=opt.max_disp)
        running_error_sum = {k: running_error_sum.get(k, 0) + error_dict.get(k, 0) for k in set(error_dict)}
        # ************************

        error_map= disp_err_to_color(prediction,disp)

        if opt.mideval:
            l_img=l_img.squeeze()
            group_color(prediction, disp, map_to_range(l_img, 0, 255).transpose(1, 2, 0), None, save_path + "/grouped/" + image_name + "-group.png")
            skimage.io.imsave(save_path + "/error-map/" + image_name + "-error-map.png", (error_map * 256).astype('uint16'))
            skimage.io.imsave(save_path + "/predicted-colored/" + image_name + "-pred.png", (disp_to_color(prediction) * 256).astype('uint16'))

        else:
            if opt.kitti2015 or opt.kitti:
                image_name = current_file[:-5]

                group_color(prediction,disp,map_to_range(l_img, 0, 255).transpose(1,2,0),None,save_path + "/grouped/" + image_name+"-group.png")
                skimage.io.imsave(save_path+"/error-map/" + image_name+"-error-map.png", (error_map *256).astype('uint16'))
                skimage.io.imsave(save_path+"/predicted-colored/" + image_name+"-pred.png", (disp_to_color(prediction)*256).astype('uint16'))
            else:
                image_name = current_file.replace('/', '_')[:-5]
                group_color(prediction,disp,map_to_range(l_img, 0, 255).transpose(1,2,0),None,save_path + "/grouped/" + image_name+"-group.png")
                skimage.io.imsave(save_path+"/error-map/" + image_name+"-error-map.png", (error_map *256).astype('uint16'))
                skimage.io.imsave(save_path+"/predicted-colored/" + image_name+"-pred.png", (disp_to_color(prediction)*256).astype('uint16'))

            if GT:
                skimage.io.imsave(save_path+"/GT/" + image_name+"-GT.png", (disp_to_color(disp) *256).astype('uint16'))

            if print_input_images:
                skimage.io.imsave(save_path + "/input_images/" + image_name + "-input.png",
                                  (map_to_range(l_img, 0, 255).transpose(1,2,0) * 256).astype('uint16'))


    ettime = time.time()

    # *************** For 2,3,5 pixel errors
    tmp_dict = {k: running_error_sum.get(k, 0) / images_processed for k in set(running_error_sum)}
    # ************************

    avg_error = avg_error / images_processed
    avg_rate = avg_rate / images_processed
    avg_d1_error= (avg_d1_error / images_processed)*100
    tmp_dict['D1_error']=  avg_d1_error

    print("===> Test: Avg. Error dictionary: ({})".format(tmp_dict))
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}".format(images_processed, avg_error,
                                                                                          avg_rate))
    print(f"====>Average  D1 error : {avg_d1_error:.4f}")
    print(f'Inference Time per Image {cummulative_inference_time/images_processed:4}')
    print('End-to-End Time Per Image of Size={}x{}={}sec'.format(opt.crop_width, opt.crop_height, (ettime - sttime) / (index + 1)))