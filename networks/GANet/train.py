from __future__ import print_function
import argparse

from libs.GANet.modules.GANet import MyLoss2
import shutil
import os
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.functional as F
from dataloader.data import get_training_set, get_test_set

# ****************************************************************************************
from utils.pixel_error import calc_error

import skimage
import skimage.io
import skimage.transform

from torch.utils.tensorboard import SummaryWriter
from utils.flop_counter.flops_counter import * # flop counter tool has been adapted for custom layers in GANet
from utils.helper_func import *
import wandb
# ****************************************************************************************


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, default='/data/rahim/stereo_data/', help="data root")
parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
parser.add_argument('--save_path', type=str, default='./checkpoint/', help="location to save models")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

# ****************************************************************************************
parser.add_argument('--tensorboard_logs', type=str, default='runs/tmp1', help="path to log for tensorboard")
parser.add_argument('--convolution_type', type=str, default='convolution', help='Type of convolution i.e. 3D Conv, FwSC or FDwSC')
parser.add_argument('--val_save_path', type=str, default='', help='path to save validation set predicted images')
parser.add_argument('--epoch_nums', type=int, default=1, help='starting epoch number, can send one if want to resume')
parser.add_argument('--wandb_project_name', type=str, default="test", help='name of the weights and baises project')
parser.add_argument('--wandb_run_name', type=str, default="test", help='name of the run inside weights and baises project')
# ****************************************************************************************
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

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_path, opt.training_list, [opt.crop_height, opt.crop_width], opt.left_right,
                             opt.kitti, opt.kitti2015, opt.shift)
# test_set = get_test_set(opt.data_path, opt.val_list, [576,960], opt.left_right, opt.kitti, opt.kitti2015)
test_set = get_test_set(opt.data_path, opt.val_list, [opt.crop_height, opt.crop_width], opt.left_right, opt.kitti,
                        opt.kitti2015)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                                  drop_last=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(maxdisp=opt.max_disp, conv_type=opt.convolution_type)

criterion = MyLoss2(thresh=3, alpha=2)
if cuda:
    model = torch.nn.DataParallel(model).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    #        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

# ****************************************************************************************

# print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# default `log_dir` is "runs" - we'll be more specific here
if not opt.resume:
    try:
        shutil.rmtree(opt.tensorboard_logs, ignore_errors=False, onerror=None)
    except:
        pass

if opt.val_save_path:
    val_save_path= opt.val_save_path
else:
    val_save_path= None

writer = SummaryWriter(opt.tensorboard_logs)
writer.add_scalar('Model parameters from model.params()', pytorch_total_params)


@timer   # this is a wrapper function defined in utils/helper_func to calculate time of a given function e.g. train
def train(epoch):

    # TODO: take log_freq as argument/ adapt according to kitti/SF eg. 20 logs per epoch
    wandb.watch(model,log='gradients', log_freq=50)

    epoch_loss = 0
    epoch_error0 = 0
    epoch_error1 = 0
    epoch_error2 = 0
    valid_iteration = 0

    # ****************************************************************************************
    # Estimate computational information only once
    if epoch==1:
        flops, params = get_model_complexity_info(model, input_res=(3, opt.crop_height, opt.crop_width),
                                                  input_constructor=prepare_input,
                                                  as_strings=True,
                                                  print_per_layer_stat=True)  # make it true to see layer wise stats
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        writer.add_scalar('Computational complexity {} '.format(flops), float(flops[:4]))
        writer.add_scalar('Number of parameters {} '.format(params), float(params[:4]))

        wandb.log({'Computational complexity': flops,
                   'Number of parameters': params
                   })
    # ****************************************************************************************

    model.train()
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1],
                                                                                  requires_grad=True), Variable(
            batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            optimizer.zero_grad()

            if opt.model == 'GANet11':
                disp1, disp2 = model(input1, input2)

                disp0 = (disp1 + disp2) / 2.
                if opt.kitti or opt.kitti2015:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * criterion(
                        disp2[mask], target[mask])
                else:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * F.smooth_l1_loss(
                        disp2[mask], target[mask], reduction='mean')
            elif opt.model == 'GANet_deep':
                disp0, disp1, disp2 = model(input1, input2)
                if opt.kitti or opt.kitti2015:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                        disp1[mask], target[mask], reduction='mean') + criterion(disp2[mask], target[mask])
                else:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                        disp1[mask], target[mask], reduction='mean') + F.smooth_l1_loss(disp2[mask], target[mask],
                                                                                        reduction='mean')
            else:
                raise Exception("No suitable model found ...")

            loss.backward()
            optimizer.step()
            error0 = torch.mean(torch.abs(disp0[mask] - target[mask]))
            error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
            error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error0 += error0.item()
            epoch_error1 += error1.item()
            epoch_error2 += error2.item()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} {:.4f} {:.4f})".format(epoch, iteration,
                                                                                              len(training_data_loader),
                                                                                              loss.item(),
                                                                                              error0.item(),
                                                                                              error1.item(),
                                                                                              error2.item()))

            wandb.log({'train_iteration': iteration,
                       'train_iter_Loss': loss.item(),
                       'train_iter_Error0': error0.item(),
                       'train_iter_Error1': error1.item(),
                       'train_iter_Error2': error2.item(),
                       })

            sys.stdout.flush()
    # ****************************************************************************************
    writer.add_scalar('Avg. Training Loss', epoch_loss / valid_iteration, epoch)
    # wandb.log({'Avg_Training_loss': epoch_loss / valid_iteration
    #            })
    # ****************************************************************************************

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Error: ({:.4f} {:.4f} {:.4f})".format(epoch,
                                                                                                 epoch_loss / valid_iteration,
                                                                                                 epoch_error0 / valid_iteration,
                                                                                                 epoch_error1 / valid_iteration,
                                                                                                 epoch_error2 / valid_iteration))

    wandb.log({'train_Epoch': epoch,
               'Avg_train_Loss': epoch_loss / valid_iteration,
               'Avg_train_Error0': epoch_error0 / valid_iteration,
               'Avg_train_Error1': epoch_error1 / valid_iteration,
               'Avg_train_Error2': epoch_error2 / valid_iteration,
               })

def val(epoch):
    epoch_error2 = 0

    valid_iteration = 0
    model.eval()

    # ***********************************************
    if val_save_path:
        save_path = val_save_path + "_current_epoch"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
    # ***********************************************
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=False), Variable(batch[1],
                                                                                   requires_grad=False), Variable(
            batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            with torch.no_grad():
                disp2 = model(input1, input2)

                # ****************************************************************************************
                error_dict = calc_error(disp2, target)
                running_error_sum = {k: running_error_sum.get(k, 0) + error_dict.get(k, 0) for k in set(error_dict)}

                # TODO: Conditional saving of the validation results
                # to save a evaluate validation time qualitative results
                if val_save_path:
                    if opt.kitti or opt.kitti2015:
                        if epoch % 50 == 0 and epoch >= 100:
                            for i, temp in enumerate(disp2):
                                temp = temp.cpu()
                                temp = temp.detach().numpy()
                                savename = save_path + "/{}-{}-pred.png".format(str(iteration + 1), str(i))
                                skimage.io.imsave(savename, (temp * 256).astype('uint16'))
                                # temp = target[i, :, :].cpu()
                                # temp = temp.detach().numpy()
                                # savename = save_path + "/{}-{}-GT.png".format(str(iteration + 1), str(i))
                                # skimage.io.imsave(savename, (temp * 256).astype('uint16'))
                    else:
                        for i, temp in enumerate(disp2):
                            temp = temp.cpu()
                            temp = temp.detach().numpy()
                            savename = save_path + "/{}-{}-pred.png".format(str(iteration + 1), str(i))
                            skimage.io.imsave(savename, (temp * 256).astype('uint16'))
                # ****************************************************************************************

                error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                valid_iteration += 1
                epoch_error2 += error2.item()
                print("===> Test({}/{}): Error: ({:.4f})".format(iteration, len(testing_data_loader), error2.item()))

                wandb.log({
                    "val_iteration": iteration,
                    "val_iter_error2": error2.item()
                })

    # ****************************************************************************************
    writer.add_scalar('Avg. Validation Loss', epoch_error2 / valid_iteration, epoch)
    wandb.log({
        "validation epoch": epoch,
        "Avg_val_error2": epoch_error2/valid_iteration
    })

    tmp_dict = {k: running_error_sum.get(k, 0) / valid_iteration for k in set(running_error_sum)}
    for key, val in tmp_dict.items():
        writer.add_scalar('{} error rate'.format(key), val, epoch)
        wandb.log({
            "Avg_val {} error rate".format(key) : val,
            "validation epoch": epoch,
        })
    print("===> Test: Avg. Error dictionary: ({})".format(tmp_dict))

    # ****************************************************************************************
    print("===> Test: Avg. Error: ({:.4f})".format(epoch_error2 / valid_iteration))

    return epoch_error2 / valid_iteration


def save_checkpoint(save_path, epoch, state, is_best):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = save_path + "_epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + '_best.pth')
        print(f" Current best epoch is: {epoch}")
    print("Checkpoint saved to {}".format(filename))

    wandb.save(filename)

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 400:
        lr = opt.lr
    else:
        lr = opt.lr * 0.1
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_validation_results(epoch, is_best):
    if opt.kitti or opt.kitti2015:
        if epoch % 50 == 0:
            try:
                shutil.rmtree(val_save_path + "_{}_epoch".format(epoch), ignore_errors=False,
                              onerror=None)  # remove folder if it already exist
            except:
                pass

            shutil.copytree(val_save_path + '_current_epoch', val_save_path + "_{}_epoch".format(epoch))
            print("validation results saved to" + val_save_path + "_{}_epoch".format(epoch))
    else:
        try:
            shutil.rmtree(val_save_path + "_{}_epoch".format(epoch), ignore_errors=False,
                          onerror=None)  # remove folder if it already exist
        except:
            pass

        shutil.copytree(val_save_path + '_current_epoch', val_save_path + "_{}_epoch".format(epoch))
        print("validation results saved to" + val_save_path + "_{}_epoch".format(epoch))

    if is_best ==True:
        try:
            shutil.rmtree(val_save_path+"_best_epoch", ignore_errors=False, onerror=None)
        except:
            pass
        shutil.copytree(val_save_path+'_current_epoch', val_save_path+"_best_epoch")

if __name__ == '__main__':
    error = 100
    with wandb.init(project=opt.wandb_project_name , config= opt, settings=wandb.Settings(start_method='fork')):
        args= wandb.config #just to ensure same params are logged in wandb and also used same in our model
        wandb.run.name =args.wandb_run_name

        for epoch in range(opt.epoch_nums, opt.nEpochs + 1):
            #        if opt.kitti or opt.kitti2015:
            lr= adjust_learning_rate(optimizer, epoch)

            wandb.log({
                "learning_rate": lr,
                "Train_epoch": epoch
            })
            train(epoch)
            is_best = False
            # TODO: remove checking validation quantitative results every epoch in case of kitti, every 25th epoch
            loss = val(epoch)
            if loss < error:
                error = loss
                is_best = True

            if opt.kitti or opt.kitti2015:
                if epoch % 50 == 0 and epoch >= 100:
                    save_checkpoint(opt.save_path, epoch, {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best)

                    if val_save_path:
                        save_validation_results(epoch, is_best)

            else:
                if epoch >= 1:
                    save_checkpoint(opt.save_path, epoch, {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best)

                    if val_save_path:
                        save_validation_results(epoch, is_best)

        save_checkpoint(opt.save_path, opt.nEpochs, {
            'epoch': opt.nEpochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)