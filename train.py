# -*- coding = utf-8 -*-

# @time:2023/5/8 12:41

# Author:Cui


# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dataprocess.datautils3d as util3d
from evaluate import evaluateMulti
from image_pool import ImagePool
from lr_adj import update_learning_rate
from model.network import Generator, Discriminator, GD_Train
from model.utils import readTxtLineAsList, get_logger, weights_init, del_file, norm


def parse_option():
    parser = argparse.ArgumentParser('PMCCGAN training and evaluation script', add_help=False)
    parser.add_argument('--file_txt_dir', default="./dataset/split/", type=str,
                        help='path split txt file dir to dataset')
    parser.add_argument('--Ex_num', default=1, type=int, help='path split txt file dir to dataset')
    # dataset
    parser.add_argument('--data_l_cut_path', default='./dataset/LPET_cut', type=str,
                        help='LPET cut data path')
    parser.add_argument('--data_s_cut_path', default='./dataset/SPET_cut', type=str,
                        help='SPET cut data path')
    parser.add_argument('--data_mri_cut_path', default='./dataset/T1_cut', type=str,
                        help='T1 cut data path')
    # training
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1, type=int, help="batch size for single GPU")
    parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES var from this string')
    parser.add_argument('--lr_G', '--learning-rate-G', default=2e-4, type=float, help='initial learning rate of G')
    parser.add_argument('--lr_D', '--learning-rate-D', default=2e-4, type=float, help='initial learning rate of D')
    parser.add_argument('--lamb', default=100, type=float, dest='lamb')
    parser.add_argument('--beta1', default=0.9, type=float, dest='beta1')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--arc', default='Proposed', type=str, help='Network architecture')
    # logs
    parser.add_argument('--log_dir', default="./results/pretrained_model/", type=str, help='logger information dir')
    # validation
    parser.add_argument('--val_epoch_inv', default=1, type=int, help='validation interval epochs')
    # use checkpoint
    parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint during training')
    parser.add_argument('--checkpoint_file_path', default="./check_point.pkl", type=str, help='pretrained weight path')
    # use pretrained model
    parser.add_argument('--pretrained_model', action='store_true', help='pretrained weight path')
    parser.add_argument('--pretrained_model_dir', default="./results/pretrained_model", type=str,
                        help='pretrained weight path')
    parser.add_argument('--temp_val_img_cut_dir', default="./temp/", type=str, help='validation image cut path')

    args, unparsed = parser.parse_known_args()

    return args


def train(args):
    """Del and create result folders"""
    if not os.path.exists(args.pretrained_model_dir):
        os.makedirs(args.pretrained_model_dir)
    if not os.path.exists(args.temp_val_img_cut_dir):
        os.makedirs(args.temp_val_img_cut_dir)
    else:
        del_file(args.temp_val_img_cut_dir)

    """Dataset"""
    train_txt_path = args.file_txt_dir + r"Ex" + str(args.Ex_num) + r"/train.txt"
    val_txt_path = args.file_txt_dir + r"Ex" + str(args.Ex_num) + r"/val.txt"
    train_imgs = readTxtLineAsList(train_txt_path)
    val_imgs = readTxtLineAsList(val_txt_path)
    trainloader = util3d.loadMultiData(args.data_l_cut_path, '', args.data_s_cut_path, '', args.data_mri_cut_path, '',
                                       prefixs=train_imgs, batch_size=args.batch_size, shuffle=True)
    valloader = util3d.loadMultiData(args.data_l_cut_path, '', args.data_s_cut_path, '', args.data_mri_cut_path, '',
                                     prefixs=val_imgs, batch_size=1, shuffle=False)

    device = torch.device('cuda:' + args.devices if torch.cuda.is_available() else "cpu")

    imgpool = ImagePool(5)

    """Hyper-parameters"""
    lr_G, lr_D = args.lr_G, args.lr_D
    beta1 = args.beta1
    lamb = args.lamb  #
    epochs = args.epochs
    start_epoch = 0

    """Network"""
    G = Generator(layers=[1, 1, 1, 1],  # 2, 2, 2, 2
                  embed_dims=[64, 128, 256, 512],  # 64, 128, 256, 512
                  mlp_ratios=[8, 8, 4, 4],
                  heads=[4, 4, 8, 8], head_dim=[24, 24, 24, 24],
                  type=args.arc).to(device)
    D = Discriminator().to(device)

    L1 = nn.L1Loss().to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, 0.999))

    """Loggings"""""
    log_dir = args.log_dir + r"Ex" + str(args.Ex_num) + "/" + str(args.arc) + "_lrG_" + '{:g}'.format(
        args.lr_G) + "_lamb_"'{:g}'.format(lamb)
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_full_path = log_dir + '/' + time_str + '.log'
    logger = get_logger(log_full_path)

    """Training"""
    D_Loss, G_Loss, Epochs = [], [], range(1, epochs + 1)
    PSNR_val_best, PSNR_val_epoch_best = start_epoch, start_epoch
    torch.cuda.empty_cache()
    # init
    G.apply(weights_init)
    D.apply(weights_init)

    for epoch in range(start_epoch, epochs):
        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0
        for i, (x, y) in enumerate(trainloader):
            X = norm(x)
            Y = norm(y)

            d_loss, g_loss = GD_Train(D, G, X, Y, optimizer_G, optimizer_D, L1, device, imgpool, lamb=lamb)
            D_losses.append(d_loss)
            G_losses.append(g_loss)
            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
            print('[%d / %d]: batch#%d loss_d= %.6f  loss_g= %.6f lr_g=%.6f lr_d=%.6f' %
                  (epoch + 1, epochs, i, d_l, g_l, optimizer_G.state_dict()['param_groups'][0]['lr'],
                   optimizer_D.state_dict()['param_groups'][0]['lr']))

        D_Loss.append(d_l)
        G_Loss.append(g_l)
        logger.info("Train => Epoch:{} Avg.D_Loss:{} Avg.G_Loss:{}".format(epoch, d_l, g_l))

        torch.save(G, os.path.join(log_dir, 'last_model.pkl'))

        # schedulerG.step()
        # schedulerD.step()
        if epoch > 50:
            update_learning_rate(optimizer_G)
            update_learning_rate(optimizer_D)

        # save and update check_point
        if args.use_checkpoint:
            checkpoint = {
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }
            torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pkl'))

        # validate every val_epoch_inv
        if epoch % args.val_epoch_inv == 0:
            logger.info('Validation => Epoch {}'.format(epoch))
            ave_psnr, ave_ssim, ave_mnse = evaluateMulti(G, valloader, device=device)
            if ave_psnr >= PSNR_val_best:
                PSNR_val_best, PSNR_val_epoch_best = ave_psnr, epoch
                torch.save(G.state_dict(), os.path.join(log_dir, str(args.Ex_num) + '_best_PSNR_model.pkl'))
                logger.info('Val => Model saved for better PSNR!')
                logger.info('Val => Epoch:{} PSNR:{:.6f} SSIM:{:.6f} MRSE:{:.6f}'.format(
                    epoch + 1, ave_psnr, ave_ssim, ave_mnse))
                logger.info('Val => Best PSNR(Epoch:{}):{:.6f}'.format(PSNR_val_epoch_best, PSNR_val_best))
            else:
                logger.info('Val => Model NOT saved for better PSNR!')
                logger.info('Val => Epoch:{} PSNR:{:.6f} SSIM:{:.6f} MRSE:{:.6f}'.format(
                    epoch + 1, ave_psnr, ave_ssim, ave_mnse))
                logger.info('Val => Best PSNR(Epoch:{}):{:.6f} '.format(PSNR_val_epoch_best, PSNR_val_best))


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    args = parse_option()
    print(args)
    train(args)
