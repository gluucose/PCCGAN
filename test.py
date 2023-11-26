# -*- coding = utf-8 -*-

# @time:2023/5/8 12:41

# Author:Cui

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import shutil

import SimpleITK as sitk
import numpy as np
import torch
from medpy.io import load
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import dataprocess.datautils3d as util3d
from model.network import Generator
from model.utils import readTxtLineAsList, norm


def predict_patches(val_imgs, valloader, pretrained_model):
    model = torch.load(pretrained_model, map_location='cpu')
    model.eval()
    model = Generator(layers=[1, 1, 1, 1],
                      embed_dims=[64, 128, 256, 512],
                      mlp_ratios=[8, 8, 4, 4],
                      heads=[4, 4, 8, 8], head_dim=[24, 24, 24, 24]).to(device)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'), False)
    model.eval()

    for i, (image_l, image_s, mri) in enumerate(valloader):
        image_l = norm(image_l).to(device)
        image_s = norm(image_s)
        image_s = np.squeeze(image_s.detach().numpy())
        image_mri = norm(mri).to(device)

        res = model(image_l, image_mri)
        # res = image_l  # for test only
        res = res.cpu().detach().numpy()
        res = np.squeeze(res)

        # save the predicted patches
        savImg = sitk.GetImageFromArray(res)
        filename = f'cut_{i:04d}' + '.img'
        savepath = './imgs/predicted_' + val_imgs[0] + '_patches'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        sitk.WriteImage(savImg, savepath + '/' + filename)
        # save the real patches
        savImg = sitk.GetImageFromArray(image_s)
        filename = f'cut_{i:04d}' + '.img'
        savepath = './imgs/real_' + val_imgs[0] + '_patches'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        sitk.WriteImage(savImg, savepath + '/' + filename)


def concat(path, savedir, val_imgs):
    all_names = []
    for root, dirs, files in os.walk(path):
        all_names = (files)
    all_name = []
    for i in all_names:
        if os.path.splitext(i)[1] == ".img":
            # print(i)
            all_name.append(i)
    # all_name=all_name.sort(key=lambda k:(int(k[-7:-4])))
    # print(all_name, len(all_name))
    cnt = 0
    clips = []
    stride_3d = [16, 16, 16]
    window_3d = [64, 64, 64]
    for file in all_name:
        # print('file: ', file)
        image_path = os.path.join(path, file)
        image, h = load(image_path)
        image = np.array(image)
        # image = np.moveaxis(image, [0, 1, 2], [2, 1, 0])
        clips.append(image)
        if (len(clips) == 125):  # 729 for 8
            cnt = 0
            s_d, s_h, s_w = stride_3d
            w_d, w_h, w_w = window_3d
            counter = np.zeros([128, 128, 128])
            D, H, W = counter.shape
            num_d = (D - w_d) // s_d + 1
            num_h = (H - w_h) // s_h + 1
            num_w = (W - w_w) // s_w + 1
            res_collect = np.zeros([128, 128, 128])
            # print(num_d, num_h, num_w)
            for i in range(num_d):
                for j in range(num_h):
                    for k in range(num_w):
                        counter[i * s_d:i * s_d + w_d, j * s_h:j * s_h + w_h, k * s_w:k * s_w + w_w] += 1
                        x = clips[cnt]
                        cnt += 1
                        res_collect[i * s_d:i * s_d + w_d, j * s_h:j * s_h + w_h, k * s_w:k * s_w + w_w] += x
            res_collect /= counter
            res = res_collect
            cnt = 0
            clips = []
            res = np.moveaxis(res, [0, 1, 2], [2, 1, 0])
            y = np.where(res < 0.01)
            res[y] = 0.0

            if not os.path.exists(savedir):
                os.makedirs(savedir)
            savImg = sitk.GetImageFromArray(res)
            sitk.WriteImage(savImg, savedir + '/' + val_imgs[0] + '.img')


def cal(predicted_path, real_path):
    res, h = load(predicted_path)
    image_s, h = load(real_path)
    if res.max() > 1:
        res = norm(res)
    if image_s.max() > 1:
        image_s = norm(image_s)

    y = np.nonzero(image_s)
    image_s_1 = image_s[y]
    res_1 = res[y]
    # calculate PSNR
    cur_psnr = psnr(image_s_1, res_1, data_range=1.)
    cur_ssim = ssim(res, image_s, multichannel=True)
    cur_nmse = nmse(image_s, res) ** 2
    print("PSNR: %.6f   SSIM: %.6f   NMSE: %.6f" % (cur_psnr, cur_ssim, cur_nmse))


def run():
    val_txt_path = file_txt_dir + r"Ex" + str(Ex_num) + r"/val.txt"
    val_imgs = readTxtLineAsList(val_txt_path)

    valloader = util3d.loadMultiData(data_l_cut_path, '', data_S_cut_path, '',
                                     data_mri_cut_path, '', prefixs=val_imgs,
                                     batch_size=1, shuffle=False)

    """predict patches"""
    print('============= predict patches ===================')
    predict_patches(val_imgs, valloader, pretrained_model)
    """concat patches"""
    print('=========== concat predicted patches =============')
    # predicted pathces
    predicted_path = './imgs/predicted_' + val_imgs[0] + '_patches'
    save_path_f = './imgs/predicted_' + val_imgs[0]
    concat(predicted_path, save_path_f, val_imgs)
    # real patches
    real_path = './imgs/real_' + val_imgs[0] + '_patches'
    save_path_r = './imgs/real_' + val_imgs[0]
    concat(real_path, save_path_r, val_imgs)
    """calculate metrics"""
    print('============= calculate metrics ==================')
    save_path_fake = save_path_f + '/' + val_imgs[0] + '.img'
    save_path_real = save_path_r + '/' + val_imgs[0] + '.img'
    cal(save_path_real, save_path_fake)

    shutil.rmtree(predicted_path)
    shutil.rmtree(real_path)
    shutil.rmtree(save_path_r)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_l_cut_path = r'./dataset/LPET_cut'
    data_S_cut_path = r'./dataset/SPET_cut'
    data_mri_cut_path = r'./dataset/T1_cut'
    file_txt_dir = r'./dataset/split/'

    Ex_num = 1
    pretrained_model = r''

    run()
