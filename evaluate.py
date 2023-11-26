# -*- coding = utf-8 -*-

# @time:2023/5/8 12:41

# Author:Cui

import SimpleITK as sitk
import numpy as np
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model.utils import norm


def evaluate(model, valloader, device, ex_num, epoch, i_start=30):
    model.eval()
    total_psnr, total_nmse, i_start = [], [], i_start
    # predict & save
    predicted_image = np.zeros([128, 128, 128])
    target_image = np.zeros([128, 128, 128])
    for ii, (x, y) in enumerate(valloader):
        x = norm(x).to(device)
        z = norm(z).to(device)

        esino, epet = model(x)

        predicted_image[i_start, :, :] = np.squeeze(epet.cpu().detach().numpy())
        target_image[i_start, :, :] = np.squeeze(z.cpu().detach().numpy())
        abovez = np.where(predicted_image < 0.0)
        predicted_image[abovez] = 0.0

        i_start += 1

    predicted_image_c = np.squeeze(predicted_image)
    target_image_c = np.squeeze(target_image)
    nz = np.nonzero(target_image_c)
    predicted_image_1 = predicted_image_c[nz]
    target_image_1 = target_image_c[nz]

    cur_psnr = psnr(predicted_image_1, target_image_1, data_range=1)
    cur_ssim = ssim(predicted_image_1, target_image_1, multi_channel=1)
    cur_nmse = nmse(predicted_image, target_image) ** 2
    # save images
    saveimg = sitk.GetImageFromArray(predicted_image)
    sitk.WriteImage(saveimg, './results/training/' + 'epet_' + str(ex_num) + '_epoch_' + str(epoch + 1) + '.img')
    if epoch == 0:
        saveimg = sitk.GetImageFromArray(target_image)
        sitk.WriteImage(saveimg, './results/training/' + 'spet_' + str(ex_num) + '_epoch_' + '.img')

    return cur_psnr, cur_ssim, cur_nmse


def evaluateMulti(G, valloader, device):
    G.eval()
    PSNR_vals, SSIM_vals, NMSE_vals = list(), list(), list()
    for image_l, image_s, mri in valloader:
        testl = image_l
        image_l = norm(testl).to(device)
        tests = image_s
        image_s = norm(tests).to(device)
        testmri = mri
        testmri = norm(testmri).to(device)
        image_s = np.squeeze(image_s.cpu().detach().numpy())

        res = G(image_l, testmri)
        res = res.cpu().detach().numpy()
        res = np.squeeze(res)

        image_l = image_l.cpu().detach().numpy()
        image_l = np.squeeze(image_l)
        y = np.nonzero(image_s)  # 取非黑色部分
        image_s_1 = image_s[y]
        res_1 = res[y]
        # cal PSNR
        cur_psnr = (psnr(res_1, image_s_1, data_range=1))
        # cal ssim
        cur_ssim = ssim(res, image_s, multichannel=True)
        # cal mrse
        cur_nmse = nmse(image_s, res) ** 2

        PSNR_vals.append(cur_psnr)
        SSIM_vals.append(cur_ssim)
        NMSE_vals.append(cur_nmse)

    cur_mean_PSNR_val, cur_mean_SSIM_val, cur_mean_NMSE_val = \
        np.mean(PSNR_vals), np.mean(SSIM_vals), np.mean(NMSE_vals)

    return cur_mean_PSNR_val, cur_mean_SSIM_val, cur_mean_NMSE_val
