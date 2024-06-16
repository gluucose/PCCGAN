# -*- coding = utf-8 -*-

import SimpleITK as sitk
import numpy as np
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model.utils import norm


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
        y = np.nonzero(image_s)
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
