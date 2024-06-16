# -*- coding = utf-8 -*-

import os

import SimpleITK as sitk
import numpy as np
from medpy.io import load


def Datamake(root):
    all_names = []
    for root, dirs, files in os.walk(root):
        all_names = (files)

    all_name = []
    for i in all_names:
        if os.path.splitext(i)[1] == ".img":
            all_name.append(i)
    # print(all_name)

    # create result folder
    res_dir = root + '_cut'
    folder = os.path.exists(res_dir)
    if not folder:
        os.makedirs(res_dir)

    for file in all_name:
        image_path_mri = os.path.join(root, file)
        image_mri, h = load(image_path_mri)
        image_mri = np.array(image_mri)
        # print(image_l.shape)
        cut_cnt = 0
        for i in range(0, 5):
            for j in range(0, 5):
                for k in range(0, 5):
                    image_cut = image_mri[16 * i:64 + 16 * i, 16 * j:64 + 16 * j, 16 * k:64 + 16 * k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg, res_dir + '/' + file + '_cut' + f'{cut_cnt:03d}' + '.img')
                    cut_cnt += 1


if __name__ == '__main__':
    Datamake('./dataset/')
