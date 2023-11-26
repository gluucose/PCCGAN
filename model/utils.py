"""
Some helper functions.
"""

import argparse
import logging
import os

import SimpleITK as sitk
import numpy as np
import torch.nn as nn
from PIL import Image
from thop import profile


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def readTxtLineAsList(txt_path):
    fi = open(txt_path, 'r')
    txt = fi.readlines()
    res_list = []
    for w in txt:
        w = w.replace('\n', '')
        res_list.append(w)
    return res_list


def save_image(image_numpy, image_path):
    savImg = sitk.GetImageFromArray(image_numpy[:, :, :])
    sitk.WriteImage(savImg, image_path)


def weights_init(m):
    if isinstance(m, nn.Linear):
        # logging.info('=> init weight of Linear from xavier uniform')
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # logging.info('=> init bias of Linear to zeros')
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def print_model_parm_nums(model, x):
    flops, params = profile(model, inputs=(x,))
    print('  + FLOPs: %.2fGFLOPs' % (flops / 1024 / 1024 / 1024),
          '  + Params: %.2fM' % (params / 1024 / 1024))


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def norm(x):
    X = (x - x.min()) / (x.max() - x.min())
    return X


def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


def MatrixToImage(data):
    if (data.max() > 2):
        data = (data - data.min()) / (data.max() - data.min())
    data = data * 255
    # data=np.flipud(data)
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


if __name__ == '__main__':
    pass
