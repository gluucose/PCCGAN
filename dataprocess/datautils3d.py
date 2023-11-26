# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:55:51 2021

@author: AruZeng
"""
import os

import numpy as np
import torch
from medpy.io import load
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, root_l, subfolder_l, root_s, subfolder_s, prefixs, transform=None):
        super(MyDataset, self).__init__()
        self.prefixs = prefixs
        self.l_path = os.path.join(root_l, subfolder_l)
        self.s_path = os.path.join(root_s, subfolder_s)
        self.templ = [x for x in os.listdir(self.l_path) if os.path.splitext(x)[1] == ".img"]
        self.temps = [x for x in os.listdir(self.s_path) if os.path.splitext(x)[1] == ".img"]
        self.image_list_l = []
        self.image_list_s = []
        # 找指定前缀的数据
        for file in self.templ:
            for pre in prefixs:
                if pre in file:
                    self.image_list_l.append(file)
        # 找指定前缀的数据
        for file in self.temps:
            for pre in prefixs:
                if pre in file:
                    self.image_list_s.append(file)
        # print(self.image_list_l)
        # print(self.image_list_s)
        self.transform = transform

    def __len__(self):
        return len(self.image_list_l)

    def __getitem__(self, item):
        # 读图片（低剂量PET）
        image_path_l = os.path.join(self.l_path, self.image_list_l[item])
        # image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
        image_l, h = load(image_path_l)
        image = np.array(image_l)
        # print(image.shape)
        if self.transform is not None:
            image = self.transform(image_l)
        # 读标签(高质量PET)
        image_path_s = os.path.join(self.s_path, self.image_list_s[item])
        image_s, h2 = load(image_path_s)
        image_s = np.array(image_s)
        # print(image_l.shape)0
        # print(image_path_l,image_path_s)
        # 添加通道维度
        image_l = image_l[np.newaxis, :]
        image_s = image_s[np.newaxis, :]
        image_l = torch.Tensor(image_l)
        image_s = torch.Tensor(image_s)
        # print(image.shape)
        if self.transform is not None:
            image = self.transform(image_s)
        # 返回：影像，标签
        return image_l, image_s


class MyMultiDataset(Dataset):
    def __init__(self, root_l, subfolder_l, root_s, subfolder_s, root_mri, subfolder_mri, prefixs, transform=None):
        super(MyMultiDataset, self).__init__()
        self.prefixs = prefixs
        self.l_path = os.path.join(root_l, subfolder_l)
        self.s_path = os.path.join(root_s, subfolder_s)
        self.mri_path = os.path.join(root_mri, subfolder_mri)
        self.templ = [x for x in os.listdir(self.l_path) if os.path.splitext(x)[1] == ".img"]
        self.temps = [x for x in os.listdir(self.s_path) if os.path.splitext(x)[1] == ".img"]
        self.temp_mri = [x for x in os.listdir(self.mri_path) if os.path.splitext(x)[1] == ".img"]
        self.image_list_l = []
        self.image_list_s = []
        self.image_list_mri = []
        # 找指定前缀的数据
        for file in self.templ:
            for pre in prefixs:
                if pre in file:
                    self.image_list_l.append(file)
        # 找指定前缀的数据
        for file in self.temps:
            for pre in prefixs:
                if pre in file:
                    self.image_list_s.append(file)
        # 找指定前缀的数据
        for file in self.temp_mri:
            for pre in prefixs:
                if pre in file:
                    self.image_list_mri.append(file)
        # print(self.image_list_l)
        # print(self.image_list_s)
        self.transform = transform

    def __len__(self):
        return len(self.image_list_l)

    def __getitem__(self, item):
        # 读图片（低剂量PET）
        image_path_l = os.path.join(self.l_path, self.image_list_l[item])
        # image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
        image_l, h = load(image_path_l)
        image = np.array(image_l)
        # print(image.shape)
        if self.transform is not None:
            image = self.transform(image_l)
        # 读标签(高质量PET)
        image_path_s = os.path.join(self.s_path, self.image_list_s[item])
        image_s, h2 = load(image_path_s)
        image_s = np.array(image_s)
        # 读对于MRI
        image_path_mri = os.path.join(self.mri_path, self.image_list_mri[item])
        image_mri, h3 = load(image_path_mri)
        image_mri = np.array(image_mri)
        #
        # print(image_l.shape)0
        # print(image_path_l,image_path_s)
        # 添加通道维度
        image_l = image_l[np.newaxis, :]
        image_s = image_s[np.newaxis, :]
        image_mri = image_mri[np.newaxis, :]
        image_l = torch.Tensor(image_l)
        image_s = torch.Tensor(image_s)
        image_mri = torch.Tensor(image_mri)
        # print(image.shape)
        if self.transform is not None:
            image = self.transform(image_s)
        # 返回：影像，标签,mri
        return image_l, image_s, image_mri


# data
def loadData(root1, subfolder1, root2, subfolder2, prefixs, batch_size, shuffle=True):
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    '''
    transform = None
    # 测试已修改
    dataset = MyDataset(root1, subfolder1, root2, subfolder2, prefixs, transform=transform)
    # dataset = MyDataset(root, subfolder,transform=None)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# multi data
def loadMultiData(root1, subfolder1, root2, subfolder2, root3, subfolder3, prefixs, batch_size, shuffle=True):
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W) & (0, 255) -> (0, 1)
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1)
    ])
    '''
    transform = None
    dataset = MyMultiDataset(root1, subfolder1, root2, subfolder2, root3, subfolder3, prefixs, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def readTxtLineAsList(txt_path):
    fi = open(txt_path, 'r')
    txt = fi.readlines()
    res_list = []
    for w in txt:
        w = w.replace('\n', '')
        res_list.append(w)
    return res_list


if __name__ == '__main__':
    train_txt_path = r"D:\PyCharmProjects\CoCs3D\dataset\split\Ex1\train.txt"
    val_txt_path = r"D:\PyCharmProjects\CoCs3D\dataset\split\Ex1\val.txt"
    train_imgs = readTxtLineAsList(train_txt_path)
    print(train_imgs)
    val_imgs = readTxtLineAsList(val_txt_path)
    print(val_imgs)
    trainloader = loadMultiData('D:\PyCharmProjects\CoCs3D\dataset\processed\LPET_cut', '',
                                'D:\PyCharmProjects\CoCs3D\dataset\processed\SPET_cut', '',
                                'D:\PyCharmProjects\CoCs3D\dataset\processed\T1_cut', '',
                                prefixs=train_imgs, batch_size=4)
    print(len(trainloader))
    print("=================================")
    valloader = loadMultiData('D:\PyCharmProjects\CoCs3D\dataset\processed\LPET_cut', '',
                              'D:\PyCharmProjects\CoCs3D\dataset\processed\SPET_cut', '',
                              'D:\PyCharmProjects\CoCs3D\dataset\processed\T1_cut', '',
                              prefixs=val_imgs, batch_size=4)

    for i, (x, y, mri) in enumerate(trainloader):
        print(i)
