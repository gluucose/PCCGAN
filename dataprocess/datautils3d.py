# -*- coding: utf-8 -*-

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

        for file in self.templ:
            for pre in prefixs:
                if pre in file:
                    self.image_list_l.append(file)

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
        # LPET
        image_path_l = os.path.join(self.l_path, self.image_list_l[item])
        image_l, h = load(image_path_l)
        image = np.array(image_l)

        if self.transform is not None:
            image = self.transform(image_l)

        # SPET
        image_path_s = os.path.join(self.s_path, self.image_list_s[item])
        image_s, h2 = load(image_path_s)
        image_s = np.array(image_s)

        image_l = image_l[np.newaxis, :]
        image_s = image_s[np.newaxis, :]
        image_l = torch.Tensor(image_l)
        image_s = torch.Tensor(image_s)
        # print(image.shape)
        if self.transform is not None:
            image = self.transform(image_s)

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

        for file in self.templ:
            for pre in prefixs:
                if pre in file:
                    self.image_list_l.append(file)

        for file in self.temps:
            for pre in prefixs:
                if pre in file:
                    self.image_list_s.append(file)

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
        # LPET
        image_path_l = os.path.join(self.l_path, self.image_list_l[item])
        image_l, h = load(image_path_l)
        image = np.array(image_l)

        if self.transform is not None:
            image = self.transform(image_l)

        # SPET
        image_path_s = os.path.join(self.s_path, self.image_list_s[item])
        image_s, h2 = load(image_path_s)
        image_s = np.array(image_s)
        # MRI
        image_path_mri = os.path.join(self.mri_path, self.image_list_mri[item])
        image_mri, h3 = load(image_path_mri)
        image_mri = np.array(image_mri)

        image_l = image_l[np.newaxis, :]
        image_s = image_s[np.newaxis, :]
        image_mri = image_mri[np.newaxis, :]
        image_l = torch.Tensor(image_l)
        image_s = torch.Tensor(image_s)
        image_mri = torch.Tensor(image_mri)

        if self.transform is not None:
            image = self.transform(image_s)

        return image_l, image_s, image_mri


def loadData(root1, subfolder1, root2, subfolder2, prefixs, batch_size, shuffle=True):
    transform = None
    dataset = MyDataset(root1, subfolder1, root2, subfolder2, prefixs, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def loadMultiData(root1, subfolder1, root2, subfolder2, root3, subfolder3, prefixs, batch_size, shuffle=True):
    transform = None
    dataset = MyMultiDataset(root1, subfolder1, root2, subfolder2, root3, subfolder3, prefixs, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    pass
