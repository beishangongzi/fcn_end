# encoding: utf-8
"""
@author: FroyoZzz
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: froyorock@gmail.com
@software: garner
@file: dataset.py
@time: 2019-08-07 17:21
@desc:
"""
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, image_path="data/obt/image", mode="train"):
        assert mode in ("train", "val", "test")
        self.image_path = image_path
        self.image_list = glob(os.path.join(self.image_path, "*.npy"))
        self.mode = mode

        if mode in ("train", "val"):
            self.mask_path = self.image_path + "Masks"

        self.transform_x = T.Compose([T.ToTensor()])
        self.transform_mask = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            image_name = os.path.basename(self.image_list[index])
            X = np.load(self.image_list[index])
            X = X.astype(int)
            X = self.transform_x(X)
            X = X.to(torch.float)
            mask = np.load(os.path.join(self.mask_path, image_name))
            new_y = np.zeros([256, 256, 5])
            new_y[:, :, 0] = (mask == 0).reshape(256, 256)
            new_y[:, :, 1] = (mask == 1).reshape(256, 256)
            new_y[:, :, 2] = (mask == 2).reshape(256, 256)
            new_y[:, :, 3] = (mask == 3).reshape(256, 256)
            new_y[:, :, 4] = (mask == 4).reshape(256, 256)
            mask = self.transform_mask(new_y)

            return X, mask

        else:
            X = Image.open(self.image_list[index])
            X = self.transform_x(X)
            path = self.image_list[index]
            return X, path

    def __len__(self):
        return len(self.image_list)
