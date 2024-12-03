import os
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from PIL import Image
import glob
import random
import cv2
from models.SNR import *
from data.utils import *

class test_dataset(data.Dataset):
    def __init__(self, low_img_dir, high_img_dir, task, scale_factor, exp_mean):
        self.task = task
        self.scale_factor = scale_factor
        self.exp_mean = exp_mean
        self.train_low_data_names = []  # 初始化，训练图像文件名列表
        self.train_high_data_names = []

        self.train_low_data_names = glob.glob(low_img_dir + "*.*")  # 获取训练图像, 返回目录中的所有具有扩展名的文件
        if self.task == "test_pair":
            self.train_high_data_names = glob.glob(high_img_dir + "*.*")

        self.count = len(self.train_low_data_names)  # 数据集的样本数量

        self.low_data = []  # 存储低光图像的空列表
        self.high_data = []  # 存储低光图像的空列表

        for i in np.arange(self.count):  # 遍历每个样本
            low = load_images_transform(self.train_low_data_names[i])  # 根据当前索引 i 加载低、高光图像，并将图像数据转换到0-1的浮点数表示
            self.low_data.append(low)  # 将值缩放到[0,1], 并保存到列表中
            if self.task == "test_pair":
                high = load_images_transform(self.train_high_data_names[i])
                self.high_data.append(high)

    def __getitem__(self, index):

        exp_mean = 0.4
        low = self.low_data[index]  # 获取低光、高光 和 直方图 数据
        if self.task == "test_pair":
            high = self.high_data[index]
            exp_mean = calc_high_v_mean(high)
        elif self.task == "test_unpair":
            high = None
            exp_mean = self.exp_mean

        if self.task == "test_unpair":  # 防止某些图片过大
            height = (low.shape[0] // self.scale_factor) * self.scale_factor  # 确保 是scale_factor的倍数，保证模型正确执行
            width = (low.shape[1] // self.scale_factor) * self.scale_factor
            low = low[0:height, 0:width, :]

        img_name = self.train_low_data_names[index].split('\\')[-1]  # 提取图像文件名

        out_low = torch.from_numpy(low).float().permute(2, 0, 1)
        if self.task == "test_pair":
            out_high = torch.from_numpy(high).float().permute(2, 0, 1)

        if self.task == "test_pair":
            return out_low, out_high, exp_mean, img_name
        elif self.task == "test_unpair":
            return out_low, exp_mean, img_name

    def __len__(self):
        return self.count
