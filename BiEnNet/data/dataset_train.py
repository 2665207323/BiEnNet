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

class train_dataset(data.Dataset):
    def __init__(self, low_img_dir, high_img_dir, task, batch_w, batch_h):
        self.task = task
        self.train_low_data_names = []  # 初始化，训练图像文件名列表
        self.train_high_data_names = []

        self.batch_w = batch_w
        self.batch_h = batch_h

        self.train_low_data_names = glob.glob(low_img_dir + "*.*")  # 获取训练图像, 返回目录中的所有具有扩展名的文件
        self.train_high_data_names = glob.glob(high_img_dir + "*.*")

        self.count = len(self.train_low_data_names)  # 数据集的样本数量

        self.low_data = []  # 存储低光图像的空列表
        self.high_data = []  # 存储低光图像的空列表

        for i in np.arange(self.count):  # 遍历每个样本
            low = load_images_transform(self.train_low_data_names[i])  # 根据当前索引 i 加载低、高光图像，并将图像数据转换到0-1的浮点数表示
            high = load_images_transform(self.train_high_data_names[i])
            self.low_data.append(low)  # 将值缩放到[0,1], 并保存到列表中
            self.high_data.append(high)

    def __getitem__(self, index):

        low = self.low_data[index]  # 获取低光、高光 和 直方图 数据
        high = self.high_data[index]

        h = low.shape[0]  # 获取低光图像的高 和 宽,  为 NumPy数组
        w = low.shape[1]

        h_offset = random.randint(0, max(0, h - self.batch_h))  # 随机生成裁剪区域的起始位置，确保裁剪区域不超出图像边界
        w_offset = random.randint(0, max(0, w - self.batch_w))

        if self.task == 'train':  # train 任务时，裁剪低光和高光图像
            low = low[h_offset:h_offset + self.batch_h, w_offset:w_offset + self.batch_w]
            high = high[h_offset:h_offset + self.batch_h, w_offset:w_offset + self.batch_w]
            low, high = image_transforms(low, high)  # 数据增强
        else:
            pass

        exp_mean = calc_high_v_mean(high)

        img_name = self.train_low_data_names[index].split('\\')[-1]  # 提取图像文件名

        out_low = torch.from_numpy(low).float().permute(2, 0, 1)
        out_high = torch.from_numpy(high).float().permute(2, 0, 1)

        return out_low, out_high, exp_mean, img_name 

    def __len__(self):
        return self.count







