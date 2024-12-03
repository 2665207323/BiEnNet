import os
import math
import pickle
import random
import numpy as np
import torch
import cv2
from PIL import Image


def load_images_transform(file):
    data_lowlight = Image.open(file)  # 打开文件 图像通道顺序为：RGB
    data_lowlight = (np.asarray(data_lowlight) / 255.0)  # 将图像数据转换成numpy数组, 并/255.0缩放到[0,1]   [h,w,3]
    return data_lowlight


def image_transforms(low, high):  # 数据增强
    if random.random() > 0.5:
        low = np.fliplr(low)
        high = np.fliplr(high)

    if random.random() > 0.5:
        low = np.flipud(low)
        high = np.flipud(high)

    rotate_degree = random.randint(0, 3)
    if rotate_degree > 0:
        low = np.rot90(low, k=rotate_degree)
        high = np.rot90(high, k=rotate_degree)

    # 在返回之前，确保图像的排列方式是正确的
    low = low.copy()
    high = high.copy()

    return low, high


def calc_high_v_mean(high_image):  # 计算参考图像的期望均值μ

    high_im_filter_max = np.max(high_image, axis=2, keepdims=True)  # [h,w,1]
    high_v_mean = high_im_filter_max.mean()  # 计算参考图像的期望均值μ

    return high_v_mean


def calc_hist(low_images, nbins, exp_mean):  # 低光图像的亮度直方图

    low_images = low_images.detach().cpu().numpy()  # 将Tensor转换为Numpy类型
    exp_mean = exp_mean.detach().cpu().numpy()

    B, _, _, _ = low_images.shape
    hist_low_v_batch = np.zeros((B, nbins+1, 1, 1), dtype=np.float32)

    for b in range(B):
        low_image = low_images[b]
        # 计算 低光图像的亮度直方图
        low_im_filter_max = np.max(low_image, axis=0, keepdims=True)  # positive  通道维度上 - 低光图像在每个像素位置上的 最大亮度值

        # 获取低光图像的亮度直方图hist1 和 各个区间的边界值bin
        hist1, bins = np.histogram(low_im_filter_max, bins=nbins-2, range=(np.min(low_im_filter_max), np.max(low_im_filter_max)))

        hist2 = np.reshape(hist1, [1, nbins - 2, 1, 1])  # 将直方图hist1 重塑为一个大小为 [1, 1, nbins-2]的数组
        hist_low_v = np.zeros([1, int(nbins + 1), 1, 1])  # 创建 hist数组，存储直方图 [1, 1, bins+1], 多的一个存储高光图像的 V通道的均值
        hist_low_v[:, 0:nbins - 2, :, :] = hist2 / np.sum(hist2)  # 填充hist，同时进行归一化, 使得直方图中所有的bin之和为 1
        hist_low_v[:, nbins - 2:nbins - 1, :, :] = np.min(low_im_filter_max)  # 倒数第三个存储最小值
        hist_low_v[:, nbins - 1:nbins, :, :] = np.max(low_im_filter_max)  # 倒数第二个存储最大值
        hist_low_v[:, -1, :, :] = exp_mean[b]

        hist_low_v_batch[b] = hist_low_v


    hist_low_v_batch = torch.from_numpy(hist_low_v_batch).float()

    return hist_low_v_batch


def calc_low_denoise(low_images):

    low_images = low_images.detach().cpu().numpy()  # 将Tensor转换为Numpy类型
    B, _, _, _ = low_images.shape  # [b, 3, h, w]
    low_denoise_batch = np.zeros_like(low_images, dtype=np.float32)

    for b in range(B):
        low_image = low_images[b]  # 选择每张图像  [3, h, w]
        # 对低光图像进行预处理，使用 cv2.blur() 对低光图像进行去噪
        low_denoise = cv2.blur( low_image.transpose(1, 2, 0), (5, 5) ).transpose(2, 0, 1)  # [3, h, w]
        low_denoise_batch[b] = low_denoise

    low_denoise_batch = low_denoise_batch * 1.0 / 255.0  # 将图像缩放到(0, 1)

    low_denoise_batch = torch.from_numpy(low_denoise_batch).float()

    return low_denoise_batch
