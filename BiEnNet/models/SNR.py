import torch
import cv2
from torch import nn
import numpy as np


def SNR_mask(low, low_denoise):
    # 颜色通道转换，获得灰度图 Ig --- RGB图像 --> 灰度图像， 并使用标准的加权平均方法(0.299 * R + 0.587 * G + 0.114 * B)
    low = low[:, 0:1, :, :] * 0.299 + low[:, 1:2, :, :] * 0.587 + low[:, 2:3, :, :] * 0.114 \
          + low[:, 3:4, :, :] * 0.299 + low[:, 4:5, :, :] * 0.587 + low[:, 5:6, :, :] * 0.114
    # 对去噪后的图像同样进行通道转换
    low_denoise = low_denoise[:, 0:1, :, :] * 0.299 + low_denoise[:, 1:2, :, :] * 0.587 + low_denoise[:, 2:3, :, :] * 0.114 \
                  + low_denoise[:, 3:4, :, :] * 0.299 + low_denoise[:, 4:5, :, :] * 0.587 + low_denoise[:, 5:6, :, :] * 0.114

    '''计算低光灰度图Ig 和 去噪后的低光灰度图I^g 之间的差异，得到噪声图 N'''
    noise = torch.abs(low - low_denoise)
    '''对 I^g 和 N 做逐元素除法，得到信噪比图S(也相当于掩码mask) -- S = I^g / N'''
    mask = torch.div(low_denoise, noise + 0.0001)

    batch_size = mask.shape[0]  # 掩码的 batch_size
    height = mask.shape[2]  # 掩码的 高 h
    width = mask.shape[3]  # 掩码的 宽 w
    mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]  # 每个样本中mask的最大值
    mask_max = mask_max.view(batch_size, 1, 1, 1)  # 将最大值扩展为与掩码相同的形状
    mask_max = mask_max.repeat(1, 1, height, width)  # 将最大值扩展为与掩码相同的形状
    mask = mask * 1.0 / (mask_max + 0.0001)  # 将掩码归一化 -- 每个像素除以最大值

    mask = torch.clamp(mask, min=0, max=1.0)  # 对掩码进行截断操作，确保其值在0到1之间
    mask = mask.float()  # 转换为浮点数类型

    return mask
