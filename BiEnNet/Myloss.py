import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import pytorch_ssim

import numpy as np
from IQA_pytorch import SSIM


# 振幅损失
class FFT_Loss(nn.Module):
    def __init__(self):
        super(FFT_Loss, self).__init__()

    def forward(self, x, gt):
        x = x + 1e-8
        gt = gt + 1e-8
        x_freq = torch.fft.rfft2(x, norm='backward')  # 暗光输入 ：时域 -> 频域
        x_amp = torch.abs(x_freq)  # 振幅
        # x_phase = torch.angle(x_freq)  # 相位

        gt_freq = torch.fft.rfft2(gt, norm='backward')  # 正常光图像 ：时域 -> 频域
        gt_amp = torch.abs(gt_freq)  # 振幅
        # gt_phase = torch.angle(gt_freq)  # 相位

        loss_amp = torch.mean(torch.sqrt(torch.pow(x_amp - gt_amp, 2) + 1e-8) )  # 振幅差的平方 的 平均，衡量两个频域信号之间的差异
        # loss_phase = torch.mean(torch.pow((x_phase - gt_phase), 2))  # 相位损失实际未使用
        return loss_amp


class CharbonnierLoss(nn.Module):  # 近似L1loss
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


'''(使用到了)  L1 和 SSIM'''
class L_recon(nn.Module):

    def __init__(self):
        super(L_recon, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()
        self.CharbonnierLoss = CharbonnierLoss()

    def forward(self, R_low, high):
        # L1 = torch.abs(R_low - high).mean()  # L1 - 绝对差异损失，即，像素级别的差异平均值
        L1 = self.CharbonnierLoss(R_low, high)  # 近似L1损失
        L2 = (1 - self.ssim_loss(R_low, high)).mean()  # SSIM的结构相似性损失
        return L1, L2


'''未真正使用'''
class L_recon_low(nn.Module):

    def __init__(self):
        super(L_recon_low, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, R_low, high, ill):
        L1 = (R_low * ill - high * torch.log(R_low * ill + 0.0001)).mean()  # R_low * ill 与 high * log(R_low * ill) 之间的平均差异

        return L1


'''(使用到了) 计算enhanced_image, img_highlight之间的颜色相似性损失  - 通过计算两个颜色通道之间的差异来度量'''
class L_color_zy(nn.Module):

    def __init__(self):
        super(L_color_zy, self).__init__()

    def forward(self, x, y):
        product_separte_color = (x * y).mean(1, keepdim=True)  # 计算 x,y 两个颜色通道的逐像素相乘的平均值，有助于捕捉颜色通道之间的相似性
        x_abs = (x ** 2).mean(1, keepdim=True) ** 0.5   # 分别计算 x 和 y 颜色通道的平方和的平方根，有助于对颜色的强度(/亮度)进行归一化
        y_abs = (y ** 2).mean(1, keepdim=True) ** 0.5
        # 第一部分 - 将颜色相乘的平均值 除以 颜色强度的平方根之和，并与1相减，得到一个度量相似性的值
        # 第二部分 - 该部分基于 余弦相似度的反余弦值，用于衡量颜色通道之间的角度差异(色调饱和度差异)
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        return loss1


'''(使用到了) 计算enhanced_image, img_highlight之间的梯度一致性损失，即结构性损失L_structure
首先通过卷积操作计算了每个通道的水平和垂直梯度，然后计算了不同尺度的梯度一致性损失'''
class L_grad_cosist(nn.Module):

    def __init__(self):
        super(L_grad_cosist, self).__init__()
        # 定义right、down两个卷积核，分别用于计算水平和垂直梯度   1-增强梯度  -1-抑制梯度  0-不进行卷积
        # .unsqueeze(0).unsqueeze(0)   -   卷积核 3*3 --> 1*1*3*3
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_of_one_channel(self, x, y):  # 分别计算enhanced_image, img_highlight的单通道的水平和垂直梯度，并取绝对值，保证梯度非负
        D_org_right = F.conv2d(x, self.weight_right, padding="same")
        D_org_down = F.conv2d(x, self.weight_down, padding="same")
        D_enhance_right = F.conv2d(y, self.weight_right, padding="same")
        D_enhance_down = F.conv2d(y, self.weight_down, padding="same")
        return torch.abs(D_org_right), torch.abs(D_enhance_right), torch.abs(D_org_down), torch.abs(D_enhance_down)

    def gradient_Consistency_loss_patch(self, x, y):
        # B*C*H*W     对输入图像进行一些预处理，将像素值归一化到 [0, 1] 范围
        # 计算x和y中的最小像素值，然后取绝对值，并使用.detach() 方法将其从计算图中分离，以便不会影响之后的梯度计算
        # 然后，将这些最小值从输入图像中减去，将像素值归一化到非负范围
        min_x = torch.abs(x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y

        # B*1*1,3 梯度一致性的第一部分 - 像素值乘积的一致性损失 和 基于余弦相似度的损失
        product_separte_color = (x * y).mean([2, 3], keepdim=True)  # 计算x,y的宽、高通道的逐像素相乘的平均值
        x_abs = (x ** 2).mean([2, 3], keepdim=True) ** 0.5  # 计算平均绝对值梯度
        y_abs = (y ** 2).mean([2, 3], keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        # 梯度一致性的第二部分
        product_combine_color = torch.mean(product_separte_color, 1, keepdim=True)  # 逐像素值乘积的平均值
        x_abs2 = torch.mean(x_abs ** 2, 1, keepdim=True) ** 0.5  # 平均绝对值梯度
        y_abs2 = torch.mean(y_abs ** 2, 1, keepdim=True) ** 0.5
        loss2 = torch.mean(1 - product_combine_color / (x_abs2 * y_abs2 + 0.00001)) + torch.mean(
            torch.acos(product_combine_color / (x_abs2 * y_abs2 + 0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        # 低水平、高水平、低垂直、高垂直
        x_R1, y_R1, x_R2, y_R2 = self.gradient_of_one_channel(x[:, 0:1, :, :], y[:, 0:1, :, :])  # R通道的水平和垂直梯度
        x_G1, y_G1, x_G2, y_G2 = self.gradient_of_one_channel(x[:, 1:2, :, :], y[:, 1:2, :, :])  # G通道的水平和垂直梯度
        x_B1, y_B1, x_B2, y_B2 = self.gradient_of_one_channel(x[:, 2:3, :, :], y[:, 2:3, :, :])  # B通道的水平和垂直梯度
        x = torch.cat([x_R1, x_G1, x_B1, x_R2, x_G2, x_B2], 1)  # 将三个通道的梯度合并，确保每个通道的梯度在两个方向上都有两个都对应的张量
        y = torch.cat([y_R1, y_G1, y_B1, y_R2, y_G2, y_B2], 1)

        B, C, H, W = x.shape
        loss = self.gradient_Consistency_loss_patch(x, y)  # 整个图像的梯度一致性损失
        # 四个分块的梯度一致性损失（未使用）
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, 0:W // 2], y[:, :, 0:H // 2, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, 0:W // 2], y[:, :, H // 2:, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, W // 2:], y[:, :, 0:H // 2, W // 2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, W // 2:], y[:, :, H // 2:, W // 2:])

        return loss  # +loss1#+torch.mean(torch.abs(x-y))#+loss1


'''(使用到了) 通过计算enhanced_image, img_highlight之间的梯度一致性损失 来衡量 亮度损失L_brightness'''
class L_bright_cosist(nn.Module):

    def __init__(self):
        super(L_bright_cosist, self).__init__()

    def gradient_Consistency_loss_patch(self, x, y):
        # B*C*H*W  对输入图像进行一些预处理，将像素值归一化到 [0, 1] 范围
        min_x = torch.abs(x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y

        # B*1*1,3  梯度一致性的第一部分 - 像素值乘积的一致性损失 和 基于余弦相似度的损失
        product_separte_color = (x * y).mean([2, 3], keepdim=True)  # 计算x,y的宽、高通道的逐像素相乘的平均值
        x_abs = (x ** 2).mean([2, 3], keepdim=True) ** 0.5  # 平均绝对值梯度
        y_abs = (y ** 2).mean([2, 3], keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        # 梯度一致性的第二部分
        product_combine_color = torch.mean(product_separte_color, 1, keepdim=True)
        x_abs2 = torch.mean(x_abs ** 2, 1, keepdim=True) ** 0.5
        y_abs2 = torch.mean(y_abs ** 2, 1, keepdim=True) ** 0.5
        loss2 = torch.mean(1 - product_combine_color / (x_abs2 * y_abs2 + 0.00001)) + torch.mean(
            torch.acos(product_combine_color / (x_abs2 * y_abs2 + 0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        B, C, H, W = x.shape
        loss = self.gradient_Consistency_loss_patch(x, y)  # 整个图像的梯度一致性损失
        # 四个分块的梯度一致性损失（未使用）
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, 0:W // 2], y[:, :, 0:H // 2, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, 0:W // 2], y[:, :, H // 2:, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, W // 2:], y[:, :, 0:H // 2, W // 2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, W // 2:], y[:, :, H // 2:, W // 2:])

        return loss  # +loss1#+torch.mean(torch.abs(x-y))#+loss1


class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))


ssim = SSIM()
psnr = PSNR()


def validation(model, val_loader, epoch):
    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img, high_exp_mean, img_name = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda(), imgs[3]
            enhanced_image, low_csn = model(low_img, high_exp_mean)

        ssim_value = ssim(enhanced_image, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_image, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('epoch' + str(epoch) + ':' + 'the PSNR is' + str(PSNR_mean) + '                 the SSIM is' + str(SSIM_mean))

    return SSIM_mean, PSNR_mean
