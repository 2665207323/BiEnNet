import random
import warnings

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
from data import dataset_train
import models.model
import Myloss
import numpy as np
from torchvision import transforms
from Myloss import validation
import re

PSNR_mean = 0
SSIM_mean = 0

def seed_torch(seed=20240302):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch()
cudnn.benchmark = True  # 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
        if classname.find('Conv') != -1 and classname not in ['Conv2d_Hori_Veri_Cross', 'Conv2d_Diag_Cross']:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def amp_aug(x, y):  # 基于频域的亮度扰动方案 x-normal  y-input
    x = x + 1e-8
    y = y + 1e-8
    x_freq = torch.fft.rfft2(x, norm='backward')  # 二维傅里叶变换 - 正常光
    x_amp = torch.abs(x_freq)  # 振幅
    x_phase = torch.angle(x_freq)  # 相位

    y_freq = torch.fft.rfft2(y, norm='backward')  # 二维傅里叶变换 - 暗光输入
    y_amp = torch.abs(y_freq)  # 振幅
    y_phase = torch.angle(y_freq)  # 相位

    mix_alpha = torch.rand(1).to(device) / 0.5  # 为alpha生成一个随机的数, 并将其除以0.5, 得到一个介于[0,2]的随机数
    mix_alpha = torch.clip(mix_alpha, 0.65, 1)  # 将alpha的值限制在 0-0.75 之间
    y_amp = mix_alpha * y_amp + (1 - mix_alpha) * x_amp  # 振幅组合

    real = y_amp * torch.cos(y_phase)  # 计算复数的实部
    imag = y_amp * torch.sin(y_phase)  # 计算复数的虚部
    y_out = torch.complex(real, imag) + 1e-8  # 构造成复数，添加小常数，防止除零错误
    y_out = torch.fft.irfft2(y_out) + 1e-8  # 逆快速傅里叶变换，转换回时域表示

    return y_out


# 定义早停机制
class EarlyStopping:
    def __init__(self, patience=2000, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

    def get_counter(self):
        return self.counter


def train(config):
    scale_factor = config.scale_factor  # 缩放因子
    BiEnNet = models.model.BiEnNet(scale_factor, config.nbins).to(device)  # 加载模型到cuda
    early_stopping = EarlyStopping(patience=3000, delta=0.001)  # 设置适当的patience和delta值

    BiEnNet.apply(weights_init)

    if config.load_pretrain == True:  # 加载预训练权重
        BiEnNet.load_state_dict(torch.load(config.pretrain_dir))


    train_data = dataset_train.train_dataset(low_img_dir=config.lowlight_images_path, high_img_dir=config.highlight_images_path,
                                             task=config.task[0], batch_w=config.patch_size, batch_h=config.patch_size,)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    val_data = dataset_train.train_dataset(low_img_dir=config.val_lowlight_images_path, high_img_dir=config.val_highlight_images_path,
                                           task=config.task[1], batch_w=50, batch_h=config.patch_size,)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    loss_amp = Myloss.FFT_Loss()
    loss_low = Myloss.CharbonnierLoss()
    L_color_zy = Myloss.L_color_zy()  # 基于 余弦相似度 的颜色相似性损失
    L_grad_cosist = Myloss.L_grad_cosist()  # 结构损失 L_structure
    L_bright_cosist = Myloss.L_bright_cosist()  # 亮度损失 L_brightness
    L_recon = Myloss.L_recon()  # 计算 近似L1损失 CharbonnierLoss 和 SSIM损失


    optimizer = torch.optim.Adam(BiEnNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    BiEnNet.train()
    loss = 0
    ssim_high = 0
    psnr_high = 0
    epoch_high = 0

    for epoch in range(config.num_epochs):
        total_loss = 0.0  # 用于计算每个epoch的平均损失
        for iteration, (img_lowlight, img_highlight, high_exp_mean, img_name) in enumerate(train_loader):

            img_lowlight = img_lowlight.to(device)
            img_highlight = img_highlight.to(device)
            high_exp_mean = high_exp_mean.to(device)

            enhanced_image, low_lnm = BiEnNet(img_lowlight, high_exp_mean)

            img_low_aug = amp_aug(img_highlight, img_lowlight)  # 对暗光输入进行频率扰动
            enhanced_image_aug, img_low_aug_lnm = BiEnNet(img_low_aug, high_exp_mean)  # 对扰动低光图像进行亮度通道归一化处理

            loss_low_lnm = 20 * loss_low(img_lowlight, low_lnm)
            loss_lnm_high = 10 * loss_low(low_lnm, img_highlight)

            loss_low_aug = 20 * loss_low(img_low_aug, img_low_aug_lnm)
            loss_aug_high = 10 * loss_low(img_low_aug_lnm, img_highlight)

            loss_low_lnm_aug = 80 * loss_low(low_lnm, img_low_aug_lnm)
            loss_aug_amp = 0.05*loss_amp(enhanced_image_aug, img_highlight)

            loss_struct = L_grad_cosist(enhanced_image, img_highlight)  # 结构损失 L_structure
            loss_bright = L_bright_cosist(enhanced_image, img_highlight)  # 亮度损失 L_brightness
            loss_L1, loss_ssim = L_recon(enhanced_image, img_highlight)  # 计算 近似L1损失CharbonnierLoss 和 SSIM损失
            loss_col = torch.mean(L_color_zy(enhanced_image, img_highlight))  # 基于 余弦相似度 的颜色相似性损失L_color

            # 总损失
            loss = loss_ssim + loss_L1 + loss_struct + loss_bright + loss_col + loss_low_lnm + loss_low_aug + loss_low_lnm_aug + loss_aug_amp + loss_lnm_high + loss_aug_high

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 累加损失 - 用于早停判断

            if ((iteration + 1) % config.display_iter) == 0:  # 2
                if ((epoch + 1) <= 1000 and (epoch + 1) % config.snapshot_iter[0] == 0) or ((epoch + 1) > 1000 and (epoch + 1) % config.snapshot_iter[1] == 0):  # 20, 100
                    torchvision.utils.save_image(torch.concat([img_lowlight[0], low_lnm[0], img_low_aug_lnm[0], enhanced_image[0]], dim=2), config.sample_dir + str(epoch) + '.png')
                    BiEnNet.eval()
                    SSIM_mean, PSNR_mean = validation(BiEnNet, val_loader, epoch)
                    if PSNR_mean > psnr_high:
                        psnr_high = PSNR_mean
                        epoch_high = epoch
                        print('the highest PSNR value is:', str(psnr_high))
                        torch.save(BiEnNet.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))  # 保存模型
                    with open(config.snapshots_folder + 'log.txt', 'a+') as f:
                        f.write('epoch' + str(epoch) + ':' + 'the PSNR is' + str(PSNR_mean) + 'the SSIM is' + str(
                            SSIM_mean) + '\n')

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        early_stopping(avg_loss)  # 检查是否要提前停止
        # 在需要获取counter的值时调用get_counter方法
        current_counter = early_stopping.get_counter()

        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Average Loss, loss_low_lnm: {avg_loss, loss_low_lnm.item()}, Current Counter: {current_counter}")
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 在训练完成后将最大PSNR值和对应的epoch输出，并写入log.txt文件
    print('the highest PSNR value is:', str(psnr_high))
    with open(config.snapshots_folder + 'log.txt', 'a+') as f:
        f.write('the highest PSNR at epoch ' + str(epoch_high) + ': ' + 'PSNR is ' + str(psnr_high) + '\n')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="/home/huangshenghui/dataset/LOL_v2/our485/low/")
    parser.add_argument('--highlight_images_path', type=str, default="/home/huangshenghui/dataset/LOL_v2/our485/high/")
    parser.add_argument('--val_lowlight_images_path', type=str, default="/home/huangshenghui/dataset/LOL_v2/eval142/low/")
    parser.add_argument('--val_highlight_images_path', type=str, default="/home/huangshenghui/dataset/LOL_v2/eval142/high/")

    parser.add_argument('--task', type=str, nargs='+', default=["train", "val"])
    parser.add_argument('--nbins', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=200)
    parser.add_argument('--exp_mean', type=float, default=0.55)
    parser.add_argument('--sample_dir', type=str, default="/home/huangshenghui/FLW_SSC_CSN_SNR_ECP/sample_train_aug_p100_bin14_b128_s2000_v2/")

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=30000)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=2)
    parser.add_argument('--snapshot_iter', type=int, nargs='+', default=[20, 100])
    parser.add_argument('--scale_factor', type=int, default=20)
    parser.add_argument('--snapshots_folder', type=str, default="/home/huangshenghui/FLW_SSC_CSN_SNR_ECP/weight_train_aug_p100_bin14_b128_s2000_v2/")
    parser.add_argument('--resume_checkpoint', type=str, default="", help="Path to the checkpoint file to resume training, default from the first epoch")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="/home/huangshenghui/FLW_SSC_CSN_SNR_ECP/weight_train_aug_p100_bin14_b128_s2000_v2/best_Epoch.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
