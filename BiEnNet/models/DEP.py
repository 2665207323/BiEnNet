import torch
from torch import nn
from models.CDC import cdcconv
from models.utils import *

# 曝光学习模块
class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.cdc = cdcconv(nc, nc)
        self.fuse = nn.Conv2d(2 * nc, nc, 1, 1, 0)

    def forward(self, x):
        x_conv = self.conv(x)
        x_cdc = self.cdc(x)
        x_out = self.fuse(torch.cat([x_conv, x_cdc], 1))

        return x_out


# 双曝光处理 DEP 模块
class DEP(nn.Module):
    def __init__(self, nc):
        super(DEP, self).__init__()
        self.relu = nn.ReLU()
        self.norm = LNM_block2(nc)
        self.prcessblock = ProcessBlock(nc)
        self.fuse1 = nn.Conv2d(2 * nc, nc, 1, 1, 0)
        self.fuse2 = nn.Conv2d(2 * nc, nc, 1, 1, 0)
        self.post = nn.Sequential(
            nn.Conv2d(2 * nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        x_norm = self.norm(x)  # 曝光不变表示 - 实例标准化层
        x_p = self.relu(x)  # ReLU
        x_n = self.relu(-x)  # NegReLU
        x_p = self.prcessblock(x_p)  # 曝光学习块 - ReLU
        x_n = -self.prcessblock(x_n)  # (曝光学习块 - NegReLU) + 翻转
        x_p1 = self.fuse1(torch.cat([x_norm, x_p], 1))  # 1*1卷积
        x_n1 = self.fuse2(torch.cat([x_norm, x_n], 1))  # 1*1卷积
        x_out = self.post(torch.cat([x_p1, x_n1], 1))

        return x_out + x


if __name__ == "__main__":

    conv = nn.Conv2d(32, 32, 3)
    cdc = cdcconv(32, 32)
    print('total parameters:', sum(param.numel() for param in conv.parameters()))
    print('total parameters:', sum(param.numel() for param in cdc.parameters()))
    # print(norm_low.shape)
