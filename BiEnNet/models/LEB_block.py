import os
import torch
import torch.nn as nn
from models.utils import *
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


class Aff_channel(nn.Module):
    def __init__(self, dim, channel_first=True):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x1 = torch.tensordot(x, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = x * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        return x2


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.leakrelu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.leakrelu(self.fc1(x))
        x = self.act(x)
        x = self.drop(x)
        x = self.leakrelu(self.fc2(x))
        x = self.drop(x)
        return x


class LEB(nn.Module):  # 局部增强块
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=Aff_channel, init_values=1e-4):
        super().__init__()
        self.leakrelu = nn.LeakyReLU(inplace=True)
        self.pos_embed = SCConv(dim, dim, 3, 1, 1, 1, 1, 4, 0, nn.BatchNorm2d)
        self.norm1 = norm_layer(dim)  # 光归一化1

        # PWConv-DWConv-PWConv ( 1x1 -> 5x5 -> 1x1)增强局部细节
        self.conv1 = nn.Conv2d(dim, dim, 1)  # in_dim=out_dim=16
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = SCConv(inplanes=dim, planes=dim, kernel_size=5, padding=2, groups=dim)  # in_dim=out_dim=16

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 随机深度的下降路径，看这是否比这里的下降更好，作用于 PWConv-DWConv-PWConv 部分
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)  # 光归一化2
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)  # k1
        self.gamma_2 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)  # k2
        # 两个1×1卷积分别增强序列（token）表示
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)  # 3x3的深度卷积 用于 位置编码
        B, C, H, W = x.shape   # C=16
        norm_x = x.flatten(2).transpose(1, 2)  # 拉伸，铺平  C=600*400=240000
        norm_x = self.norm1(norm_x)  # 光归一化1  C=600*400=240000
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)  # C=16

        # 第一个残差和
        x = x + self.drop_path(self.gamma_1 * self.leakrelu(self.conv2(self.attn(self.leakrelu(self.conv1(norm_x))))))  # C=16
        # 光归一化2
        norm_x = x.flatten(2).transpose(1, 2)
        norm_x = self.norm2(norm_x)
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)  # C=16
        # 单个PEM内部的后半部分（第二个残差和）
        x = x + self.drop_path(self.gamma_2 * self.mlp(norm_x))  # C=16
        return x




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cb_block = LEB(dim=16)
    xx = torch.Tensor(1, 16, 400, 600)
    xx = cb_block(xx)
    print(xx.shape)
