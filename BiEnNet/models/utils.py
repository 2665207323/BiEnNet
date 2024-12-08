# 2020 CVPR
# https://github.com/MCG-NKU/SCNet
# https://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
import torch
from torch import nn
import torch.nn.functional as F


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, pooling_r=4, pool_pad=0, norm_layer=nn.BatchNorm2d):
        super(SCConv, self).__init__()

        # Use a 1x1 Conv layer to adjust the number of channels in identity
        self.conv_identity = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False)
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r, padding=pool_pad),
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )

    def forward(self, x):
        identity = self.conv_identity(x)  # 使用 1x1 卷积来调整通道数，确保 identity 张量的通道数与 planes 一致
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)  [b, planes, h, w]
        out = self.k4(out)  # k4
        return out


# 可微门控模块
class Generate_gate(nn.Module):
    def __init__(self, channels):
        super(Generate_gate, self).__init__()
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(channels, channels//2, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(channels//2, channels, 1),
                                  nn.ReLU())

        self.epsilon = 1e-8

    def forward(self, x):
        alpha = self.proj(x)
        gate = (alpha ** 2) / (alpha ** 2 + self.epsilon)  # 跨通道维度的二进制指标 g = α² / (α²+ϵ)

        return gate


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # 将 x 沿第一个维度(dim=1)均匀的分割成两份, 用于门控机制
        return x1 * x2  # 通道数为 x 的一半


class block(nn.Module):
    def __init__(self, channels, drop_out_rate=0.):
        super(block, self).__init__()
        mid_dim = channels * 2
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=mid_dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = SCConv(inplanes=mid_dim, planes=mid_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=mid_dim, pooling_r=4, pool_pad=0, norm_layer=nn.BatchNorm2d)
        self.conv3 = nn.Conv2d(in_channels=mid_dim // 2, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention  简化的通道注意力
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=mid_dim // 2, out_channels=mid_dim // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

        self.sg = SimpleGate()  # 简单的门控机制  通道数减半

        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=mid_dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=mid_dim // 2, out_channels=channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.droupout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.droupout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        x_copy = x

        x_copy = self.norm1(x_copy)  # C=channels
        x_copy = self.conv1(x_copy)  # C=channels * 2
        x_copy = self.conv2(x_copy)  # C=channels * 2
        x_copy = self.sg(x_copy)  # Simple Gate  过滤信息
        x_copy = x_copy * self.sca(x_copy)  # Simplified Channel Attention(简化的通道注意力)  通过学习一个转换函数来重新校准每个通道的重要性
        x_copy = self.conv3(x_copy)
        x_copy = self.droupout1(x_copy)
        y = x + x_copy * self.beta

        x_copy = self.conv4(self.norm2(y))
        x_copy = self.sg(x_copy)
        x_copy = self.conv5(x)
        x_copy = self.droupout2(x_copy)
        z = y + x_copy * self.gamma

        return z


class LNM_block(nn.Module):
    def __init__(self, channels):
        super(LNM_block, self).__init__()
        self.gate = Generate_gate(channels)
        for i in range(channels):
            setattr(self, 'CSN_' + str(i), nn.InstanceNorm2d(1, affine=True))  # 实例归一化  IN(x) = γ * ( (x - μ(x)) / σ(x)) + β


    def forward(self, x):
        C = x.shape[1]  # 获取通道数
        gate = self.gate(x)
        x_copy = torch.cat([getattr(self, 'CSN_' + str(i))(x[:, i, :, :][:, None, :, :]) for i in range(C)], dim=1)
        x = gate * x_copy + (1 - gate) * x    # xn+1 = (1-g) ⨀ xn + g ⨀ x'n
        return x


class LNM_block2(nn.Module):
    def __init__(self, channels):
        super(LNM_block2, self).__init__()
        self.gate2 = Generate_gate(channels)
        for i in range(channels):
            setattr(self, 'CSN2_' + str(i), nn.InstanceNorm2d(1, affine=True))


    def forward(self, x):
        C = x.shape[1]  # 获取通道数
        gate2 = self.gate2(x)
        x_copy = torch.cat([getattr(self, 'CSN2_' + str(i))(x[:, i, :, :][:, None, :, :]) for i in range(C)], dim=1)
        x = gate2 * x_copy + (1 - gate2) * x  # xn+1 = (1-g) ⨀ xn + g ⨀ x'n
        return x


if __name__ == '__main__':
    # x = torch.randn(1, 32, 16, 16)  # 创建随机输入张量
    model = block(16)
    model1 = SCConv(16, 16)
    csn = LNM_block(16)
    # print(model2(x).shape)  # 打印模型输出的形状
    # for name in model1.parameters():
    #     print(name)
    print('total parameters:', sum(param.numel() for param in model.parameters()))
    print('total parameters:', sum(param.numel() for param in model1.parameters()))
    print('total parameters:', sum(param.numel() for param in csn.parameters()))
