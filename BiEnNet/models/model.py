import warnings
from models.LEB_block import *
from models.SNR import *
from models.CDC import cdcconv
from models.LNM import *
from models.DEP import *
from models.utils import *
from data.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


'''深度卷积'''
class Deep_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Deep_Conv, self).__init__()
        self.depth_conv = SCConv(
            inplanes=in_ch,
            planes=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            pooling_r=4,
            pool_pad=0,
            norm_layer=nn.BatchNorm2d
        )

    def forward(self, input):
        out = self.depth_conv(input)
        return out


'''空洞卷积 - 扩大感受野'''
class Dilated_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilated_Conv, self).__init__()
        # 空洞卷积
        self.depth_conv = SCConv(
            inplanes=in_ch,
            planes=out_ch,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,  # 卷积核中的每个元素之间有一个间隔为 2 的空隙。这将使卷积核在进行卷积操作时，跳过一些像素，从而扩大了感受野。
            groups=1,
            pooling_r=4,
            pool_pad=0,
            norm_layer=nn.BatchNorm2d
        )

    def forward(self, input):
        out = self.depth_conv(input)
        return out


'''逐点卷积 - 获取直方图的特征，这些特征图可以提供直方图的各种统计信息和特征, 帮助模型在进行图像增强时更好地处理不同的亮度、对比度和色彩平衡等问题'''
class Point_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Point_Conv, self).__init__()
        # 逐点卷积
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.point_conv(input)
        return out


class mlp_net(nn.Module):  # (以 微光图像的 V通道 和 期望的平均亮度 为输入   MLP --- 5 layers)
    def __init__(self, nbins, out_dim):
        super(mlp_net, self).__init__()
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.g_conv1 = Point_Conv(2*(nbins + 1), out_dim)  # 用于亮度直方图 in_dim=32+1, out_dim=16
        self.g_conv2 = Point_Conv(out_dim, out_dim)
        self.g_conv3 = Point_Conv(out_dim + 2*(nbins + 1), out_dim)
        self.g_conv4 = Point_Conv(out_dim, out_dim)
        self.g_conv5 = Point_Conv(out_dim, 8)

    def forward(self, hist, x_channel):
        out1 = self.leakyrelu(self.g_conv1(hist))
        out2 = self.leakyrelu(self.g_conv2(out1))  # out_dim=16,   逐点卷积
        out3 = self.leakyrelu(self.g_conv3(torch.cat([out2, hist], dim=1)))  # out_dim=16,   将g2 和 输入hist在通道维度上拼接，并逐点卷积
        out4 = self.leakyrelu(self.g_conv4(out3))  # out_dim=16
        out5 = self.leakyrelu(self.g_conv5(out4))  # out_dim=8

        retouch_out = retouch(x_channel, out5)  # 图像修复，得到全局建议  [b,8,h,w]
        return retouch_out


def retouch(x, x_r):  # 图像修复函数 ---- 将经过逐点卷积层(mlp)之后得到的out5应用到输入x_V上，生成修复后的亮度调整图     将不同通道的修复权重与输入相乘并累加

    x = x + x_r[:, 0:1, :, :] * (-torch.pow(x, 2) + x)  # 将输入张量 x 与 x_r 张量的第一个通道进行逐元素相乘，并将结果添加到原始输入 x 中     (引入非线性性质)
    x = x + x_r[:, 1:2, :, :] * (-torch.pow(x, 2) + x)
    x = x + x_r[:, 2:3, :, :] * (-torch.pow(x, 2) + x)
    x = x + x_r[:, 3:4, :, :] * (-torch.pow(x, 2) + x)
    x = x + x_r[:, 4:5, :, :] * (-torch.pow(x, 2) + x)
    x = x + x_r[:, 5:6, :, :] * (-torch.pow(x, 2) + x)
    x = x + x_r[:, 6:7, :, :] * (-torch.pow(x, 2) + x)

    enhance_image = x + x_r[:, 7:8, :, :] * (-torch.pow(x, 2) + x)  # [b,1,h,w]
    return enhance_image


class LEN(nn.Module):
    def __init__(self, out_dim):
        super(LEN, self).__init__()
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.e_conv1 = Deep_Conv(4+3, out_dim)  # in_dim = 7, out_dim = 16

        self.e_conv2 = Deep_Conv(out_dim, out_dim)
        self.e_conv3 = Deep_Conv(out_dim + 3 + 3 + 1, out_dim)
        self.e_conv33 = Dilated_Conv(out_dim, out_dim)
        self.e_conv4 = Dilated_Conv(out_dim, out_dim)
        self.e_conv5 = Deep_Conv(out_dim * 2, out_dim)

    def forward(self, x_cat, x_V_up, retouch_image):

        x1 = self.leakyrelu(self.e_conv1(torch.cat([x_cat - x_V_up / 2, x_V_up / 2], 1)))
        x2 = self.leakyrelu(self.e_conv2(x1))
        x3 = self.leakyrelu(self.e_conv3(torch.cat([x2, x_cat, retouch_image], 1)))
        x33 = self.leakyrelu(self.e_conv33(x3))
        x4 = self.leakyrelu(self.e_conv4(x33))
        x5 = self.leakyrelu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        return x5


# 局部残差 + 信噪比图
class local_block(nn.Module):
    def __init__(self, in_dim=3*2, out_dim=16):
        super(local_block, self).__init__()
        # self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, groups=1)
        self.conv = SCConv(in_dim, out_dim, 3, 1, 1, 1, 1, 4, 0, nn.BatchNorm2d)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        LFE_Block = [LEB(out_dim, drop_path=0.01), LEB(out_dim, drop_path=0.05)]
        self.LFE = nn.Sequential(*LFE_Block)

    def forward(self, x_cat, x_denoise_cat):
        fea = self.leaky_relu(self.conv(x_cat))  # [b, c, h, w]
        res_fea = self.LFE(fea) + fea  # [b, out_dim, h, w]  局部残差

        snr_map = SNR_mask(x_cat, x_denoise_cat)  # [b, 1, h, w]
        snr_map = snr_map.repeat(1, fea.shape[1], 1, 1)  # [b, out_dim, h, w]在通道维度上进行赋值，以匹配特征图的通道数。这样，每个通道都有相同的掩码
        return res_fea, snr_map


class BiEnNet(nn.Module):

    def __init__(self, scale_factor=20, nbins=32, in_dim=3, out_dim=16):
        super(BiEnNet, self).__init__()

        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.scale_factor = scale_factor  # 缩放因子
        self.nbins = nbins  # 直方图的bin的数量, = 32

        self.depth_conv1 = Deep_Conv(out_dim, out_dim)
        self.depth_conv2 = Deep_Conv(out_dim * 2, out_dim)
        self.empty_conv1 = Dilated_Conv(out_dim, out_dim)
        self.empty_conv2 = Dilated_Conv(out_dim, out_dim)
        self.conv_last = Deep_Conv(out_dim, 3)
        self.mlp_net = mlp_net(nbins, out_dim=out_dim).to(device)
        self.len_net = LEN(out_dim=out_dim).to(device)
        self.lnm_net = LNM_net(in_dim, out_dim//2).to(device)
        self.dep_net = DEP(out_dim).to(device)
        self.local_net = local_block(3 * 2, out_dim).to(device)

    def forward(self, x, exp_mean):  # hist-[[batch, bins+1, 1, 1]]

        x_lnm = self.lnm_net(x)  # 通道归一化模块  [b, 3, h, w]
        x_cat = torch.cat([x, x_lnm], 1)  # [b, 6, h, w]

        # BSF
        x_denoise = calc_low_denoise(x).to(device)
        x_denoise_lnm = calc_low_denoise(x_lnm).to(device)
        x_denoise_cat = torch.cat([x_denoise, x_denoise_lnm], 1)

        hist_low = calc_hist(x, self.nbins, exp_mean).to(device)  # 计算暗光图像的亮度直方图 [b, bins+1, 1, 1]
        hist_low_lnm = calc_hist(x_lnm, self.nbins, exp_mean).to(device)
        hist_low_cat = torch.cat([hist_low, hist_low_lnm], 1)  # [b, 2*(bins+1), 1, 1]

        x_V = x.max(1, keepdim=True)[0]  #  [b, 1, h, w] 取x的每个通道上的最大值，并保持与 x 相同的维度，并取元组(value, id)的第一个元素

        if self.scale_factor == 1:
            x_V_up = torch.mean(x_V, [2, 3], keepdim=True) + x_V * 0  # [b, 1, h, w]

        else:  # 否则, 通过插值函数interpolate先下采样(1/scale)，再上采样(scale)    确保后面各个特征图拼接时，尺寸不冲突
            x_V_down = F.interpolate(x_V, scale_factor=1/self.scale_factor, mode='bilinear')
            x_V_up = F.interpolate(x_V_down, scale_factor=self.scale_factor, mode='bilinear')

        # GBA (MLP-5层)(逐点卷积)  对亮度直方图进行mlp  并进行图像修复，得到全局色相,饱和度和亮度调整建议
        retouch_image = self.mlp_net(hist_low_cat, x_V)  # hist : [b,bins+1,1,1]->[b,8,1,1]  x_V : [b,1,h,w]   retouch_V : [b,1,h,w]
        fea_V = self.len_net(x_cat, x_V_up, retouch_image)

        # 局部残差 + 信噪比融合
        local_fea, snr_map = self.local_net(x_cat, x_denoise_cat)  # 局部分支
        enhance_fea = fea_V * (1 - snr_map) + local_fea * snr_map  # 特征融合  [b, 16, h, w]

        enhance_fea = self.leakyrelu(self.depth_conv1(enhance_fea))

        # DEP
        enhance_fea_dep = self.dep_net(enhance_fea)
        enhance_fea_out = self.leakyrelu(self.empty_conv1(enhance_fea_dep))
        enhance_fea_out = self.leakyrelu(self.empty_conv2(enhance_fea_out))
        out_fea = self.leakyrelu(self.depth_conv2(torch.cat([enhance_fea, enhance_fea_out], 1)))

        # 应用softplus非线性激活函数，将卷积后的特征图的每个像素值映射到非负区间
        enhance_image = F.softplus(self.conv_last(out_fea))

        # 返回 增强后的图像
        return enhance_image, x_lnm


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # net = Local_pred_S()  # param - 22454

    Net2 = BiEnNet(nbins=8, out_dim=16)  # 8->38911  16->138895
    Net1 = local_block(6, 12)
    Net = LNM_net(3, 16)
    print('total parameters:', sum(param.numel() for param in Net2.parameters()))
    print('total parameters:', sum(param.numel() for param in Net1.parameters()))
    print('total parameters:', sum(param.numel() for param in Net.parameters()))
    # print('total parameters:', sum(param.numel() for param in snr.parameters()))  # 42796
    # print(m.shape, a.shape)

