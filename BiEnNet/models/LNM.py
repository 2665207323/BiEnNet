import numpy as np
from models.utils import *


class LNM_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LNM_net, self).__init__()
        self.lnm_block = LNM_block(out_channels*2)
        self.start = SCConv(inplanes=in_channels, planes=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, pooling_r=4, pool_pad=0, norm_layer=nn.BatchNorm2d)
        self.ending = SCConv(inplanes=out_channels, planes=in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, pooling_r=4, pool_pad=0, norm_layer=nn.BatchNorm2d)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        encoder_list = [1]
        decoder_list = [1]
        middle_blc = 1
        dim = out_channels  # 16

        for i in encoder_list:
            self.encoders.append(
                nn.Sequential(
                    *[block(dim) for _ in range(i)]
                )
            )
            self.downs.append(
                nn.Conv2d(dim, 2*dim, 2, 2)
            )
            dim = dim * 2  # 32

        self.middle = nn.Sequential(*[block(dim) for _ in range(middle_blc)])

        for i in decoder_list:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim*2, 1, bias=False),
                    nn.PixelShuffle(2)
                )

            )
            dim = dim // 2  # 16
            self.decoders.append(
                nn.Sequential(
                    *[block(dim) for _ in range(i)]
                )
            )

    def forward(self, low):
        x = self.start(low)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x_lnm = self.lnm_block(x)
        x_out = self.middle(x_lnm)
        for decoder, up, encs_skip in zip(self.decoders, self.ups, encs[::-1]):
            x_out = up(x_out)
            x_out = x_out + encs_skip
            x_out = decoder(x_out)
        x_out = self.ending(x_out)
        x_out = x_out + low
        return x_out


if __name__ == "__main__":
    batch_size = 16
    expected_mean = 0.5
    low_images_batch = np.random.rand(batch_size, 3, 256, 256)  # 示例低光图像批次
    low_images_batch = torch.from_numpy(low_images_batch).float()
    print(low_images_batch)

    model = LNM_net(3, 16)
    # histograms = calc_hist(low_images_batch, num_bins, expected_mean)
    norm_low = model(low_images_batch)
    print(norm_low.shape)



