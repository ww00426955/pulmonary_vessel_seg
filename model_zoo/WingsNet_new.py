import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo.BaseModelClass import BaseModel
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels1, upsampling_scale=1, out_channels2=16):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.upsamping_scale = upsampling_scale

        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels1, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm3d(self.out_channels1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(self.out_channels1, self.out_channels1, kernel_size=1, stride=1)
        self.sigmoid1 = nn.Sigmoid()

        self.conv3 = nn.Conv3d(self.out_channels1, self.out_channels1, kernel_size=1, stride=1)
        self.sigmoid2 = nn.Sigmoid()

        self.conv_out = nn.Conv3d(self.out_channels1, self.out_channels2, kernel_size=1, stride=1)

    def forward(self, x):
        f0 = self.relu1(self.instance_norm1(self.conv1(x)))
        f1 = self.sigmoid1(self.conv2(f0))
        f1 = f1 * f0
        fb = self.sigmoid2(self.conv3(f1))
        fb = fb * f1

        fg = F.interpolate(self.conv_out(fb), scale_factor=self.upsamping_scale)
        return fb, fg

class droplayer(nn.Module):
    def __init__(self, channel_num=1, thr=0.3):
        super(droplayer, self).__init__()
        self.channel_num = channel_num
        self.threshold = thr
    def forward(self, x):
        if self.training:
            r = torch.rand(x.shape[0],self.channel_num,1,1,1).cuda()
            r[r<self.threshold] = 0
            r[r>=self.threshold] = 1
            r = r*self.channel_num/(r.sum()+0.01)
            return x*r
        else:
            return x


class WingsNet_new(BaseModel):
    def __init__(self, in_channels, out_channels):
        super(WingsNet_new, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool = nn.MaxPool3d(2, 2)
        self.ConvBlock1 = ConvBlock(1, 8, 1)
        self.ConvBlock2 = ConvBlock(8, 16, 1)
        self.ConvBlock3 = ConvBlock(16, 32, 1)
        self.ConvBlock4 = ConvBlock(32, 32, 2)
        self.ConvBlock5 = ConvBlock(32, 32, 2)
        self.ConvBlock6 = ConvBlock(32, 64, 2)
        self.ConvBlock7 = ConvBlock(64, 64, 4)
        self.ConvBlock8 = ConvBlock(64, 64, 4)
        self.ConvBlock9 = ConvBlock(64, 64, 4)
        self.ConvBlock10 = ConvBlock(64, 64, 8)
        self.ConvBlock11 = ConvBlock(64, 64, 8)
        self.ConvBlock12 = ConvBlock(64, 64, 8)

        self.deep_conv = nn.Conv3d(64, 64, 1, 1)

        self.ConvBlock13 = ConvBlock(64, 64, 4)
        self.ConvBlock14 = ConvBlock(64, 64, 4)
        self.ConvBlock15 = ConvBlock(128, 64, 2)
        self.ConvBlock16 = ConvBlock(64, 64, 2)
        self.ConvBlock17 = ConvBlock(128, 32, 1)
        self.ConvBlock18 = ConvBlock(32, 16, 1)

        self.drop_layer1 = droplayer(192)
        self.drop_layer2 = droplayer(96)

        self.conv_out1 = nn.Conv3d(192, 1, 1)
        self.conv_out2 = nn.Conv3d(96, 1, 1)



    def forward(self, x):
        fb_1, fg_1 = self.ConvBlock1(x)
        fb_2, fg_2 = self.ConvBlock2(fb_1)
        fb_3, fg_3 = self.ConvBlock3(fb_2)

        fb_pool_1 = self.maxpool(fb_3)
        fb_4, fg_4 = self.ConvBlock4(fb_pool_1)
        fb_5, fg_5 = self.ConvBlock5(fb_4)
        fb_6, fg_6 = self.ConvBlock6(fb_5)

        fb_pool_2 = self.maxpool(fb_6)
        fb_7, fg_7 = self.ConvBlock7(fb_pool_2)
        fb_8, fg_8 = self.ConvBlock8(fb_7)
        fb_9, fg_9 = self.ConvBlock9(fb_8)

        fb_pool_3 = self.maxpool(fb_9)
        fb_10, fg_10 = self.ConvBlock10(fb_pool_3)
        fb_11, fg_11 = self.ConvBlock11(fb_10)
        fb_12, fg_12 = self.ConvBlock12(fb_11)

        fb_deep_feature = self.deep_conv(fb_12)

        fb_deep_feature_up = F.interpolate(fb_deep_feature, scale_factor=2)
        fb_13, fg_13 = self.ConvBlock13(fb_deep_feature_up)
        fb_14, fg_14 = self.ConvBlock14(fb_13)

        fb_14_cat = torch.cat([fb_14, fb_9], dim=1)   # 128

        fb_14_cat_up = F.interpolate(fb_14_cat, scale_factor=2)
        fb_15, fg_15 = self.ConvBlock15(fb_14_cat_up)
        fb_16, fg_16 = self.ConvBlock16(fb_15)

        fb_16_cat = torch.cat([fb_16, fb_6], dim=1)

        fb_16_cat_up = F.interpolate(fb_16_cat, scale_factor=2)
        fb_17, fg_17 = self.ConvBlock17(fb_16_cat_up)
        fb_18, fg_18 = self.ConvBlock18(fb_17)

        fg_encoder = torch.cat([fg_1, fg_2, fg_3, fg_4, fg_5, fg_6, fg_7, fg_8, fg_9, fg_10, fg_11, fg_12], dim=1) # 12 * 16 = 192
        fg_decoder = torch.cat([fg_13, fg_14, fg_15, fg_16, fg_17, fg_18], dim=1)    # 96

        fg_encoder_drop_layer = self.drop_layer1(fg_encoder)
        fg_decoder_drop_layer = self.drop_layer2(fg_decoder)

        out1 = nn.Sigmoid()(self.conv_out1(fg_encoder_drop_layer))
        out2 = nn.Sigmoid()(self.conv_out2(fg_decoder_drop_layer))

        return out1, out2

    def test(self,device='cpu'):
        input_tensor = torch.rand(2, 1, 32, 32, 32)
        ideal_out = torch.rand(2, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        # summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("WingsNet_new test is complete")

if __name__ == '__main__':
    input_x = torch.randn(1, 1, 64, 64, 64).cuda()
    net = WingsNet_new(1, 1).cuda()
    out0, out1 = net(input_x)
    print(out0.shape, out1.shape)

