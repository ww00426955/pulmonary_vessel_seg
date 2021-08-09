import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_zoo.BaseModelClass import BaseModel


class CSDN_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(CSDN_conv, self).__init__()
        self.depth_conv = nn.Conv3d(in_channels=in_ch,
                                  out_channels=in_ch,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  groups=in_ch)

        self.point_conv = nn.Conv3d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, padding=1),
            # CSDN_conv(in_ch, out_ch, kernel_size=3, stride=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01),

            nn.Conv3d(out_ch, out_ch, 3, 1, padding=1),
            # CSDN_conv(out_ch, out_ch, kernel_size=3, stride=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.5)
        )


    def forward(self, x):
        x = self.conv(x)

        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            nn.Conv3d(in_ch, out_ch, 2, 2),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01),
            double_conv(out_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        # if mode == 'convtran':
        # self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, 2)
        self.conv = double_conv(in_ch, out_ch)

        # mode = 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        # if mode == 'upsample':
        self.up = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        # self.conv = double_conv(in_ch, out_ch)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffX = x2.size()[3] - x1.size()[4]
        # diffY = x2.size()[3] - x1.size()[4]
        # # print('sizes',x1.size(),x2.size(),diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2)
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):    # out_ch: num_classes
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv(x)
        # x = self.sigm(x)
        # x = self.softmax(x)
        return x
        # return x.clamp_(0, 1)


class UNet_ours(BaseModel):
    def __init__(self, n_channels, n_classes):
        super(UNet_ours, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 16)
        self.up4 = up(32, 16)
        self.outc = outconv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print('x1: ', x1.shape)
        x2 = self.down1(x1)
        # print('x2: ', x2.shape)
        x3 = self.down2(x2)
        # print('x3: ', x3.shape)
        x4 = self.down3(x3)
        # print('x4: ', x4.shape)
        x5 = self.down4(x4)
        # print('x5: ', x5.shape)
        x = self.up1(x5, x4)   # 30 and 64
        # print('x: ', x.shape)
        x = self.up2(x, x3)
        # print('x: ', x.shape)
        x = self.up3(x, x2)
        # print('x: ', x.shape)
        x = self.up4(x, x1)
        # print('x: ', x.shape)
        x = self.outc(x)
        # print('x: ', x.shape)
        return x

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, self.input_channels, 48, 48, 48)
        ideal_out = torch.rand(1, self.num_classes, 48, 48, 48)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        # summary(self.to(torch.device(device)), (self.input_channels, 12, 12, 12),device=device)
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("DenseNet3D-1 test is complete")

def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


net = UNet_ours(n_channels=1, n_classes=1)
net.apply(init)



if __name__ == "__main__":

    net  = UNet_ours(n_channels=1, n_classes=1)
    a, b = net.count_params()
    print(a, b)
    net.apply(init)
    inputs = np.ones((1,1,64,64,64))
    inputs = torch.FloatTensor(inputs)
    net = net.cuda()
    net.train()
    inputs = inputs.cuda()
    outputs = net(inputs)
    print(outputs.shape)
