# Author： Dongdong Zhao
# 参照Kits19比赛中 top 1 方案写的网络模型
import torch
import torch.nn as nn
from model_zoo.BaseModelClass import BaseModel

class ResidualBlock_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_down, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down = nn.Conv3d(self.in_channels, self.out_channels, 2, 2)
        self.res_block = ResidualBlock(self.out_channels, self.out_channels)

    def forward(self, x):
        out = self.down(x)
        out = self.res_block(out)
        return out

class ResidualBlock_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = nn.ConvTranspose3d(self.in_channels,self.out_channels, 2, 2)
        self.res_block = ResidualBlock(self.out_channels, self.out_channels)

    def forward(self, x):
        out = self.up(x)
        out = self.res_block(out)
        return out

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.res_conv = nn.Sequential(nn.Conv3d(self.in_channels, self.out_channels, 3, 1, padding=1),
#                                   nn.InstanceNorm3d(self.out_channels),
#                                   nn.ReLU(inplace=True),
#                                   nn.Conv3d(self.out_channels, self.out_channels, 3, 1, padding=1),
#                                   nn.InstanceNorm3d(self.out_channels)
#                                   )
#         self.conv_extral = nn.Conv3d(self.out_channels, self.in_channels, 3, 1, padding=1)
#
#     def forward(self, x):
#         out = self.res_conv(x)
#         if x.shape != out.shape:
#             out = self.conv_extral(out)
#         out = out + x
#         return out
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_conv = nn.Sequential(nn.Conv3d(self.in_channels, self.in_channels // 2, 1, 1),
                                  nn.InstanceNorm3d(self.out_channels),
                                  nn.ReLU(inplace=True),

                                  nn.Conv3d(self.in_channels // 2, self.in_channels // 2, 3, 1, padding=1),
                                  nn.InstanceNorm3d(self.out_channels),
                                  nn.ReLU(inplace=True),

                                  nn.Conv3d(self.in_channels // 2, self.out_channels, 1, 1),
                                  nn.InstanceNorm3d(self.out_channels)
                                  )
        self.conv_extral = nn.Conv3d(self.out_channels, self.in_channels, 3, 1, padding=1)
        self.relu = nn.ReLU(in_channels)
    def forward(self, x):
        out = self.res_conv(x)
        if x.shape != out.shape:
            out = self.conv_extral(out)
        out = out + x
        out = self.relu(out)
        return out

class ResUNet_3D(BaseModel):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResUNet_3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_conv = nn.Sequential(nn.Conv3d(self.in_channels, 30, 3, 1, padding=1),
                                     nn.InstanceNorm3d(30),
                                     nn.ReLU(inplace=True))
        self.residual_block1 = ResidualBlock(30, 30)
        self.down1 = ResidualBlock_down(30, 60)
        self.residual_block2 = ResidualBlock(60, 60)
        self.down2 = ResidualBlock_down(60, 120)
        self.residual_block3 = ResidualBlock(120, 120)
        self.down3 = ResidualBlock_down(120, 240)
        self.residual_block4 = ResidualBlock(240, 240)
        self.down4 = ResidualBlock_down(240, 480)
        self.residual_block5 = ResidualBlock(480, 480)
        self.up1 = ResidualBlock_up(480, 240)
        self.up2 = ResidualBlock_up(480, 120)
        self.up3 = ResidualBlock_up(240, 60)
        self.up4 = ResidualBlock_up(120, 30)
        self.out_conv = nn.Conv3d(60, 1, 1)

    def forward(self, x):
        out1 = self.in_conv(x)  # 对输入进行卷积—>（1， 30， 64， 64， 64）
        out1 = self.residual_block1(out1)
        out2 = self.down1(out1)   # 第一次下采样完成  # （1， 60， 32， 32，32）

        out2 = self.residual_block2(out2)
        out3 = self.down2(out2)    # 第二次下采样完成  # （1， 120， 16， 16，16）

        out3 = self.residual_block3(out3)
        out4 = self.down3(out3)  # 第三次下采样完成  # （1， 240， 8， 8，8）

        out4 = self.residual_block4(out4)
        out5 = self.down4(out4)   # （1， 480， 4， 4，4）

        core_feature = self.residual_block5(out5)   # （1， 480， 4， 4，4）

        out5 = self.up1(core_feature)    # (1, 240, 8, 8, 8)
        out5 = torch.cat((out5, out4), dim=1)   #  # (1, 480, 8, 8, 8)

        out6 = self.up2(out5)    # (1, 120, 16, 16, 16)
        out6 = torch.cat((out6, out3), dim=1)   #(1, 240, 16, 16, 16)

        out7 = self.up3(out6)     # (1, 60, 32, 32, 32)
        out7 = torch.cat((out7, out2), dim=1)  # (1, 120, 32, 32, 32)

        out8 = self.up4(out7)     #(1, 30, 64, 64, 64)
        out8 = torch.cat((out8, out1), dim=1)     #(1, 60, 64, 64, 64)
        out = self.out_conv(out8)
        out = nn.Sigmoid()(out)
        return out

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        # summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("ResUNet_2stage test is complete")

if __name__ == '__main__':
    input = torch.randn(1, 1, 64, 64, 64)
    net = ResUNet_3D()
    out = net(input)
    print(out.shape)