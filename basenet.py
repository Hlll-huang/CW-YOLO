import torch
from torch import nn
from torch.nn import functional

#卷积块
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False ):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.sub_module(x)

#残差块
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.sub_module(x)+x

#卷积集合块
class ConvolutionalSetLayer(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ConvolutionalSetLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_module(x)

#下采样
class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)

#上采样
class UpSamplingLayer(nn.Module):
    def __init__(self):
        super(UpSamplingLayer, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")

class Yolo_V3_Net_Hl(nn.Module):
    def __init__(self):
        super(Yolo_V3_Net_Hl, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSamplingLayer(32, 64),

            ResidualLayer(64, 32),

            DownSamplingLayer(64, 128),

            ResidualLayer(128, 64),
            ResidualLayer(128, 64),

            DownSamplingLayer(128, 256),

            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
        )

        self.trunk_26 = nn.Sequential(
            DownSamplingLayer(256, 512),

            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
        )

        self.trunk_13 = nn.Sequential(
            DownSamplingLayer(512, 1024),

            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
        )

        #13X13的检测网络
        self.convert_13 = nn.Sequential(
            ConvolutionalSetLayer(1024, 512)
        )
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 24, 1, 1, 0)
        )

        # 检测26X26的网络
        self.up_13_to_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 3, 1, 1),
            UpSamplingLayer()
        )

        self.convert_26 = nn.Sequential(
            ConvolutionalSetLayer(768, 256)
        )
        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 24, 1, 1, 0)
        )

        #检测52X52的网络
        self.up_26_to_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 3, 1, 1),
            UpSamplingLayer()
        )

        self.convert_52 = nn.Sequential(
            ConvolutionalSetLayer(384, 128)
        )
        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 24, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_13_out = self.convert_13(h_13)
        detection_13_out = self.detection_13(convset_13_out)

        up_13_to_26_out = self.up_13_to_26(convset_13_out)
        cat_13_to_26 = torch.cat((up_13_to_26_out,h_26),dim=1)
        convset_26_out = self.convert_26(cat_13_to_26)
        detection_26_out = self.detection_26(convset_26_out)

        up_26_to_52_out = self.up_26_to_52(convset_26_out)
        cat_26_to_52 = torch.cat((up_26_to_52_out, h_52), dim=1)
        convset_52_out =  self.convert_52(cat_26_to_52)
        detection_52_out = self.detection_52(convset_52_out)

        return detection_13_out,detection_26_out,detection_52_out

if __name__ == '__main__':
    net = Yolo_V3_Net_Hl()
    x = torch.randn(1,3,608,608)
    y = net(x)
    print(y[0].size(),y[1].size(),y[2].size())
