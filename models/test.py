import torch
from torch import nn
import math

class CCEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CCEM, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()

    def forward(self, x):

        a = self.conv(self.gmp(x)+self.gap(x))
        b = self.bn(a)
        return a.shape,b.shape

if __name__ == '__main__':
    """a = torch.randn(2,3,12,12)
    model = CCEM(3,3)
    y1, y2 = model(a)
    print(y1)111
    print(y2)"""
    m = math.log(1)
    x = -2
    for i in range(1,11):
        print(math.log10(i))