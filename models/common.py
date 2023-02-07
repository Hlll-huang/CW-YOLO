# This file contains modules common to various models

import math
import warnings
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

from PIL import Image
from google.auth.transport import requests
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


import torch
from torch import nn
from torch.nn import functional

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.Hardswish() if act else nn.Identity()  #nn.Identity表示不使用激活函数
        self.act = nn.SiLU() if act else nn.Identity()   # 更改

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        # ”*“操作符可以把一个list拆成一个独立的元素
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x = torch.cat(x, self.d)
        return x


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)

# 此类未用
class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model

    def forward(self, x, size=640, augment=False, profile=False):
        # supports inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   opencv:     x = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:        x = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:      x = np.zeros((720,1280,3))  # HWC
        #   torch:      x = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:   x = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(x, torch.Tensor):  # torch
            return self.model(x.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        if not isinstance(x, list):
            x = [x]
        shape0, shape1 = [], []  # image and inference shapes
        batch = range(len(x))  # batch size
        for i in batch:
            x[i] = np.array(x[i])[:, :, :3]  # up to 3 channels if png
            s = x[i].shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(x[i], new_shape=shape1, auto=False)[0] for i in batch]  # pad
        x = np.stack(x, 0) if batch[-1] else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        x = self.model(x, augment, profile)  # forward
        x = non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in batch:
            if x[i] is not None:
                x[i][:, :4] = scale_coords(shape1, x[i][:, :4], shape0[i])
        return x


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


#用于第二级的分类
class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


"""
@huanglu 2021/12/13
卷积要写成这样子:Conv_BN_Act
"""
# CCEM模块
class CCEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CCEM, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        #self.conv = Conv(in_channels, out_channels, 1, 1)

    def forward(self, x):
        weights_matrix = self.act(self.conv(self.gmp(x)+self.gap(x)))
        #weights_matrix = self.act(self.bn(self.conv(self.gmp(x) + self.gap(x))))
        #weights_matrix = self.conv1(self.gmp(x) + self.gap(x))

        return x*weights_matrix


class CIEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CIEM, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        #self.conv = Conv(in_channels, out_channels, 1, 1)

    def forward(self, x):
        weights_matrix = self.act(self.conv(self.gmp(x)+self.gap(x)))
        #weights_matrix = self.act(self.bn(self.conv(self.gmp(x) + self.gap(x))))
        #weights_matrix = self.conv1(self.gmp(x) + self.gap(x))

        return x*weights_matrix

class NewCCEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NewCCEM, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, 4*in_channels, 1, 1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4*in_channels, out_channels, 1, 1)
        self.act2 = nn.Sigmoid()
        #self.conv = Conv(in_channels, out_channels, 1, 1)

    def forward(self, x):
        weights_matrix = self.act2(self.conv2(self.act1(self.conv1(self.gmp(x)+self.gap(x)))))
        #weights_matrix = self.conv(self.gmp(x) + self.gap(x))
        return x*weights_matrix

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
       # self.out_channels = out_channels

    def forward(self, x):
        result = torch.zeros_like(x[0])
        for t in x:
            result += t
        return result

class WFFM(nn.Module):
    def __init__(self, a, b, c):  # a,b,c的值相加等于1且在（0，1）之间 注：还有优化空间(取值的判断)
        super(WFFM, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        x = self.a*x[0]+self.b*x[1]+self.c*x[2]
        return x

class WFFM_LEARNED(nn.Module):
    """把平衡因子变为可学习的参数，减少手动验证代码的复杂工作量。"""
    def __init__(self, a, b, c):
        super(WFFM_LEARNED, self).__init__()
        self.a = nn.Parameter(torch.tensor([a]), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([b]), requires_grad=True)
        self.c = nn.Parameter(torch.tensor([c]), requires_grad=True)

    def forward(self, x):
        x = self.a*x[0]+self.b*x[1]+self.c*x[2]
        return x

class WFFM_LEARNED_SOFTMAX(nn.Module):
    """把平衡因子变为可学习的参数，减少手动验证代码的复杂工作量。"""
    def __init__(self):
        super(WFFM_LEARNED_SOFTMAX, self).__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]), requires_grad=True)
        self.weight_sm = F.softmax(self.weight, dim=-1)

    def forward(self, x):
        x = self.weight_sm[0]*x[0]+self.weight_sm[1]*x[1]+self.weight_sm[2]*x[2]
        return x

"""
    huanglu  **ParNet
"""
class SSE(nn.Module):  #输入的通道和输出的通道要一致
    def __init__(self, in_channels, out_channels):
        super(SSE, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.sse_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bn(x)
        return torch.mul(x, self.sse_module(x))

class Par_Block(nn.Module): # 由于SSE的存在，则输入的通道和输出的通道一致
    def __init__(self, in_channels, out_channels):
        super(Par_Block, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.sse = nn.Sequential(
            SSE(in_channels, out_channels)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv_1x1(x)+self.conv_3x3(x)+self.sse(x))

#可控制深度的Par_n_Block
class Par_n_Block(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super(Par_n_Block, self).__init__()
        self.n_par_block = nn.Sequential(*[Par_Block(in_channels, out_channels) for _ in range(n)])

    def forward(self, x):
        return self.n_par_block(x)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()
        self.down_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.sse_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1),  # 这里更改过，以前的步长是2
            nn.Sigmoid()
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act((self.down_1(x)+self.down_2(x))*self.sse_3(x))

"Fuse_huanglu  不改变尺度大小,通道数量不变；；；注：3x3卷积，s=1，输出是多少"
"原始的fuse就是一个concat+shuffle+downsampling"
"""class Fuse(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(Fuse, self).__init__()
        self.dim = dim
        self.shuffle = nn.ChannelShuffle(1)
        self.drop = nn.Dropout(0.1)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, groups=2),
            nn.BatchNorm2d(out_channels)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, groups=2),
            nn.BatchNorm2d(out_channels)
        )
        self.branch_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, groups=2),
            nn.Sigmoid()
        )
        self.act = nn.SiLU()

    def forward(self, x):           # 这里的x是list类型
        x = torch.cat(x, self.dim)         # 按第一个维度进行拼接（NxCxWxH）
        x = self.shuffle(x)
        return self.act((self.branch_1(x)+self.branch_2(x))*self.branch_3(x))
"""
# cat+1x1卷积-》保持输入通道和输出的通道一致
class Fuse(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super(Fuse, self).__init__()
        self.dim = dim
        #self.shuffle = nn.ChannelShuffle(1)  # BCWH 按通道的维度打乱
        self.dropblock = DropBlock(5, 0.5)
        self.conv = Conv(in_channels, out_channels, 1, 1, g=1)

    def forward(self, x):  # 这里的x是list类型
        x = torch.cat(x, self.dim)  # 按第一个维度进行拼接（NxCxWxH）

        x = x[:, torch.randperm(x.size(1))]  # 按照通道随机排列
        #x = self.shuffle(x)
        #x = self.dropblock(x)   # 使用dropblock丢弃一些神经元
        return self.conv(x)

#no 3x3卷积的fuse
class Fuse_no_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(Fuse_no_3x3, self).__init__()
        self.dim = dim
        self.shuffle = nn.ChannelShuffle(1)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, groups=2),
            nn.BatchNorm2d(out_channels)
        )
        self.branch_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, groups=2),
            nn.Sigmoid()
        )
        self.act = nn.SiLU()

    def forward(self, x):           # 这里的x是list类型
        x = torch.cat(x, self.dim)         # 按第一个维度进行拼接（NxCxWxH）
        x = self.shuffle(x)
        return self.act(self.branch_1(x)*self.branch_3(x))


class DropBlock(nn.Module):  # 在训练的时候使用，推理的时候不使用，故需要增加判断
    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x):
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

if __name__ == '__main__':
    ##x1 = torch.randn(3, 2, 4)
    ##x2 = torch.randn(3, 2, 4)

    ##w = torch.randn(1,2,4)

    # print("y.size:", (x1*torch.sigmoid(w)).size())  # 验证空间注意力的可行性
    #print("x1[0][0][0]",x1[0][0][0])
    #print("x2[0][0][0]", x2[0][0][0])
    ##list = []
    ##list.append(x1)
    #3list.append(x2)
    #for i, t in enumerate(list):
    #b = torch.zeros(x1.size())
    #print("b.size:", b.size())
   # a = 0.4*list[0]+0.3*list[1]
   # print(a.size(),a[0][0][0])
    #for t in list:
        #print(list[i])
        #b = b + list[i]
      #  b += t

   # print(a[0][0][0])

    """import torchvision.transforms as T

    # 获取图像
    r = "E:/huanglu_lab/yolov5-3.1/inference/images/000000000536.jpg"

    img = Image.open(r)
    tr = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    x = tr(img)
    drop_block = DropBlock(block_size=19, p=0.8)
    x_drop = drop_block(x)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    plt.axis("off")
    axs[1].imshow(x_drop[0, :, :].squeeze().numpy())
    plt.axis("off")
    plt.show()"""

    a = torch.randn(6, 3, 14, 14)
    ad = torch.nn.AdaptiveAvgPool2d(1)
    conv = nn.Conv2d(3, 64, 1, 1)
    # model = Downsampling(3,64)
    #model = conv(a)
    a = ad(a)
    print("a.shape:",a.shape)
    y = conv(a)
    print("y.shape:",y.shape)