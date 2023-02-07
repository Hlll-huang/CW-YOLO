#detect和Model的构建代码
import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import time
import math

from thop import clever_format
from torch.profiler import profile
from torchsummary import summary

"""
    调试的时候设置断点，F8下一步
"""
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from models.common import *
from models.experimental import MixConv2d, CrossConv
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer   ch指有128，256，512
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor   VOC：20+5
        self.nl = len(anchors)  # number of detection layers = 3 表示有三个尺度
        self.na = len(anchors[0]) // 2  # number of anchors  =3 表示每个尺度上anchors数为三
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # 模型需要保持的参数有两种：反向传播需要被优化器更新的，称parameter；另一种是，反向传播中不需要被优化器更新的，称buffer
        # 第二种参数需要创建tensor，然后将tensor通过register_buffer（）注册
        # 可以通过model.buffers()返回，注册完后参数也会自动保存到OrderDict中
        # 注：buffer的更新在forward中，optim.step只更新nn.parameter类型的参数
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # BCWH调整BWHC

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no)) # 预测框的坐标信息

        return x if self.training else (torch.cat(z, 1), x) # 根据判断是否做训练还是测试，返回预测框坐标信息

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # 划分单元网格
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='ccem_wffm_yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="utf-8") as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # -1表示获取Detect()组件。。。。model的最后一层就是detect模块
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            # m.stride是下采样的次数【8，16，32】
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)   # 检查anchor和stride顺序是否一致
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:   # augment 在test| detect时候做TTA数据增强，但是会增加检测时间
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop # 估算pytorch模型FLOPS的模块
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                #print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            with torch.no_grad():
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)

# 解析网络模型配置文件并构造网络模型
def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # 将模型结构的depth_multiple width_multiple提取出，赋值
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings   eval表示字符类型的转化
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        # 控制深度的代码
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m in [Conv, Bottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, CCEM, CIEM, NewCCEM, Par_Block, Downsampling, Par_n_Block]:
            c1, c2 = ch[-1 if f == -1 else f+1], args[0]   # 输入的通道,输出的通道
            #logger.info((c1, c2))
            # 控制宽度（卷积核的个数）的代码
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]  # 每个模块所有的参数
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)   # 插入深度改变的参数
                n = 1
            #if m is Par_n_Block:
            #    logger.info(args)
        elif m in [WFFM, WFFM_LEARNED]:
            args = [*args]  # 每个模块所有的参数
            #logger.info(args)
        elif m is Fuse:  # 原始parnet的fuse
            c1, c2 = sum([ch[-1 if x == -1 else x + 1] for x in f]), args[0]
            # 控制宽度（卷积核的个数）的代码
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]  # 模块所有的参数
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in [Add, WFFM_LEARNED_SOFTMAX]:
            pass
        #elif m is CCEM:
            # CCEM如果指某一层（不是-1层），那么需要把ch[f+1]，但是这种写法要保证，CCEM的from参数不是-1。
            # 另一种方法：写在最开始定定义的时候，但是需要判断from的参数是否是-1
            # c1, c2 = ch[f+1], args[0]  # 输入的通道,输出的通道
           # logger.info(c1)
            # 控制宽度（卷积核的个数）的代码
            # c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
           # args = [c1, c2, *args[1:]]  # 每个模块所有的参数
        else:
            c2 = ch[-1 if f == -1 else f+1]
            #logger.info(c2)

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type t :"model.common.focus......"
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cw_yolo/cw_yolox.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)

    from pytorch_benchmark import benchmark

    sample = torch.randn(1, 3, 640, 640)

    #summary(model, (3, 640, 640))
    #print("model:", summary(model, (3, 640, 640)))

    "huanglu 2021.12.11"
    # 打印model
    #print("------------Model------------\n", model)

    start_time = time.time()
    x = torch.randn(4, 3, 608, 608).to(device)
    y = model(x)
    end_time = time.time()
    print("模型输出：", y[0].size(), y[1].size(), y[2].size())
    print("运行时间：", end_time-start_time)
    #model.train()

    """# Profile
    img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    #img = torch.randn( 3, 608, 608).to(device)
    y = model(img, profile=True)

    # Tensorboard
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter()
    print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    tb_writer.add_graph(model.model, img)  # add model to tensorboard
    tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard"""
