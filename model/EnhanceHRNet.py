from torch.utils.checkpoint import checkpoint
from thop import profile
from .hrnet import HighResolutionNet
from .modules import *
import torch
import torch.nn as nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
            has_bn=True, has_relu=True, efficient=False,groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding,groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x

class PRM(nn.Module):

    def __init__(self, output_chl_num, efficient=False):
        super(PRM, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_prm_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
                stride=1, padding=1, has_bn=True, has_relu=True,
                efficient=efficient)
        self.conv_bn_relu_prm_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.conv_bn_relu_prm_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.sigmoid2 = nn.Sigmoid()
        self.conv_bn_relu_prm_3_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.conv_bn_relu_prm_3_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                stride=1, padding=4, has_bn=True, has_relu=True,
                efficient=efficient,groups=self.output_chl_num)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out
        out_2 = torch.nn.functional.adaptive_avg_pool2d(out_1, (1,1))
        out_2 = self.conv_bn_relu_prm_2_1(out_2)
        out_2 = self.conv_bn_relu_prm_2_2(out_2)
        out_2 = self.sigmoid2(out_2)
        out_3 = self.conv_bn_relu_prm_3_1(out_1)
        out_3 = self.conv_bn_relu_prm_3_2(out_3)
        out_3 = self.sigmoid3(out_3)
        out = out_1.mul(1 + out_2.mul(out_3))
        return out

class FPN(nn.Module):
    def __init__(self,in_channels,blocks=4):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        # self.block0 = BasicBlock(self.in_channels,self.in_channels)
        self.block1 = BasicBlock(self.in_channels,self.in_channels)
        self.down1 = nn.Conv2d(self.in_channels,self.in_channels*2,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channels*2)
        self.block2 = BasicBlock(self.in_channels*2,self.in_channels*2)
        self.down2 = nn.Conv2d(self.in_channels*2,self.in_channels*4,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(self.in_channels * 4)
        self.block3 = BasicBlock(self.in_channels*4,self.in_channels*4)
        self.down3 = nn.Conv2d(self.in_channels*4,self.in_channels*8,kernel_size=3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(self.in_channels * 8)

    def forward(self, x):
        res = list()
        # x = self.block0(x)
        res.append(x)
        x1 = self.block1(x)
        x1 = self.down1(x1)
        x1 = self.bn1(x1)
        res.append(x1)
        x2 = self.block2(x1)
        x2 = self.down2(x2)
        x2 = self.bn2(x2)
        res.append(x2)
        x3 = self.block3(x2)
        x3 = self.down3(x3)
        x3 = self.bn3(x3)
        res.append(x3)
        return res

class EnhanceHRNet(nn.Module):
    def __init__(self, num_joints=17):
        super(EnhanceHRNet, self).__init__()
        self.stem = Stem(3,32,32)
        self.FPN = FPN(32)
        # 载入hrnet原模型
        self.hrnet = HighResolutionNet()
        # weights_path = '../pose_hrnet_w32_256x192.pth'
        # assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
        # self.hrnet.load_state_dict(torch.load(weights_path, map_location='cuda', weights_only=True))
        self.hrnet.to(device)
        self.PRM = PRM(output_chl_num=32)
        self.final_layer = nn.Conv2d(32, num_joints, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.stem(x)
        x_list = self.FPN(x)
        hr_out = self.hrnet(x_list)
        hr_out = self.PRM(hr_out)
        hr_out = self.final_layer(hr_out)
        return hr_out

if __name__ == '__main__':
    model = EnhanceHRNet()
    x = torch.randn(2, 3, 256, 192).cuda()
    # x = torch.randn(2, 3, 384, 288).cuda()
    batch = x.shape[0]
    model.to(device)
    model(x)
    # print(hr_out.shape)
    # 统计参数总量
    flops, params = profile(model, inputs=(x,))
    print('FLOPs = ' + str((flops / 1000 ** 3) / batch) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')





