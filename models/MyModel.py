import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from typing import Union, List, Dict, Any, cast

class MyModel(nn.Module):
    def __init__(self, params_dict, chrom_i):
        super(MyModel, self).__init__()
        self.params_dict = params_dict
        self.chrom_i = chrom_i
        self.feature = self.make_conv_layers(self.params_dict['conv'][self.chrom_i]) #[]
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifer = self.make_fc_layers(self.params_dict['fc'][self.chrom_i]) #[]

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifer(x)
        return x

    def make_fc_layers(self, cfg):
        layers = []
        in_channels = self.params_dict['conv'][self.chrom_i][-2] * 7 * 7
        for v in cfg:
            layers += [nn.Linear(in_channels, v),
                       nn.ReLU(True),
                       nn.Dropout()]
            in_channels = v
        layers += [nn.Linear(v, 10)]
        return nn.Sequential(*layers)

    def make_conv_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)  # 确认v是int类型，并返回v
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                               nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                    in_channels = v
        return nn.Sequential(*layers)


def chrom2conv(chrom):
    convs_all = []
    for i in range(chrom.shape[0]): #0-9 == Nind
        convs = []
        for j in range(chrom.shape[1] - 1): #0-5 ---> 0-4
            j = j + 1  # 1-5
            if j < 4: conv_num = 64 * 2 ** (j - 1)
            else: conv_num = 512
            if chrom[i, j]!=0:
                for t in range(chrom[i, j]):
                    convs.append(conv_num)
                convs.append('M')
        # print(convs)
        convs_all.append(convs)
    return convs_all

def chrom2fc(chrom):
    fcs_all = []
    for i in range(chrom.shape[0]): #0-9 == Nind
        fcs = []
        for j in range(chrom.shape[1]): #0-4
            if chrom[i,j] != 0:
                hidden_neurons = 64 * chrom[i, j]
                fcs.append(hidden_neurons)
        fcs_all.append(fcs)
    return fcs_all

def get_chrom_conv(param_1, param_2, Nind):
    """
    param_1: 为卷积模块个数 [0]为下限，[1]为上限
    param_2: 卷积模块中对应的 out_channels
    Nind: 一代种群的大小
    最后得到chrom
    """
    chrom_conv = np.zeros(shape=(Nind, param_1[1]), dtype=int)

    for i in range(Nind):
        num_param_1 = np.random.randint(low=param_1[0], high=param_1[1], dtype=int) #不包括high
        num_param_2 = np.random.randint(low=param_2[0], high=param_2[1], size=(num_param_1), dtype=int)
        chrom_conv[i, 0] = num_param_1
        chrom_conv[i, 1:num_param_1+1] = num_param_2

    return chrom_conv

def get_chrom_fc(param_1, param_2, Nind):
    """
    param_1: fc 层数 2-5 [0]为下限，[1]为上限
    param_2: fc 中每层的神经元个数 [1,8] 在转换为fc时需要 * 64
    Nind: 一代种群的大小
    最后得到chrom_fc
    """
    chrom_fc = np.zeros(shape=(Nind, param_1[1]), dtype=int)

    for i in range(Nind):
        num_param_1 = np.random.randint(low=param_1[0], high=param_1[1], dtype=int)  # 不包括high
        num_param_2 = np.random.randint(low=param_2[0], high=param_2[1], size=(num_param_1), dtype=int)
        chrom_fc[i, 0] = num_param_1
        chrom_fc[i, 1:num_param_1+1] = num_param_2

    return chrom_fc

def get_chrom_cfgs(conv_num=[2,5], conv_output=[1,3], fc_num=[2,5], fc_output=[1,8], Nind=10):
    # conv_num = cfg.PARA.CNN_params.conv_num
    # conv_output = cfg.PARA.CNN_params.conv_output
    # fc_num = cfg.PARA.CNN_params.fc_num
    # fc_output = cfg.PARA.CNN_params.fc_output

    chrom_conv = get_chrom_conv(conv_num, conv_output, Nind)
    chrom_fc = get_chrom_fc(fc_num, fc_output, Nind)
    chrom = np.hstack((chrom_conv, chrom_fc))

    cfgs = {}
    convs = chrom2conv(chrom_conv)
    fcs = chrom2fc(chrom_fc)
    cfgs['conv'] = convs
    cfgs['fc'] = fcs
    return chrom, cfgs



if __name__=='__main__':
    # d = {
    #     'conv_channels': [6, 16],
    #     'conv_kernel_size': [5, 5],
    #     'pool_kernel_size': 2,
    #     'pool_stride': 2,
    #     'fc_features': [120, 84],
    # }
    #
    #
    # net = MyCNN(d)
    # print(net)

    # chrom, cfgs = get_chrom_cfgs()
    # print(chrom)
    # print(chrom.shape)
    # print(cfgs)
    #
    # net = MyModel(cfgs, 0)
    # print(net)

    a = 2
    b = list(a)
    print(b)












