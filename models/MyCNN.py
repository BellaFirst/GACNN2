import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class MyCNN(nn.Module):
    def __init__(self, params_dict): #params_dict是一个参数字典，包括多个参数，{'conv_channels'--2, 'conv_kernel_size'--2, 'pool_kernel_size'--1, 'pool_stride'--1, 'fc_features'--2}
        super(MyCNN, self).__init__()
        """
        conv1 --> pool --> conv2 --> pool --> fc1 -> fc2 -> fc3
        
        """
        self.pmd = params_dict
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.pmd['conv_channels'][0], kernel_size=self.pmd['conv_kernel_size'][0])
        self.conv2 = nn.Conv2d(in_channels=self.pmd['conv_channels'][0], out_channels=self.pmd['conv_channels'][1], kernel_size=self.pmd['conv_kernel_size'][1])
        self.pool = nn.MaxPool2d(kernel_size=self.pmd['pool_kernel_size'], stride=self.pmd['pool_stride'])

        temp = (32 - self.pmd['conv_kernel_size'][0] + 2*0) / 1 + 1
        temp = int((temp - self.pmd['pool_kernel_size']) / self.pmd['pool_stride'] + 1)
        temp = (temp - self.pmd['conv_kernel_size'][1] + 2 * 0) / 1 + 1
        temp = int((temp - self.pmd['pool_kernel_size']) / self.pmd['pool_stride'] + 1)
        temp = self.pmd['conv_channels'][1] * temp * temp
        self.temp = int(temp)

        self.fc1 = nn.Linear(in_features=self.temp,
                             out_features=self.pmd['fc_features'][0])
        self.fc2 = nn.Linear(in_features=self.pmd['fc_features'][0], out_features=self.pmd['fc_features'][1])
        self.fc3 = nn.Linear(in_features=self.pmd['fc_features'][1], out_features=10)

    def forward(self, x):
        # pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # pdb.set_trace()
        # x = x.view(-1, self.pmd['conv_channels'][1]*self.pmd['conv_kernel_size'][1]*self.pmd['conv_kernel_size'][1])
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__=='__main__':
    d = {
        'conv_channels': [6, 16],
        'conv_kernel_size': [5, 5],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc_features': [120, 84],
    }

    net = MyCNN(d)
    print(net)






