#-*- coding:utf-8 _*-
import os
import pdb
import argparse
import pickle as pkl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from mmcv import Config
import numpy as np
from dataset import Cifar10
from models import MyModel, vgg16
from log import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #全局变量

def parser():
    parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
    # parse.add_argument('--local_rank',default=0,type=int,help='node rank for distributedDataParallel')
    parse.add_argument('--config','-c',default='./config/config.py',help='config file path')
    # parse.add_argument('--net','-n',type=str,required=True,help='input which model to use')
    # parse.add_argument('--net','-n',default='MyLenet5')
    # parse.add_argument('--pretrain','-p',action='store_true',help='Location pretrain data')
    # parse.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
    # parse.add_argument('--epoch','-e',default=None,help='resume from epoch')
    # parse.add_argument('--gpuid','-g',type=int,default=0,help='GPU ID')
    # parse.add_argument('--NumClasses','-nc',type=int,default=)
    args = parse.parse_args()
    return args

def train_valid(net, dict, criterion, optimizer, train_loader, valid_loader, args, log, cfg, epoches=20):
    best_loss = np.inf
    for epoch in range(epoches):
        net.train()
        train_loss = 0.0
        train_total = 0.0
        for i, data in enumerate(train_loader, 0):
            train_length = len(train_loader)  # length = 54000 / batch_size
            inputs, labels = data #inputs[100,1,28,28] labels[100]没有进行onehot

            # inputs = inputs.view(inputs.size(0),-1) #把图片拉平
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)

        net.eval()
        valid_loss = 0.0
        valid_total = 0.0
        with torch.no_grad():  # 强制之后的内容不进行计算图的构建，不使用梯度反传
            for i, data in enumerate(valid_loader, 0):
                valid_length = len(valid_loader)
                inputs, labels = data
                # inputs = inputs.view(inputs.size(0), -1)  # 把图片拉平
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                # correct += (predicted == labels).sum()
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        log.logger.info('Epoch:%d,Train_Loss:%.5f,Valid_Loss:%.5f '
                        % (epoch + 1, train_loss/train_length, valid_loss/valid_length))

        # 这里不仅需要保存参数，还需呀保存网络架构，即params_dict
        if  (valid_loss / valid_length) < best_loss:
            log.logger.info('Validation loss decreased ({:.5f} --> {:.5f}).  Saving model ...'.format(best_loss, valid_loss/valid_length))
            best_loss = valid_loss / valid_length
            checkpoint = {'net': net.state_dict(), 'epoch': epoch}
            if not os.path.exists(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/'): os.mkdir(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/')
            torch.save(checkpoint, cfg.PARA.utils_paths.checkpoint_path + 'GACNN/' + 'best_ckpt.pth')

            # with open(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/' + 'best_net_params.pkl', 'wb') as f:
            #     pkl.dump(dict, f, pkl.HIGHEST_PROTOCOL)

        with open(cfg.PARA.utils_paths.visual_path + 'GACNN' + '_Loss.txt', 'a') as f:
            f.write('Epoch:%d,Train_Loss:%.5f,Valid_Loss:%.5f '
                        % (epoch + 1, train_loss/train_length, valid_loss/valid_length))
            f.write('\n')
    return best_loss


def test(net, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(test_loader, 0):
            images, labels = data #labels是具体的数值
            images, labels = images.to(device), labels.to(device)
            # images = images.view(images.size(0), -1)  # 把图片拉平
            outputs = net(images) #outputs:[100,10]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()#.item()
        acc = correct.cpu().numpy() / total
    return acc

def Train_GACNN(params_dict, chrom_i, args, cfg, log, epoches=20):
    # args = parser()
    # cfg = Config.fromfile(args.config)
    # log = Logger(cfg.PARA.utils_paths.log_path + 'GA_MyCNN' + '_log.txt', level='info')

    log.logger.info('==> Preparing dataset <==')
    cifar10 = Cifar10(batch_size = cfg.PARA.train.batch_size)
    # subtrain_loader, subvalid_loader = cifar10.Download_SubTrain_SubValid()
    # print(len(subtrain_loader), len(subvalid_loader))

    train_loader, valid_loader = cifar10.Download_Train_Valid()
    test_loader = cifar10.Download_Test()

    log.logger.info('==> Loading model <==')
    # net = vgg16()
    net = MyModel(params_dict, chrom_i)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    print(net)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=cfg.PARA.train.lr)   # , momentum=cfg.PARA.train.momentum

    log.logger.info('==> Waiting Train <==')
    loss = train_valid(net=net, dict=params_dict, criterion=criterion, optimizer=optimizer, train_loader=train_loader,
                       valid_loader=valid_loader, args=args, log=log, cfg=cfg, epoches=epoches)

    # log.logger.info("==> Waiting Test <==")
    # # with open(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/' + 'best_net_params.pkl', 'rb') as f:
    # #     best_net_params = pkl.load(f)
    # checkpoint = torch.load(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/' + 'best_ckpt.pth')

    # # 进行测试时，net 和保存的 net 是不一样的，所以需要重新设置 net    可是之前就可以，说明没问题啊。。。奇怪
    # net2 = MyCNN(best_net_params)
    # net2.to(device)
    # net2.load_state_dict(checkpoint['net'])
    # test_acc = test(net=net2, test_loader=test_loader)

    # net.load_state_dict(checkpoint['net'])
    # test_acc = test(net=net, test_loader=test_loader)
    # log.logger.info('Test ACC = {:.5f}'.format(test_acc))
    # log.logger.info('==> One Train & Valid & Test End <==')
    return loss


if __name__=='__main__':

    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path + 'GA_CNN' + '_test_Train_log.txt', level='info')

    params = {
        'conv': [[64, 64, 'M', 256, 'M'],
                 [64, 'M', 128, 128, 128, 'M'],
              [64, 64, 'M', 128, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M'],
              [64, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 'M'], [64, 64, 64, 'M', 128, 128, 'M'],
              [64, 'M', 128, 'M', 256, 'M'], [64, 64, 'M', 128, 128, 128, 'M', 256, 'M'],
              [64, 64, 64, 'M', 128, 'M', 256, 'M', 512, 512, 'M', 512, 512, 512, 'M'], [64, 'M', 128, 128, 'M'],
              [64, 64, 64, 'M', 128, 'M', 256, 256, 'M', 512, 'M']],
        'fc': [[256, 64],
               [256, 448, 320, 256, 64],
               [512, 512, 256, 64],
               [192, 128, 384, 384],
               [128, 256, 64], [192, 512, 384, 192], [256, 64, 320, 128, 256], [128, 128, 512], [256, 128, 384, 448, 256],
               [256, 384, 512, 384, 320]]}
    Train_GACNN(params, 2, args, cfg, log)
















