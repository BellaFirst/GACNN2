#-*- coding:utf-8 _*-
import os
import pdb
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mmcv import Config
import numpy as np
from dataset import Cifar10
from log import Logger
from GA import GA_CNN2
from GA.GA_CNN2 import *
from Rand_POP import Rand_POP_CNN2
import geatpy as ea

'''
    主函数
    总体实现所有的功能
    1. args, cfg, log
    2. 开始进化

'''

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

def GA_CNN2_MAIN():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path + 'GA_CNN' + '_log_new.txt', level='info')


    # GA_POP
    gacnn = GA_CNN2(args, cfg, log)
    # gacnn.Init_Chrom()#初始化种群
    gacnn.Evolution() #开始进化
    gacnn.Plot_Save() #绘图并保存数据

    # Rand_POP
    randcnn = Rand_POP_CNN2(args, cfg, log)
    randcnn.Evolution()

def GA_CNN2_BEST():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path + 'GA_CNN' + '_log_new_best.txt', level='info')

    '''
    best_Objv = 1.64425
    best_chrom_i = [1. 2. 0. 0. 0. 0. 1. 6. 0. 0. 0.]
    
    经过多次epoches，最终的loss为0.75，对应的精度约为70%以上。
    '''
    best_chrom_i = np.array([[1, 2, 0, 0, 0, 0, 1, 6, 0, 0, 0]], dtype=int)
    print(best_chrom_i)


    gacnn = GA_CNN2(args, cfg, log)
    best_paramsdict = gacnn.chrom2paramsdict(best_chrom_i)
    print(best_paramsdict)
    best_i_objv = Train_GACNN(best_paramsdict, 0, args, cfg, log,epoches=100)
    print(best_i_objv)


if __name__ == '__main__':
    # GA_CNN2_MAIN()
    GA_CNN2_BEST()





















