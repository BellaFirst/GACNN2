import sys
import os
import scipy.io as scio
import geatpy as ea
import numpy as np
import time
import pdb
import argparse
from mmcv import Config
from Train import Train_GACNN
from models import MyCNN
from models import MyModel
from log import Logger


""" Example:
    params_dict = {
        'conv_channels':[6, 16],
        'conv_kernel_size':[5, 5],
        'pool_kernel_size':[2],
        'pool_stride':[2],
        'fc_features':[120, 84],
    }
"""

class GA_CNN2:
    def __init__(self, args, cfg, log):
        self.args = args
        self.log = log
        self.cfg = cfg

        '''GA_params'''
        self.MAXGEN = self.cfg.PARA.GA_params.MAXGEN
        self.Nind = self.cfg.PARA.GA_params.Nind
        self.maxormins = self.cfg.PARA.GA_params.maxormins  # -1：最大化 1：最小化
        self.xov_rate = self.cfg.PARA.GA_params.xov_rate  # 交叉概率

        '''chrom'''
        self.params_dict = None
        self.FieldDR = None
        self.chrom_all = None
        self.Objv_all = None


        '''记录每一代的数据'''
        self.obj_trace = np.zeros((self.MAXGEN, 2))  # [MAXGEN, 2] 其中[0]记录当代种群的目标函数均值，[1]记录当代种群最优个体的目标函数值
        self.var_trace = np.zeros((self.MAXGEN, self.cfg.PARA.CNN_params.conv_num[1]+self.cfg.PARA.CNN_params.fc_num[1]+2))  # 记录当代种群最优个体的变量值
        self.time = None

        '''记录所有种群中的最优值'''
        self.best_gen = None
        self.best_Objv = None
        self.best_chrom_i = None
        # self.X = None # 把X和Y用chrom_all和Objv_all代替
        # self.Y = None

    def get_FieldDR(self):
        lb_1 = self.cfg.PARA.CNN_params.conv_num[0]
        ub_1 = self.cfg.PARA.CNN_params.conv_num[1]

        lb_2 = []
        lb_2.append(self.cfg.PARA.CNN_params.conv_output[0])
        lb_2 = lb_2 * ub_1
        ub_2 = []
        ub_2.append(self.cfg.PARA.CNN_params.conv_output[1])
        ub_2 = ub_2 * ub_1

        lb_3 = self.cfg.PARA.CNN_params.fc_num[0]
        ub_3 = self.cfg.PARA.CNN_params.fc_num[1]

        lb_4 = []
        lb_4.append(self.cfg.PARA.CNN_params.fc_output[0])
        lb_4 = lb_4  * ub_3
        ub_4 = []
        ub_4.append(self.cfg.PARA.CNN_params.fc_output[1])
        ub_4 = ub_4 * ub_3

        lb = np.hstack((lb_1, lb_2, lb_3, lb_4))
        ub = np.hstack((ub_1, ub_2, ub_3, ub_4))
        varTypes = [1] * (ub_1 + 1 + ub_3 + 1)
        FieldDR = np.vstack((lb, ub, varTypes))
        self.FieldDR = FieldDR  # 这个是固定不变的

    def get_Objv_i(self, chrom): #输入的chrom个体数 = 每一代种群的个数Nind
        chrom = np.int32(chrom)
        num = chrom.shape[0]
        Objv = np.zeros(shape=(num, 1))
        params_dict = self.chrom2paramsdict(chrom)
        for i in range(num):
            Objv[i] = Train_GACNN(params_dict, i, self.args, self.cfg, self.log)

        # Objv = np.random.rand(num, 1)
        return Objv

    def Evolution(self): #
        start_time = time.time()

        # 初始化种群
        self.get_FieldDR()
        Init_chrom = self.get_init_chrom()

        # 开始进化
        self.log.logger.info('==> This is Init GEN <==' )
        Init_Objv = self.get_Objv_i(Init_chrom)
        best_ind = np.argmax(Init_Objv * self.maxormins) #记录最优个体的索引值

        self.chrom_all = Init_chrom
        self.Objv_all = Init_Objv

        for gen in range(self.MAXGEN):
            self.log.logger.info('==> This is No.%d GEN <==' % (gen))

            if gen==0: #第一代和后面有所不同
                chrom = Init_chrom
                Objv = Init_Objv

            else:
                chrom = NewChrom
                Objv = NewObjv

            FitnV = ea.ranking(Objv * self.maxormins)
            Selch = chrom[ea.selecting('rws', FitnV, self.Nind-1), :] #轮盘赌选择 Nind-1 代，与上一代的最优个体再进行拼接
            Selch = ea.recombin('xovsp', Selch, self.xov_rate) #重组，即交叉
            Selch = ea.mutate('mutswap', 'RI', Selch, self.FieldDR) #变异
            Objv_Selch = self.get_Objv_i(Selch)

            NewChrom = np.vstack((chrom[best_ind, :], Selch)) #将上一代的最优个体与现在的种群拼接
            NewObjv =np.vstack((Objv[best_ind, :], Objv_Selch))
            best_ind = np.argmax(NewObjv * self.maxormins)

            self.chrom_all = np.vstack((self.chrom_all, NewChrom))
            self.Objv_all = np.vstack((self.Objv_all, NewObjv))

            self.obj_trace[gen, 0] = np.sum(NewObjv) / self.Nind  # 记录当代种群的目标函数均值
            self.obj_trace[gen, 1] = NewObjv[best_ind]  # 记录当代种群最有给他目标函数值
            self.var_trace[gen, :] = NewChrom[best_ind, :]  # 记录当代种群最优个体的变量值
            self.log.logger.info('GEN=%d,best_Objv=%.5f,best_chrom_i=%s\n'
                                 % (gen, NewObjv[best_ind], str(NewChrom[best_ind, :])))  # 记录每一代的最大适应度值和个体

        self.Save_chroms(self.chrom_all)
        self.Save_objvs(self.Objv_all)

        end_time = time.time()
        self.time = end_time - start_time
        self.log.logger.info('The time of Evoluation is %.5f s. ' % self.time)

    def Plot_Save(self):
        self.best_gen = np.argmax(self.obj_trace[:, [1]])
        self.best_Objv = self.obj_trace[self.best_gen, 1]
        self.best_chrom_i = self.var_trace[self.best_gen]

        # pdb.set_trace()
        ea.trcplot(trace=self.obj_trace,
                   labels=[['POP Mean Objv', 'Best Chrom i Objv']],
                   titles=[['Mean_Best_Chromi_Objv']],
                   save_path=self.cfg.PARA.CNN_params.save_data_path,
                   xlabels=[['GEN']],
                   ylabels=[['ACC']])

        with open(self.cfg.PARA.CNN_params.save_bestdata_txt, 'a') as f:
            f.write('best_Objv=%.5f,best_chrom_i=%s,total_time=%.5f\n' % (
            self.best_Objv, str(self.best_chrom_i), self.time))

        np.savetxt(self.cfg.PARA.CNN_params.save_data_path + 'MeanChromi_Objv.txt', self.obj_trace[:, 0])
        np.savetxt(self.cfg.PARA.CNN_params.save_data_path + 'BestChromi_Objv.txt', self.obj_trace[:, 1])
        np.savetxt(self.cfg.PARA.CNN_params.save_data_path + 'BestChromi.txt', self.var_trace)

    def Save_chroms(self, chrom):
        self.log.logger.info('==> Save Chroms to file <==')
        scio.savemat(self.cfg.PARA.CNN_params.save_x_mat, {"chrom": chrom})

    def Save_objvs(self, Objv):
        self.log.logger.info('==> Save Objvs to file <==')
        scio.savemat(self.cfg.PARA.CNN_params.save_y_mat, {"objv": Objv})

    def chrom2conv(self, chrom):
        convs_all = []
        for i in range(chrom.shape[0]): # 0-9 == Nind
            convs = []
            for j in range(chrom.shape[1] - 1):  # 0-5 ---> 0-4
                j = j + 1  # 1-5
                if j < 4:
                    conv_num = 64 * 2 ** (j - 1)
                else:
                    conv_num = 512
                if chrom[i, j] != 0:
                    for t in range(chrom[i, j]):
                        convs.append(conv_num)
                    convs.append('M')
            # print(convs)
            convs_all.append(convs)
        return convs_all

    def chrom2fc(self, chrom):
        fcs_all = []
        for i in range(chrom.shape[0]):  # 0-9 == Nind
            fcs = []
            for j in range(chrom.shape[1]):  # 0-4
                if chrom[i, j] != 0:
                    hidden_neurons = 64 * chrom[i, j]
                    fcs.append(hidden_neurons)
            fcs_all.append(fcs)
        return fcs_all

    def get_chrom_conv(self, param_1, param_2, Nind):
        """
        param_1: 为卷积模块个数 [0]为下限，[1]为上限
        param_2: 卷积模块中对应的 out_channels
        Nind: 一代种群的大小
        最后得到chrom
        """
        chrom_conv = np.zeros(shape=(Nind, param_1[1]+1), dtype=int)

        for i in range(Nind):
            num_param_1 = np.random.randint(low=param_1[0], high=param_1[1]+1, dtype=int)  # 不包括high
            num_param_2 = np.random.randint(low=param_2[0], high=param_2[1]+1, size=(num_param_1), dtype=int)
            chrom_conv[i, 0] = num_param_1
            chrom_conv[i, 1:num_param_1 + 1] = num_param_2

        return chrom_conv

    def get_chrom_fc(self, param_1, param_2, Nind):
        """
        param_1: fc 层数 2-5 [0]为下限，[1]为上限
        param_2: fc 中每层的神经元个数 [1,8] 在转换为fc时需要 * 64
        Nind: 一代种群的大小
        最后得到chrom_fc
        """
        chrom_fc = np.zeros(shape=(Nind, param_1[1]+1), dtype=int)

        for i in range(Nind):
            num_param_1 = np.random.randint(low=param_1[0], high=param_1[1]+1, dtype=int)  # 不包括high
            num_param_2 = np.random.randint(low=param_2[0], high=param_2[1]+1, size=(num_param_1), dtype=int)
            chrom_fc[i, 0] = num_param_1
            chrom_fc[i, 1:num_param_1 + 1] = num_param_2

        return chrom_fc

    def get_init_chrom(self):
        conv_num = self.cfg.PARA.CNN_params.conv_num
        conv_output = self.cfg.PARA.CNN_params.conv_output
        fc_num = self.cfg.PARA.CNN_params.fc_num
        fc_output = self.cfg.PARA.CNN_params.fc_output

        chrom_conv = self.get_chrom_conv(conv_num, conv_output, self.Nind)
        chrom_fc = self.get_chrom_fc(fc_num, fc_output, self.Nind)
        chrom = np.hstack((chrom_conv, chrom_fc))
        return chrom


    def chrom2paramsdict(self, chrom):
        params_dict = {}
        chrom_conv = chrom[:, 0:self.cfg.PARA.CNN_params.conv_num[1]+1]
        chrom_fc = chrom[:, self.cfg.PARA.CNN_params.conv_num[1]+1:]
        params_dict['conv'] = self.chrom2conv(chrom_conv)
        params_dict['fc'] = self.chrom2fc(chrom_fc)
        return params_dict




def parser():
    parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
    parse.add_argument('--config','-c',default='../config/config.py',help='config file path')
    args = parse.parse_args()
    return args


def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path + 'GA_CNN' + '_testlog.txt', level='info')

    ga = GA_CNN2(args, cfg, log)
    chrom = ga.get_init_chrom()
    print(chrom)
    params = ga.chrom2paramsdict(chrom)
    print(params)
    # print(params['conv'])

    # ga.Evolution()
    # ga.Plot_Save()
    # print(ga.obj_trace)
    # print('*'*20)
    # print(ga.obj_trace[:, 1])
    # print('*' * 20)
    # print(ga.var_trace)

if __name__ == '__main__':
    main()


