import numpy as np
import pdb
import scipy.io as scio

def get_x_y(filename, num_params, num_lines):
    """
    filename: 文件保存的位置及名字
    num_params: 每一行中参数的个数
    num_lines: 一共有多少行，代表参数的总个数
    """
    x1, x2, x3, y = [], [], [], []

    fh = open(filename, 'r')
    for line in fh:
        line = line.strip('\n')
        line = line.replace('[', '').replace(']', '') #进行一些处理，可能不需要，看情况而定。
        words = line.split(',')  #共得到num_params个参数


        neurons = words[0].split('=')[1].split(' ')  # 将neurons转化为分开转换
        temp_hn = []
        for n in neurons: temp_hn.append(int(n))

        x1.append(temp_hn)
        x2.append(float(words[1].split('=')[1]))
        x3.append(int(words[2].split('=')[1]))
        y.append(float(words[3].split('=')[1]))

    X1, X2, X3, Y = np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(y)

    X = np.hstack((X1, X2.reshape((-1,1)), X3.reshape((-1,1))))
    Y = Y.reshape((-1,1))

    scio.savemat('../cache/data/X.mat', {"X": X})
    scio.savemat('../cache/data/Y.mat', {"Y": Y})



