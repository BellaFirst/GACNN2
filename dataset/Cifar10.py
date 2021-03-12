import torch.nn as nn
import pdb
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader,random_split
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
"""
CIFAR10:
(3,32,32),
train_data:60000
test_data: 10000
"""

class Cifar10():
    def __init__(self, batch_size):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#三通道，三个均值，三个方差
        ])

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.root = '../../DATASET/Cifar10' #根目录下的DATASET
        self.download = True
        self.batch_size = batch_size
        self.train_rate = 0.9

        self.subtrain_length = 6400 #10000
        self.subvalid_length = 1280 #2000

        self.cifar10_train_loader = None
        self.cifar10_valid_loader = None
        self.cifar10_test_loader = None

    def Download_Train_Valid(self):
        train_dataset = datasets.CIFAR10(root=self.root,train=True,transform=self.tf,download=self.download)

        full_length = len(train_dataset)
        train_length = int(self.train_rate * full_length)
        valid_length = full_length - train_length
        train_data, valid_data = random_split(train_dataset,[train_length,valid_length])

        cifar10_train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        cifar10_valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=True)

        self.cifar10_train_loader = cifar10_train_dataloader
        self.cifar10_valid_loader = cifar10_valid_dataloader
        return self.cifar10_train_loader, self.cifar10_valid_loader

    def Download_SubTrain_SubValid(self):
        train_dataset = datasets.CIFAR10(root=self.root, train=True, transform=self.tf, download=self.download)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[0:self.subtrain_length], indices[self.subtrain_length:(self.subtrain_length + self.subvalid_length)]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        cifar10_subtrain_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)
        cifar10_subvalid_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler)

        return cifar10_subtrain_dataloader, cifar10_subvalid_dataloader

    def Download_Test(self):
        test_dataset  = datasets.CIFAR10(root=self.root, train=False, transform=self.tf, download=self.download)
        cifar10_test_dataloader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=True)
        self.cifar10_test_loader = cifar10_test_dataloader
        return self.cifar10_test_loader

    def Img_show(self, img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def Plot_img(self, dataloader):
        # get some random training images
        dataiter = iter(dataloader)
        images, labels = dataiter.next()

        # show images
        self.Img_show(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))


if __name__=='__main__':
    cifar = Cifar10(batch_size=100)
    sub_train, sub_valid = cifar.Download_SubTrain_SubValid()
    # train_dataloader, valid_dataloader = cifar.Download_Train_Valid()
    # test_dataloader = cifar.Download_Test()
    # cifar.Plot_img(test_dataloader)

    # tf = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 三通道，三个均值，三个方差
    # ])
    #
    # cifar_traindata = datasets.CIFAR10(root='../Cifar10/cifar-10-python', train=True, transform=tf)
    # print(cifar_traindata)

    # mnist = datasets.MNIST(root='../Mnist', train=True, transform=tf)