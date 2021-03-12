import torch.nn as nn
import pdb
from torchvision import transforms,datasets
from torch.utils.data import DataLoader,random_split
from PIL import Image
import numpy as np
"""
Mnist:
28*28,
train_data:60000
test_data: 10000
"""

class Mnist():
    def __init__(self, batch_size):
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

        self.root = '../Mnist'
        self.download = True
        self.batch_size = batch_size

        self.mnist_traindata = None
        self.mnist_validdata = None
        self.mnist_testdata = None

    def Download_Train_Valid(self):
        train_dataset = datasets.MNIST(root=self.root,train=True,transform=self.tf,download=self.download)

        full_length = len(train_dataset)
        train_length = int(0.9 * full_length)
        valid_length = full_length - train_length
        train_data, valid_data = random_split(train_dataset,[train_length,valid_length])

        mnist_train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        mnist_valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=True)

        self.mnist_traindata = mnist_train_dataloader
        self.mnist_validdata = mnist_valid_dataloader
        return self.mnist_traindata, self.mnist_validdata

    def Download_Test(self):
        test_dataset  = datasets.MNIST(root=self.root, train=False, transform=self.tf, download=self.download)
        mnist_test_dataloader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=True)
        self.mnist_testdata = mnist_test_dataloader
        return self.mnist_testdata

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.save('tt.png')
    # pil_img.show()

if __name__=='__main__':
    batch_size = 100
    Mnist = Mnist(batch_size)
    traindata,validdata = Mnist.Download_Train_Valid()
    # for i, data in enumerate(traindata, 0):
    #     images, labels = data  # labels是具体的数值
    #     images, labels = images.cuda(), labels.cuda()
    #     images = images.view(images.size(0), -1)  # 把图片拉平
    #
    #     img0 = images[0].cpu()
    #     img0 = img0.reshape(28,28)
    #
    #     # pdb.set_trace()
    #     img_show(img0)
