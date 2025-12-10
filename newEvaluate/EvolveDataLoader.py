import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import sys
dir=os.path.abspath("..")
sys.path.append(dir)

from newSetting.config import Config
from Tool.cutout import Cutout

if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params


PATH="/home/zzh/Projects/Dataset/"
# 进化过程中使用的DataLoader
class Evolve_DataLoader():
    def __init__(self,type,ntk):
        """设置数据集

        Args:
            type (int): 1表示cifar10,2表示cifar100
            ntk (bool): True表示使用ntk的batch_size,False表示使用acc的batch_size
        """
        self.type=type
        self.dataset_path=PATH+"cifar10" if self.type==1 else PATH+"cifar100"
        self.length=16 if self.type==1 else 8
        print(self.dataset_path)
        self.batch_size=params["ntk_batch_size"] if ntk else params["acc_batch_size"]
    
    def data_process(self):
        transform_train=transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=self.length)
        ])
        print("cutout!")
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        return (transform_train,transform_test)


    def data_loader(self):
        transform_train,transform_test=self.data_process()
        if self.type==1:
            train_data=torchvision.datasets.CIFAR10(root=self.dataset_path,train=True,download=False,transform=transform_train)
            valid_data=torchvision.datasets.CIFAR10(root=self.dataset_path,train=False,download=False,transform=transform_test)
        else:
            train_data=torchvision.datasets.CIFAR100(root=self.dataset_path,train=True,download=False,transform=transform_train)
            valid_data=torchvision.datasets.CIFAR100(root=self.dataset_path,train=False,download=False,transform=transform_test)
        train_loader=t.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True, num_workers=2)
        test_loader=t.utils.data.DataLoader(valid_data,batch_size=self.batch_size,shuffle=True,num_workers=2)
        return (train_loader,test_loader)

if __name__=="__main__":
    
    dataLoader=Evolve_DataLoader(2,True)
    train_loader,test_loader=dataLoader.data_loader()
    print(len(train_loader))
    print(len(test_loader))
