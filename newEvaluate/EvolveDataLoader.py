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
        """设置数据集基础信息与 batch 大小"""
        self.type=type
        self.dataset_path=PATH+"cifar10" if self.type==1 else PATH+"cifar100"
        self.length=16 if self.type==1 else 8
        print(self.dataset_path)
        self.batch_size=params["ntk_batch_size"] if ntk else params["acc_batch_size"]
    
    def data_process(self):
        """构建训练与验证的变换流水线（含 Cutout）"""
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
        """返回 CIFAR10/100 的训练与验证 DataLoader"""
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
"""CIFAR 数据加载与增强（Cutout 版本）

模式
- ntk=True: 使用较小的 `ntk_batch_size` 以快速评估 NTK
- ntk=False: 使用 `acc_batch_size` 训练与验证

增强
- 训练：RandomCrop(32, padding=4) + HorizontalFlip + Normalize + Cutout
- 验证：Normalize
"""
