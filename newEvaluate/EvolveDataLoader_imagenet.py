import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import sys
import PIL
dir=os.path.abspath("..")
sys.path.append(dir)

from newSetting.config import Config
from Tool.randomerasing import RandomErasing

if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params


PATH="/home/cv_dataset/"
# 进化过程中使用的DataLoader
class Evolve_DataLoader():
    def __init__(self,type,ntk):
        """设置数据集路径与 batch 大小

        Args:
            type: 数据集类型（此处 ImageNet 固定使用 `ImageFolder`）
            ntk: 是否为 NTK 模式（决定 batch 大小）
        """
        self.type=type
        self.dataset_path=PATH+"ImageNet"
        self.length=16 if self.type==1 else 8
        print(self.dataset_path)
        self.batch_size=params["ntk_batch_size"] if ntk else params["acc_batch_size"]
    
    def data_process(self):
        """构建训练与验证的变换流水线"""
        transform_train=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, )
        ])
        print(transform_train)
        print("re!")
        transform_test=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        return (transform_train,transform_test)


    def data_loader(self):
        """返回训练与验证 DataLoader（使用 ImageFolder）"""
        transform_train,transform_test=self.data_process()
        train_data=torchvision.datasets.ImageFolder(root=os.path.join(self.dataset_path,"train"),transform=transform_train)
        valid_data=torchvision.datasets.ImageFolder(root=os.path.join(self.dataset_path,"val"),transform=transform_test)
        # train_sampler = t.utils.data.distributed.DistributedSampler(train_data)
        train_loader=t.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True, num_workers=32,pin_memory=True)
        test_loader=t.utils.data.DataLoader(valid_data,batch_size=self.batch_size,shuffle=True,num_workers=32,pin_memory=True)
        # train_loader=t.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=(train_sampler is None),sampler=train_sampler,pin_memory=False)
        # test_loader=t.utils.data.DataLoader(valid_data,batch_size=self.batch_size,shuffle=True)
        return (train_loader,test_loader)

if __name__=="__main__":
    
    dataLoader=Evolve_DataLoader(2,True)
    train_loader,test_loader=dataLoader.data_loader()
    print(len(train_loader))
    print(len(test_loader))
"""ImageNet 数据加载与增强（用于进化评估与训练）

模式
- ntk=True: 使用较小的 `ntk_batch_size` 以快速计算 NTK
- ntk=False: 使用 `acc_batch_size` 进行训练与验证

增强
- 训练：RandomResizedCrop(224) + HorizontalFlip + Normalize + RandomErasing
- 验证：Resize(256) + CenterCrop(224) + Normalize
"""
