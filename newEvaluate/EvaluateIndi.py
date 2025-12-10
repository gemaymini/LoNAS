import sys
import os
import torch.nn as nn
dir=os.path.abspath("..")
sys.path.append(dir)

from newEvaluate.EvolveDataLoader_imagenet import Evolve_DataLoader
from procedures import get_ntk_n
from newSetting.config import Config
from newEvaluate.ConstructNet import ConstructNet
from newEvaluate.TrainNet_imagenet import TrainNet
from Tool.utils import cal_indi_parameters_and_flops
import numpy as np
import torch.distributed as dist
# dist.init_process_group(backend="nccl", init_method="tcp://172.18.65.193:20000",world_size=1,rank=0)
if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params

dataset_type=params["dataset"]
class EvaluateIndi():
    """评估个体（包括评估 NTK 以及 ACC）"""
    def __init__(self,log):
        self.log=log
        self.train_loader,self.valid_loader=Evolve_DataLoader(dataset_type,True).data_loader()
        print(len(self.train_loader))
        print(len(self.valid_loader))
        self.trainnet=TrainNet(params,self.log)

    def evaluate_indi_by_ntk(self,indi):
        """计算个体的 NTK 条件数，并写入到个体对象"""
        cal_indi_parameters_and_flops(indi)
        #TODO 降低ntk的波动
        device_ids = [0, 1]
        num_classes=1000
        net=ConstructNet(indi,num_classes)
        # TODO 单GPU
        # net.to(params["device"])
        net = nn.DataParallel(net, device_ids=device_ids)
        net.cuda()
        # net = nn.parallel.DistributedDataParallel(net,device_ids=[0,1])
        # net.to(device_ids[0])
        ntk=0.0
        num=10
        for i in range(num):
            ntk+=get_ntk_n(self.train_loader, [net], recalbn=0, train_mode=True, num_batch=1)[0]
        indi.ntk=round(ntk/num,3)
        # indi.ntk=np.random.randint(100)
        self.log.info("个体{}的适应度NTK为：{}".format(indi.id,indi.ntk))


    def evaluate_indi_by_acc(self,indi):
        """训练并记录个体的最佳 ACC（实际由 TrainNet 写入个体）"""
        cal_indi_parameters_and_flops(indi)
        self.trainnet.train(indi,self.train_loader,self.valid_loader)
        self.log.info(indi)

    
"""个体评估：NTK 条件数与训练准确率

说明
- evaluate_indi_by_ntk: 构建网络并用少量 batch 计算 NTK 条件数，作为适应度
- evaluate_indi_by_acc: 按配置训练若干 epoch，记录最佳 Top-1（或精度）
"""
