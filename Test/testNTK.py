import sys
import os
dir=os.path.abspath("..")
sys.path.append(dir)

from newEvaluate.EvolveDataLoader import Evolve_DataLoader
from procedures import get_ntk_n
from newSetting.config import Config
from newEvaluate.ConstructNet import ConstructNet
from newEvaluate.TrainNet import TrainNet
from Tool.utils import cal_indi_parameters_and_flops
from newGA.Individual import Individual
import numpy as np
import pickle
import time
if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params

class EvaluateIndi():
    """评估个体（包括评估NTK以及acc）
    """
    def __init__(self):
        pass


    def evaluate_indi_by_ntk(self,indi):
        self.train_loader,self.valid_loader=Evolve_DataLoader(1,True).data_loader()
        cal_indi_parameters_and_flops(indi)
        """评估个体的适应度（ntk）
        """
        #TODO 降低ntk的波动
        net=ConstructNet(indi,10)
        net.to(params["device"])
        ntk=0.0
        num=10
        start=time.time()
        # for i in range(num):
        #     ntk+=get_ntk_n(self.train_loader, [net], recalbn=0, train_mode=True, num_batch=1)[0]
        # indi.ntk=round(ntk/num,3)
        print(indi.ntk)
        end=time.time()
        print("cost:{}".format(end-start))
        # indi.ntk=np.random.randint(100)
        


    def evaluate_indi_by_acc(self,indi):
        cal_indi_parameters_and_flops(indi)
        """评估个体的精度（acc）
        """
        pass

if __name__=="__main__":
    # evaluateIndi=EvaluateIndi()
    # path="../Output/indi_log/trained_individuals/"
    # indis=os.listdir(path)
    # for indi_name in indis:
    #     with open(os.path.join(path,indi_name),"rb") as f:
    #         indi=pickle.load(f)
    #     print(indi)
    #     for unit in indi.units:
    #         print(unit)
    #     print("batch_size:{}".format(params["ntk_batch_size"]))
    #     for _ in range(3):
    #         evaluateIndi.evaluate_indi_by_ntk(indi)
    # from newGA.Individual import Individual
    # from Tool.Log import Log
    # import pickle
    # params=Config.dev_params
    # print(params)
    # path="../Output/4.23/v4.23/trained_indis"
    # indis=os.listdir(path)
    # indis.sort()
    # # log=Log()
    # # log.info("")
    # # log.info("another 10 runs")
    # # trainnet=TrainNet(params,log)
    # for indi_file in indis:
    #     with open(os.path.join(path,indi_file),"rb") as file:
    #         indi=pickle.load(file)
    #     cal_indi_parameters_and_flops(indi)
    #     print(indi)
    #     # log.info(indi)
    #     for unit in indi.units[1:]:
    #         l=unit.block_amount
    #         unit.hasSENets=[True]*l
    #         print(unit)
    #     cal_indi_parameters_and_flops(indi)
    #     print(indi)
"""NTK 评估测试脚本（CIFAR）"""
