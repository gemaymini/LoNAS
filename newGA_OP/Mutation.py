import sys
import os
dir=os.path.abspath("..")
sys.path.append(dir)

from newGA_OP.doMutation import doMutation
from newSetting.config import Config
from procedures import get_ntk_n
from newEvaluate.ConstructNet import ConstructNet
from newEvaluate.EvolveDataLoader_imagenet import Evolve_DataLoader
from newEvaluate.EvaluateIndi import EvaluateIndi
import numpy as np
import copy

if Config.profile=="dev":
    params=Config.dev_params
elif Config.profile=="pro":
    params=Config.pro_params
else:
    params=Config.test_params

class Mutation():
    def __init__(self,log,indis,gen_epoch):
        """变异操作

        Args:
            log (Log): 记录变异操作
            indis (list[Individual]): 种群中的个体
            gen_epochs (int): 当前进化的代次
        """
        self.log=log
        self.indis=indis
        self.gen_epoch=gen_epoch
        self.gen_division1=params["gen_division1"]
        self.gen_division2=params["gen_division2"]
        self.s=params["s"]
        self.t=params["ntk_t"] #一开始基于ntk进化个体
        self.evaluateIndi=EvaluateIndi(self.log)

    def process(self):
        """执行变异
        """
        # 如果是前gen_division1代基于适应度删除，个体更换的个数为3，gen_division1~gen_division2代次之间则基于寿命删除，个体更换的个数为1，gen_division2代次之后有事基于ntk删除，个体更换的个数为3，实现先区分年龄，然后增加多样性（探索），然后逐步收敛（开发）的搜索过程
        if self.gen_epoch>self.gen_division1:
            self.t=params["spantime_t"]
        if self.gen_epoch>self.gen_division2:
            self.t=params["ntk_t"]
        # 1.随机选取s个父代个体
        s_random_indexs=np.random.choice(list(range(len(self.indis))),self.s,replace=False)
        s_indis=[self.indis[i] for i in s_random_indexs]
        # 2.从s个父代个体中选取t个适应度最优的个体
        s_indis.sort(key=lambda indi:indi.ntk)
        t_selected_inids=[copy.deepcopy(s_indis[i]) for i in range(self.t)]
        # 赋予临时的id名，同时重置个体的spantime，ntk和acc
        for indi in t_selected_inids:
            indi.id="off_spring_{}".format(indi.id)
            indi.reset()
        # 3.对这t个个体执行变异操作
        domutation=doMutation(t_selected_inids,self.log)
        self.offsprings=domutation.do_mutation()

        # 重新计算变异个体的unit的block数量、senet数量、参数量和FLOPs，同时计算适应度
        for offspring in self.offsprings:
            offspring.cal_conv_length_and_senet()
            self.evaluateIndi.evaluate_indi_by_ntk(offspring)
        # 4.子代重新编号加入种群
        for  i in self.offsprings:
            print(i)
        for index,indi in enumerate(self.offsprings):
            indi.id=params["pop_size"]+index+1
        for offspring in self.offsprings:
            self.indis.append(offspring)
        # 所有个体重新编号
        indi_id=1
        for indi in self.indis:
            indi.id="%02d%02d"%(self.gen_epoch,indi_id)
            indi_id+=1
        
        # 5.删除后t个最差的个体，前gen_division1和gen_division2~max_gen代基于适应度删除，后面的代次则基于寿命删除
        if self.gen_epoch<=self.gen_division1 or self.gen_epoch>self.gen_division2:
            self.indis.sort(key=lambda indi:indi.ntk)  
        else:
            self.indis.sort(key=lambda indi:indi.spantime)
        for _ in range(self.t):
            self.indis.pop(-1)
        
        del domutation



if __name__=="__main__":
    from Tool.Log import Log
    from newGA.Individual import Individual
    from newSetting.config import Config
    params=Config.dev_params
    logger=Log()
    indis=[]
    for i in range(1,11):
        indi=Individual(params,i)
        indi.initialize()
        indi.ntk=np.random.randint(0,100)
        indis.append(indi)
    mutation=Mutation(logger,indis,0)
    mutation.process()