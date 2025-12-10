import sys
import os

dir=os.path.abspath("..")
sys.path.append(dir)

from newGA.Individual import Individual
from newSetting.config import Config
import numpy as np
import hashlib
import copy

class Population():
    def __init__(self,params,gen_id):
        self.gen_id=gen_id #种群当前经历的进化代次
        self.indi_id=1 # 种群中最后加入的个体的编号
        self.pop_size=params["pop_size"]
        self.params=params
        self.individuals=[] # 种群中的个体列表

    def initialize(self):
        """种群的初始化
        """
        for _ in range(self.pop_size):
            ind_id="%02d%02d"%(self.gen_id,self.indi_id)
            indi=Individual(self.params,ind_id)
            indi.initialize()
            self.individuals.append(indi)
            self.indi_id+=1
    
    def add_offsprings_to_parents(self,offsprings):
        """将生成的子代加入到父代组成新的种群

        Args:
            offsprings (list[Individual]): 新生成的子代
        """
        for offspring in offsprings:
            indi=copy.deepcopy(offspring)
            indi_id="%02d%02d"%(self.gen_id,self.indi_id)
            indi.id=indi_id
            self.individuals.append(indi)
            self.indi_id+=1
        self.pop_size+=len(offsprings)
    
    def rename_indi_id(self):
        """重新对个体进行编号
        """
        self.indi_id=1
        for indi in self.individuals:
            indi.id="%02d%02d"%(self.gen_id,self.indi_id)
            self.indi_id+=1

    def cal_indi_spantime(self):
        """计算个体的寿命：indi.spantime=n表示indi个体在种群中存活的n代
        """
        for indi in self.individuals:
            indi.spantime+=1
        
if __name__=="__main__":
   pass
